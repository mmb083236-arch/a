#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <objbase.h>
#include <wincodec.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr char kModelMagic[] = "IMG24001";
constexpr std::uint32_t kOutputWidth = 1920;
constexpr std::uint32_t kOutputHeight = 1080;
constexpr std::uint32_t kGridWidth = 32;
constexpr std::uint32_t kGridHeight = 24;
constexpr std::size_t kInputSize = 256;
constexpr std::size_t kHiddenSize = 32;
constexpr std::size_t kOutputSize =
    static_cast<std::size_t>(kGridWidth) * kGridHeight * 3;
constexpr std::size_t kEpochs = 1000;
constexpr float kLearningRate = 0.5F;
constexpr std::uint64_t kMaxImageBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxImagePixels = 50'000'000ULL;

struct Pixel {
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

struct TrainingSample {
    std::vector<float> features;
    std::vector<float> pixels;
};

struct Model {
    std::vector<float> input_weights =
        std::vector<float>(kInputSize * kHiddenSize);
    std::vector<float> hidden_bias = std::vector<float>(kHiddenSize, 0.0F);
    std::vector<float> output_weights =
        std::vector<float>(kHiddenSize * kOutputSize);
    std::vector<float> output_bias = std::vector<float>(kOutputSize, 0.0F);
};

void write_u16(std::ostream& out, std::uint16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}

void write_u32(std::ostream& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.put(static_cast<char>((value >> shift) & 0xff));
    }
}

#ifndef _WIN32
std::uint16_t read_u16(const std::vector<std::uint8_t>& data, std::size_t at) {
    if (at + 2 > data.size()) {
        throw std::runtime_error("truncated BMP header");
    }
    return static_cast<std::uint16_t>(data[at]) |
           static_cast<std::uint16_t>(data[at + 1] << 8);
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& data, std::size_t at) {
    if (at + 4 > data.size()) {
        throw std::runtime_error("truncated BMP header");
    }
    return static_cast<std::uint32_t>(data[at]) |
           (static_cast<std::uint32_t>(data[at + 1]) << 8) |
           (static_cast<std::uint32_t>(data[at + 2]) << 16) |
           (static_cast<std::uint32_t>(data[at + 3]) << 24);
}

std::vector<Pixel> read_bmp(const fs::path& path, std::uint32_t& width,
                            std::uint32_t& height) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open image: " + path.string());
    }
    const std::streamoff length = static_cast<std::streamoff>(file.tellg());
    if (length < 54 || static_cast<std::uint64_t>(length) > kMaxImageBytes) {
        throw std::runtime_error("BMP is too small, too large, or unreadable: " +
                                 path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!file || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::runtime_error("not a valid BMP image: " + path.string());
    }

    const std::uint32_t pixel_offset = read_u32(bytes, 10);
    const std::uint32_t dib_size = read_u32(bytes, 14);
    const std::int32_t signed_width =
        static_cast<std::int32_t>(read_u32(bytes, 18));
    const std::int32_t signed_height =
        static_cast<std::int32_t>(read_u32(bytes, 22));
    const std::uint16_t planes = read_u16(bytes, 26);
    const std::uint16_t bits_per_pixel = read_u16(bytes, 28);
    const std::uint32_t compression = read_u32(bytes, 30);

    if (dib_size < 40 || signed_width <= 0 || signed_height == 0 ||
        signed_height == std::numeric_limits<std::int32_t>::min() ||
        planes != 1 || bits_per_pixel != 24 || compression != 0) {
        throw std::runtime_error(
            "only uncompressed 24-bit BMP images are supported: " + path.string());
    }
    width = static_cast<std::uint32_t>(signed_width);
    height = static_cast<std::uint32_t>(
        signed_height < 0 ? -signed_height : signed_height);
    if (width > 16384 || height > 16384 ||
        static_cast<std::uint64_t>(width) * height > kMaxImagePixels) {
        throw std::runtime_error("BMP dimensions are too large: " + path.string());
    }

    const std::uint64_t row_bytes =
        (static_cast<std::uint64_t>(width) * 3 + 3) & ~std::uint64_t{3};
    const std::uint64_t image_end =
        static_cast<std::uint64_t>(pixel_offset) + row_bytes * height;
    if (pixel_offset < 14 + dib_size || image_end > bytes.size()) {
        throw std::runtime_error("BMP pixel data is truncated: " + path.string());
    }

    std::vector<Pixel> pixels(static_cast<std::size_t>(width) * height);
    const bool top_down = signed_height < 0;
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t source_y = top_down ? y : height - 1 - y;
        const std::size_t row =
            pixel_offset + static_cast<std::size_t>(source_y * row_bytes);
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t at = row + static_cast<std::size_t>(x) * 3;
            pixels[static_cast<std::size_t>(y) * width + x] =
                Pixel{bytes[at + 2], bytes[at + 1], bytes[at]};
        }
    }
    return pixels;
}
#else
template <typename Interface>
class ComOwner {
public:
    ComOwner() = default;
    ComOwner(const ComOwner&) = delete;
    ComOwner& operator=(const ComOwner&) = delete;

    ~ComOwner() {
        if (value_ != nullptr) {
            value_->Release();
        }
    }

    Interface** put() {
        if (value_ != nullptr) {
            value_->Release();
            value_ = nullptr;
        }
        return &value_;
    }

    Interface* get() const {
        return value_;
    }

private:
    Interface* value_ = nullptr;
};

class ComInitialization {
public:
    ComInitialization() {
        const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(result) && result != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("could not initialize Windows COM for image decoding");
        }
        initialized_ = SUCCEEDED(result);
    }

    ~ComInitialization() {
        if (initialized_) {
            CoUninitialize();
        }
    }

    ComInitialization(const ComInitialization&) = delete;
    ComInitialization& operator=(const ComInitialization&) = delete;

private:
    bool initialized_ = false;
};

std::vector<Pixel> read_image(const fs::path& path, std::uint32_t& width,
                              std::uint32_t& height) {
    std::error_code file_error;
    const std::uint64_t file_size = fs::file_size(path, file_error);
    if (file_error || file_size == 0 || file_size > kMaxImageBytes) {
        throw std::runtime_error("image is missing, empty, or too large: " +
                                 path.string());
    }

    ComInitialization com;
    ComOwner<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                     CLSCTX_INPROC_SERVER,
                                     IID_PPV_ARGS(factory.put()));
    if (FAILED(result)) {
        throw std::runtime_error("could not create the Windows image decoder");
    }

    ComOwner<IWICBitmapDecoder> decoder;
    result = factory.get()->CreateDecoderFromFilename(
        path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad,
        decoder.put());
    if (FAILED(result)) {
        throw std::runtime_error("Windows could not decode image: " +
                                 path.string());
    }

    ComOwner<IWICBitmapFrameDecode> frame;
    result = decoder.get()->GetFrame(0, frame.put());
    if (FAILED(result)) {
        throw std::runtime_error("could not read the first image frame: " +
                                 path.string());
    }

    result = frame.get()->GetSize(&width, &height);
    if (FAILED(result) || width == 0 || height == 0 || width > 16384 ||
        height > 16384 ||
        static_cast<std::uint64_t>(width) * height > kMaxImagePixels) {
        throw std::runtime_error("image dimensions are invalid or too large: " +
                                 path.string());
    }

    ComOwner<IWICFormatConverter> converter;
    result = factory.get()->CreateFormatConverter(converter.put());
    if (FAILED(result)) {
        throw std::runtime_error("could not create an image format converter");
    }
    result = converter.get()->Initialize(
        frame.get(), GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone,
        nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(result)) {
        throw std::runtime_error("could not convert image pixels: " +
                                 path.string());
    }

    const std::uint64_t stride = static_cast<std::uint64_t>(width) * 3;
    const std::uint64_t buffer_size = stride * height;
    if (buffer_size > kMaxImageBytes ||
        buffer_size > std::numeric_limits<UINT>::max()) {
        throw std::runtime_error("decoded image is too large: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(buffer_size));
    result = converter.get()->CopyPixels(
        nullptr, static_cast<UINT>(stride), static_cast<UINT>(buffer_size),
        bytes.data());
    if (FAILED(result)) {
        throw std::runtime_error("could not read image pixels: " + path.string());
    }

    std::vector<Pixel> pixels(static_cast<std::size_t>(width) * height);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = Pixel{bytes[i * 3 + 2], bytes[i * 3 + 1], bytes[i * 3]};
    }
    return pixels;
}
#endif

std::vector<float> resize_for_training(const std::vector<Pixel>& image,
                                       std::uint32_t width,
                                       std::uint32_t height) {
    std::vector<float> result(kOutputSize);
    for (std::uint32_t y = 0; y < kGridHeight; ++y) {
        const double top =
            static_cast<double>(y) * height / kGridHeight;
        const double bottom =
            static_cast<double>(y + 1) * height / kGridHeight;
        const std::uint32_t first_y = static_cast<std::uint32_t>(top);
        const std::uint32_t last_y =
            std::min(height, static_cast<std::uint32_t>(std::ceil(bottom)));
        for (std::uint32_t x = 0; x < kGridWidth; ++x) {
            const double left =
                static_cast<double>(x) * width / kGridWidth;
            const double right =
                static_cast<double>(x + 1) * width / kGridWidth;
            const std::uint32_t first_x = static_cast<std::uint32_t>(left);
            const std::uint32_t last_x =
                std::min(width, static_cast<std::uint32_t>(std::ceil(right)));
            double red = 0.0;
            double green = 0.0;
            double blue = 0.0;
            double total_weight = 0.0;

            for (std::uint32_t source_y = first_y; source_y < last_y;
                 ++source_y) {
                const double vertical_weight =
                    std::min(bottom, static_cast<double>(source_y + 1)) -
                    std::max(top, static_cast<double>(source_y));
                for (std::uint32_t source_x = first_x; source_x < last_x;
                     ++source_x) {
                    const double horizontal_weight =
                        std::min(right, static_cast<double>(source_x + 1)) -
                        std::max(left, static_cast<double>(source_x));
                    const double weight = horizontal_weight * vertical_weight;
                    const Pixel& pixel =
                        image[static_cast<std::size_t>(source_y) * width +
                              source_x];
                    red += pixel.r * weight;
                    green += pixel.g * weight;
                    blue += pixel.b * weight;
                    total_weight += weight;
                }
            }
            const std::size_t at =
                (static_cast<std::size_t>(y) * kGridWidth + x) * 3;
            result[at] =
                static_cast<float>(red / total_weight / 127.5 - 1.0);
            result[at + 1] =
                static_cast<float>(green / total_weight / 127.5 - 1.0);
            result[at + 2] =
                static_cast<float>(blue / total_weight / 127.5 - 1.0);
        }
    }
    return result;
}

std::string lowercase(std::string text) {
    for (char& ch : text) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (byte >= 'A' && byte <= 'Z') {
            ch = static_cast<char>(byte - 'A' + 'a');
        }
    }
    return text;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char ch : lowercase(text)) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
            current.push_back(static_cast<char>(ch));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

struct SafetyResult {
    bool allowed;
    std::string reason;
};

SafetyResult check_safety(const std::string& text) {
    const std::vector<std::string> tokens = tokenize(text);
    if (tokens.empty()) {
        return {false, "description is empty"};
    }
    static const std::unordered_set<std::string> blocked = {
        "porn",       "pornography", "nude",      "nudity",
        "naked",      "erotic",      "sexual",    "genitals",
        "genital",    "rape",        "gore",      "gory",
        "dismember",  "decapitate",  "beheading", "torture",
        "suicide",    "suicidal",    "selfharm",  "kill",
        "killing",    "murder",      "slaughter", "abuse",
        "assault",    "execution",   "mutilation"};
    for (const std::string& token : tokens) {
        if (blocked.find(token) != blocked.end()) {
            return {false, "description contains a term blocked by the local safety filter"};
        }
    }
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
        if ((tokens[i] == "self" && tokens[i + 1] == "harm") ||
            (tokens[i] == "child" && tokens[i + 1] == "sexual") ||
            (tokens[i] == "child" && tokens[i + 1] == "porn") ||
            (tokens[i] == "minor" && tokens[i + 1] == "sexual") ||
            (tokens[i] == "underage" && tokens[i + 1] == "sexual")) {
            return {false, "description contains a term blocked by the local safety filter"};
        }
    }
    return {true, {}};
}

std::vector<float> text_features(const std::string& text) {
    const SafetyResult safety = check_safety(text);
    if (!safety.allowed) {
        throw std::runtime_error(safety.reason);
    }
    const std::vector<std::string> tokens = tokenize(text);
    std::vector<float> features(kInputSize, 0.0F);
    for (const std::string& token : tokens) {
        std::uint32_t hash = 2166136261U;
        for (unsigned char byte : token) {
            hash = (hash ^ byte) * 16777619U;
        }
        const std::size_t index = hash % kInputSize;
        const float sign = (hash & 0x80000000U) == 0 ? 1.0F : -1.0F;
        features[index] += sign;
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(tokens.size()));
    for (float& feature : features) {
        feature *= scale;
    }
    return features;
}

bool is_basename(const std::string& name) {
    const fs::path path(name);
    return !name.empty() && path.filename() == path &&
           name.find_first_of("/\\:") == std::string::npos;
}

bool has_supported_image_extension(const std::string& filename) {
    const std::string extension =
        lowercase(fs::path(filename).extension().string());
#ifdef _WIN32
    return extension == ".bmp" || extension == ".jpg" ||
           extension == ".jpeg" || extension == ".png" ||
           extension == ".gif" || extension == ".tif" ||
           extension == ".tiff";
#else
    return extension == ".bmp";
#endif
}

std::string trim(std::string text) {
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<TrainingSample> load_training_samples() {
    const fs::path train_dir("train");
    fs::create_directories(train_dir);
    const fs::path captions_path = train_dir / "dis.discr";
    if (!fs::exists(captions_path)) {
        std::ofstream example(captions_path);
        example << "# One image-description pair per line:\n"
                   "# cat.jpg: a cat sitting in green grass\n";
        throw std::runtime_error(
            "created train/dis.discr; add supported images to train/, write "
            "their descriptions in that file, then run train again");
    }

    std::ifstream captions(captions_path);
    if (!captions) {
        throw std::runtime_error("cannot read train/dis.discr");
    }
    std::vector<TrainingSample> samples;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(captions, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.front() == '{' && line.back() == '}') {
            line = trim(line.substr(1, line.size() - 2));
        }
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("line " + std::to_string(line_number) +
                                     " must use an image filename: description");
        }
        const std::string filename = trim(line.substr(0, separator));
        const std::string description = trim(line.substr(separator + 1));
        if (!is_basename(filename) ||
            !has_supported_image_extension(filename) ||
            description.empty()) {
            throw std::runtime_error(
                "line " + std::to_string(line_number) +
                " needs a supported image filename directly inside train/ and a description");
        }
        const SafetyResult safety = check_safety(description);
        if (!safety.allowed) {
            throw std::runtime_error("safety check rejected line " +
                                     std::to_string(line_number) + ": " +
                                     safety.reason);
        }
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        const std::vector<Pixel> image =
#ifdef _WIN32
            read_image(train_dir / filename, width, height);
#else
            read_bmp(train_dir / filename, width, height);
#endif
        samples.push_back(
            TrainingSample{text_features(description),
                           resize_for_training(image, width, height)});
    }
    if (!captions.eof()) {
        throw std::runtime_error("failed while reading train/dis.discr");
    }
    if (samples.empty()) {
        throw std::runtime_error(
            "train/dis.discr contains no image-description entries");
    }
    return samples;
}

void initialize_model(Model& model) {
    std::mt19937 generator(0xC0FFEEU);
    const float input_limit =
        std::sqrt(6.0F / static_cast<float>(kInputSize + kHiddenSize));
    const float output_limit =
        std::sqrt(6.0F / static_cast<float>(kHiddenSize + kOutputSize));
    std::uniform_real_distribution<float> input_distribution(-input_limit,
                                                              input_limit);
    std::uniform_real_distribution<float> output_distribution(-output_limit,
                                                               output_limit);
    for (float& weight : model.input_weights) {
        weight = input_distribution(generator);
    }
    for (float& weight : model.output_weights) {
        weight = output_distribution(generator);
    }
}

std::vector<float> forward(const Model& model, const std::vector<float>& features,
                           std::vector<float>& hidden) {
    hidden.assign(kHiddenSize, 0.0F);
    for (std::size_t h = 0; h < kHiddenSize; ++h) {
        float sum = model.hidden_bias[h];
        const std::size_t row = h * kInputSize;
        for (std::size_t i = 0; i < kInputSize; ++i) {
            sum += model.input_weights[row + i] * features[i];
        }
        hidden[h] = std::tanh(sum);
    }
    std::vector<float> output(kOutputSize, 0.0F);
    for (std::size_t o = 0; o < kOutputSize; ++o) {
        float sum = model.output_bias[o];
        const std::size_t row = o * kHiddenSize;
        for (std::size_t h = 0; h < kHiddenSize; ++h) {
            sum += model.output_weights[row + h] * hidden[h];
        }
        output[o] = std::tanh(sum);
    }
    return output;
}

float train_sample(Model& model, const TrainingSample& sample) {
    std::vector<float> hidden;
    const std::vector<float> output = forward(model, sample.features, hidden);
    std::vector<float> hidden_gradient(kHiddenSize, 0.0F);
    std::vector<float> output_gradient(kOutputSize, 0.0F);
    float loss = 0.0F;

    for (std::size_t o = 0; o < kOutputSize; ++o) {
        const float difference = output[o] - sample.pixels[o];
        loss += difference * difference;
        output_gradient[o] =
            2.0F * difference * (1.0F - output[o] * output[o]) /
            static_cast<float>(kOutputSize);
        const std::size_t row = o * kHiddenSize;
        for (std::size_t h = 0; h < kHiddenSize; ++h) {
            hidden_gradient[h] += model.output_weights[row + h] * output_gradient[o];
        }
    }
    for (std::size_t o = 0; o < kOutputSize; ++o) {
        const std::size_t row = o * kHiddenSize;
        for (std::size_t h = 0; h < kHiddenSize; ++h) {
            model.output_weights[row + h] -=
                kLearningRate * output_gradient[o] * hidden[h];
        }
        model.output_bias[o] -= kLearningRate * output_gradient[o];
    }
    for (std::size_t h = 0; h < kHiddenSize; ++h) {
        const float gradient =
            hidden_gradient[h] * (1.0F - hidden[h] * hidden[h]);
        const std::size_t row = h * kInputSize;
        for (std::size_t i = 0; i < kInputSize; ++i) {
            model.input_weights[row + i] -=
                kLearningRate * gradient * sample.features[i];
        }
        model.hidden_bias[h] -= kLearningRate * gradient;
    }
    return loss / static_cast<float>(kOutputSize);
}

void save_model(const Model& model, const fs::path& path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot create model file: " + path.string());
    }
    file.write(kModelMagic, sizeof(kModelMagic) - 1);
    const auto write_values = [&file](const std::vector<float>& values) {
        for (float value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("model contains a non-finite value");
            }
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    };
    write_values(model.input_weights);
    write_values(model.hidden_bias);
    write_values(model.output_weights);
    write_values(model.output_bias);
    if (!file) {
        throw std::runtime_error("failed to write model file");
    }
}

void load_values(std::ifstream& file, std::vector<float>& values) {
    for (float& value : values) {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file || !std::isfinite(value)) {
            throw std::runtime_error("model file is truncated or invalid");
        }
    }
}

Model load_model(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("model.bin not found; run `image_ai train` first");
    }
    std::array<char, sizeof(kModelMagic) - 1> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || !std::equal(magic.begin(), magic.end(), kModelMagic)) {
        throw std::runtime_error("model.bin has an unknown or invalid format");
    }
    Model model;
    load_values(file, model.input_weights);
    load_values(file, model.hidden_bias);
    load_values(file, model.output_weights);
    load_values(file, model.output_bias);
    char trailing = 0;
    if (file.get(trailing)) {
        throw std::runtime_error("model.bin has unexpected trailing data");
    }
    if (!file.eof()) {
        throw std::runtime_error("failed while reading model.bin");
    }
    return model;
}

void train() {
    const std::vector<TrainingSample> samples = load_training_samples();
    std::cout << "Training from " << samples.size() << " image(s)...\n";
    Model model;
    initialize_model(model);
    std::vector<std::size_t> order(samples.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::mt19937 generator(0x1234ABCDU);
    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), generator);
        float loss = 0.0F;
        for (std::size_t index : order) {
            loss += train_sample(model, samples[index]);
        }
        if ((epoch + 1) % 100 == 0 || epoch + 1 == kEpochs) {
            std::cout << "Epoch " << (epoch + 1) << '/' << kEpochs
                      << " - average loss " << std::fixed << std::setprecision(4)
                      << loss / static_cast<float>(samples.size()) << '\n';
        }
    }
    save_model(model, "model.bin");
    std::cout << "Training complete. Model saved to model.bin.\n";
}

Pixel sample_grid(const std::vector<float>& values, float x, float y) {
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(kGridWidth - 1));
    const float clamped_y =
        std::clamp(y, 0.0F, static_cast<float>(kGridHeight - 1));
    const std::uint32_t x0 = static_cast<std::uint32_t>(clamped_x);
    const std::uint32_t y0 = static_cast<std::uint32_t>(clamped_y);
    const std::uint32_t x1 = std::min(x0 + 1, kGridWidth - 1);
    const std::uint32_t y1 = std::min(y0 + 1, kGridHeight - 1);
    const float dx = clamped_x - static_cast<float>(x0);
    const float dy = clamped_y - static_cast<float>(y0);
    std::array<std::uint8_t, 3> channels{};
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        const auto at = [&values, channel](std::uint32_t gx, std::uint32_t gy) {
            const std::size_t index =
                (static_cast<std::size_t>(gy) * kGridWidth + gx) * 3 + channel;
            return values[index];
        };
        const float top = at(x0, y0) * (1.0F - dx) + at(x1, y0) * dx;
        const float bottom = at(x0, y1) * (1.0F - dx) + at(x1, y1) * dx;
        const float color = top * (1.0F - dy) + bottom * dy;
        channels[channel] = static_cast<std::uint8_t>(
            std::lround((std::clamp(color, -1.0F, 1.0F) + 1.0F) * 127.5F));
    }
    return Pixel{channels[0], channels[1], channels[2]};
}

void write_bmp(const fs::path& path, const std::vector<float>& values) {
    constexpr std::uint32_t row_bytes = kOutputWidth * 3;
    constexpr std::uint32_t pixel_bytes = row_bytes * kOutputHeight;
    constexpr std::uint32_t file_bytes = 54 + pixel_bytes;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot create image file: " + path.string());
    }
    file.put('B');
    file.put('M');
    write_u32(file, file_bytes);
    write_u16(file, 0);
    write_u16(file, 0);
    write_u32(file, 54);
    write_u32(file, 40);
    write_u32(file, kOutputWidth);
    write_u32(file, kOutputHeight);
    write_u16(file, 1);
    write_u16(file, 24);
    write_u32(file, 0);
    write_u32(file, pixel_bytes);
    write_u32(file, 2835);
    write_u32(file, 2835);
    write_u32(file, 0);
    write_u32(file, 0);

    for (std::uint32_t row = 0; row < kOutputHeight; ++row) {
        const std::uint32_t y = kOutputHeight - 1 - row;
        const float gy = static_cast<float>(y) *
                         static_cast<float>(kGridHeight - 1) /
                         static_cast<float>(kOutputHeight - 1);
        for (std::uint32_t x = 0; x < kOutputWidth; ++x) {
            const float gx = static_cast<float>(x) *
                             static_cast<float>(kGridWidth - 1) /
                             static_cast<float>(kOutputWidth - 1);
            const Pixel pixel = sample_grid(values, gx, gy);
            file.put(static_cast<char>(pixel.b));
            file.put(static_cast<char>(pixel.g));
            file.put(static_cast<char>(pixel.r));
        }
    }
    if (!file) {
        throw std::runtime_error("failed while writing image: " + path.string());
    }
}

void run(const std::string& description, const fs::path& output_path) {
    const SafetyResult safety = check_safety(description);
    if (!safety.allowed) {
        throw std::runtime_error("safety check rejected prompt: " + safety.reason);
    }
    const Model model = load_model("model.bin");
    std::vector<float> hidden;
    const std::vector<float> result =
        forward(model, text_features(description), hidden);
    write_bmp(output_path, result);
    std::cout << "Saved 1920x1080 BMP to " << output_path.string() << '\n';
}

void print_help() {
    std::cout << "Local image AI (C++17, no third-party libraries)\n\n"
                 "Commands:\n"
                 "  image_ai train\n"
                 "  image_ai run [description] [--output image.bmp]\n"
                 "  image_ai                 interactive command prompt\n\n"
#ifdef _WIN32
                 "Place BMP, JPEG, PNG, GIF, or TIFF images and train/dis.discr in train/.\n"
#else
                 "Place 24-bit uncompressed BMP images and train/dis.discr in train/.\n"
#endif
                 "Safety checks scan training captions and generation prompts locally.\n";
}

void interactive() {
    std::cout << "Local image AI. Type train, run, help, or quit.\n";
    std::string command;
    while (std::cout << "> " && std::getline(std::cin, command)) {
        command = trim(command);
        if (command == "quit" || command == "exit") {
            return;
        }
        try {
            if (command == "help") {
                print_help();
            } else if (command == "train") {
                train();
            } else if (command == "run") {
                std::cout << "Describe the image: ";
                std::string description;
                if (!std::getline(std::cin, description)) {
                    return;
                }
                run(description, "generated.bmp");
            } else if (!command.empty()) {
                std::cout << "Unknown command. Type train, run, help, or quit.\n";
            }
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
        }
    }
}

void dispatch(int argc, char* argv[]) {
    if (argc == 1) {
        interactive();
        return;
    }
    const std::string command = argv[1];
    if (command == "help" || command == "--help" || command == "-h") {
        print_help();
    } else if (command == "train") {
        if (argc != 2) {
            throw std::runtime_error("usage: image_ai train");
        }
        train();
    } else if (command == "run") {
        std::string description;
        fs::path output_path = "generated.bmp";
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--output") {
                if (++i >= argc) {
                    throw std::runtime_error("--output requires a filename");
                }
                output_path = argv[i];
            } else {
                if (!description.empty()) {
                    description.push_back(' ');
                }
                description += argv[i];
            }
        }
        if (description.empty()) {
            std::cout << "Describe the image: ";
            if (!std::getline(std::cin, description)) {
                throw std::runtime_error("could not read image description");
            }
        }
        if (lowercase(output_path.extension().string()) != ".bmp") {
            throw std::runtime_error(
                "this dependency-free build writes BMP files; use an .bmp output name");
        }
        run(description, output_path);
    } else {
        throw std::runtime_error("unknown command: " + command +
                                 ". Use `image_ai help`.");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        dispatch(argc, argv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
