# Local image AI (C++17)

This is a small educational image-generation program built in C++17. The neural
network, training loop, local model format, and text safety checks are
implemented in `main.cpp`. On Windows, image decoding uses the built-in Windows
Imaging Component (WIC); it does not require a third-party image library.

## Build

With CMake and a C++17 compiler installed:

```text
cmake -S . -B build
cmake --build build --config Release
```

The executable is `build/image_ai` on Linux/macOS or
`build/Release/image_ai.exe` with a typical Windows multi-configuration
generator. You can also compile directly with a C++17 compiler. On Windows with MSVC, link
the system WIC and COM libraries:

```text
cl /std:c++17 /EHsc main.cpp windowscodecs.lib ole32.lib
```

## Add training images

On Windows, put JPEG (`.jpg`, `.jpeg`), PNG, BMP, GIF, or TIFF (`.tif`,
`.tiff`) images directly in `train/`. WIC decodes the image format and converts
it to RGB pixels for training. Animated files use their first frame. On
non-Windows builds, only 24-bit uncompressed BMP is currently supported.
The trainer area-averages the full-resolution source pixels into the network's
32×24 training grid, which preserves representative colors better than
sampling a single pixel. Generated BMP files are 1920×1080 (1080p), but this
only enlarges the coarse grid; it does not add fine image detail.

Add one image and description per line to `train/dis.discr`:

```text
cat.jpg: an orange cat sitting in green grass
lake.png: a blue lake surrounded by mountains
```

Lines may also use braces, such as `{cat.bmp: an orange cat}`. Start with `#`
for a comment. Run `image_ai train` (or start `image_ai` and enter `train`) to
train from scratch and save `model.bin` in the current directory.

## Generate an image

Run `image_ai run`, then type a prompt, or provide one as arguments:

```text
image_ai run a blue lake surrounded by mountains
image_ai run a cat in grass --output cat.bmp
```

The default image is `generated.bmp`. Output is 1920×1080 pixels (1080p). Start
the program without arguments for an interactive prompt with `train`, `run`,
`help`, and `quit` commands.

## Local safety checks and limitations

Before training, every caption is checked. Before generation, each prompt is
checked too. The offline safety filter blocks a short list of explicit sexual,
graphic-violence, self-harm, and abuse-related words/phrases. Rejected training
data stops training with a line number; a rejected generation prompt does not
produce an image. No prompts, images, or model data are sent over the network.

This keyword filter is deliberately simple: it can miss unsafe wording and can
reject benign wording that includes a blocked term. It is not a comprehensive
safety classifier, and this program does not inspect generated pixels for
unsafe content. Review training images and outputs yourself.

The model has a single hidden layer and learns a coarse 32×24 color grid from
caption-image pairs, then scales that grid to 1920×1080. It is not comparable
to a large pretrained image generator: expect abstract, blurry results,
especially with a small training set. A model is always trained from scratch
when `train` is run.
