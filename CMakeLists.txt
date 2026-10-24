cmake_minimum_required(VERSION 3.16)
project(local_image_ai LANGUAGES CXX)

add_executable(image_ai main.cpp)
target_compile_features(image_ai PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(image_ai PRIVATE /W4 /permissive-)
    target_link_libraries(image_ai PRIVATE windowscodecs ole32)
else()
    target_compile_options(image_ai PRIVATE -Wall -Wextra -Wpedantic)
endif()
