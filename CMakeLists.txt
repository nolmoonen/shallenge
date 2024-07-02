cmake_minimum_required(VERSION 3.18)

# 3.17 FindCUDAToolkit
# 3.18 CMAKE_CUDA_ARCHITECTURES
project(shallenge LANGUAGES CUDA)

add_executable(shallenge main.cu)
target_link_directories(shallenge PRIVATE CUDA::cudart_static)
target_compile_options(shallenge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math --extended-lambda --extra-device-vectorization>)
