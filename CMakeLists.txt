cmake_minimum_required(VERSION 3.1)
project(sgemm CUDA )

set(CMAKE_CUDA_STANDARD 14)

add_executable(sgemm main.cu)
target_link_libraries(sgemm -lcublas)
set_target_properties(sgemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)