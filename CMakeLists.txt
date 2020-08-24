cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
project(
  cuda_tests
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")

set(CMAKE_CXX_STANDARD 17)
set(CUDA_ROOT /opt/cuda)
set(CUDA_INC_DIR ${CUDA_ROOT}/include)

find_library(
  _CUDA_LIB cudart
  HINTS ${CUDA_ROOT}
  PATH_SUFFIXES lib lib64)
set(CUDA_LIB ${_CUDA_LIB})

include_directories(${CUDA_INC_DIR})
add_executable(malloc_test malloc_test.cpp)
set_property(TARGET malloc_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET malloc_test PROPERTY CUDA_STANDARD 11)
target_link_libraries(malloc_test ${CUDA_LIB})

add_executable(cooperative_group_test cooperative_group_test.cu)
set_property(TARGET cooperative_group_test PROPERTY CUDA_SEPARABLE_COMPILATION
                                                    ON)
set_property(TARGET cooperative_group_test PROPERTY CUDA_STANDARD 11)
target_link_libraries(cooperative_group_test ${CUDA_LIB})

add_executable(device_array_test device_array_test.cu)
set_property(TARGET device_array_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET device_array_test PROPERTY CUDA_STANDARD 11)
target_link_libraries(device_array_test ${CUDA_LIB})

add_executable(ballot_sync_test ballot_sync_test.cu)
set_property(TARGET ballot_sync_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ballot_sync_test PROPERTY CUDA_STANDARD 11)
target_link_libraries(ballot_sync_test ${CUDA_LIB})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
