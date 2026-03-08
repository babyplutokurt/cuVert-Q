cmake_minimum_required(VERSION 3.24)

# ============================================================
# CUDA Toolkit Configuration
# Usage: -DCUDA_TOOLKIT_PATH=/usr/local/cuda-12.8
# ============================================================
set(CUDA_TOOLKIT_PATH "" CACHE PATH "Path to CUDA Toolkit (e.g. /usr/local/cuda-12.8)")

if(CUDA_TOOLKIT_PATH)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_PATH}/bin/nvcc")
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_PATH}")
endif()

project(GPUFastQ VERSION 0.1.0 LANGUAGES C CXX CUDA)

# GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
# ============================================================
# nvcomp Configuration
# Usage: -DNVCOMP_ROOT=/path/to/nvcomp
# Auto-detects from pip if not specified
# ============================================================
set(NVCOMP_ROOT "" CACHE PATH "Path to nvcomp installation root")

if(NOT NVCOMP_ROOT)
    # Try python3 first, then python
    foreach(_python_cmd python3 python)
        execute_process(
            COMMAND ${_python_cmd} -c "import nvidia.libnvcomp; import os; print(os.path.dirname(nvidia.libnvcomp.__file__))"
            OUTPUT_VARIABLE NVCOMP_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _python_result
        )
        if(_python_result EQUAL 0 AND NVCOMP_ROOT)
            break()
        endif()
    endforeach()
endif()

# Fallback: scan common conda/pip site-packages locations
if(NOT NVCOMP_ROOT)
    file(GLOB _nvcomp_candidates
        "$ENV{HOME}/miniconda3/lib/python*/site-packages/nvidia/libnvcomp"
        "$ENV{HOME}/anaconda3/lib/python*/site-packages/nvidia/libnvcomp"
        "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/libnvcomp"
    )
    list(LENGTH _nvcomp_candidates _num_candidates)
    if(_num_candidates GREATER 0)
        list(GET _nvcomp_candidates 0 NVCOMP_ROOT)
    endif()
endif()

if(NOT NVCOMP_ROOT)
    message(FATAL_ERROR "nvcomp not found. Set -DNVCOMP_ROOT=<path> or pip install nvidia-libnvcomp-cu12")
endif()

message(STATUS "nvcomp root: ${NVCOMP_ROOT}")

set(NVCOMP_INCLUDE_DIR "${NVCOMP_ROOT}/include")
set(NVCOMP_LIB_DIR "${NVCOMP_ROOT}/lib64")

if(NOT EXISTS "${NVCOMP_INCLUDE_DIR}/nvcomp.h")
    message(FATAL_ERROR "nvcomp headers not found at ${NVCOMP_INCLUDE_DIR}")
endif()

# Create imported target for nvcomp
add_library(nvcomp SHARED IMPORTED)
set_target_properties(nvcomp PROPERTIES
    IMPORTED_LOCATION "${NVCOMP_LIB_DIR}/libnvcomp.so.5"
    INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
)

# ============================================================
# GPUFastQ Library
# ============================================================
add_library(gpufastq_lib
    src/codec_bsc.cpp
    src/codec_gpu.cu
    src/codec_gpu_nvcomp.cu
    src/compression_workflow.cpp
    src/decompression_workflow.cpp
    src/fastq_parser.cpp
    src/gpu_compressor.cu
    src/serializer.cpp
)

set(BSC_ENABLE_CUDA ON CACHE BOOL "Enable libbsc CUDA support in submodule build" FORCE)
set(BSC_ENABLE_OPENMP ON CACHE BOOL "Enable libbsc OpenMP support in submodule build" FORCE)
set(BSC_BUILD_SHARED_LIB OFF CACHE BOOL "Build libbsc as a static library" FORCE)
add_subdirectory(deps/libbsc deps/libbsc-build EXCLUDE_FROM_ALL)

set(PROJECT_RUNTIME_RPATH
    "${NVCOMP_LIB_DIR}"
    "/usr/local/lib"
    "$<TARGET_FILE_DIR:CUDA::cudart>"
)

if(OpenMP_gomp_LIBRARY)
    get_filename_component(OPENMP_RUNTIME_DIR "${OpenMP_gomp_LIBRARY}" DIRECTORY)
    list(APPEND PROJECT_RUNTIME_RPATH "${OPENMP_RUNTIME_DIR}")
endif()

list(REMOVE_DUPLICATES PROJECT_RUNTIME_RPATH)

target_include_directories(gpufastq_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/libbsc
)

target_link_libraries(gpufastq_lib PUBLIC
    nvcomp
    CUDA::cudart
    bsc::libbsc
)

set_target_properties(gpufastq_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
# GPUFastQ Executable
# ============================================================
add_executable(cuvertq src/main.cpp)
target_link_libraries(cuvertq PRIVATE gpufastq_lib)

set_target_properties(cuvertq PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}"
    INSTALL_RPATH "${PROJECT_RUNTIME_RPATH}"
)

# ============================================================
# Ablation Executable
# ============================================================
add_executable(gpufastq_ablation src/ablation.cpp)
target_link_libraries(gpufastq_ablation PRIVATE gpufastq_lib)

set_target_properties(gpufastq_ablation PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}"
    INSTALL_RPATH "${PROJECT_RUNTIME_RPATH}"
)

# ============================================================
# Ablation Zstd Executable
# ============================================================
add_executable(gpufastq_ablation_zstd src/ablation_zstd.cpp)
target_link_libraries(gpufastq_ablation_zstd PRIVATE gpufastq_lib)

set_target_properties(gpufastq_ablation_zstd PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}"
    INSTALL_RPATH "${PROJECT_RUNTIME_RPATH}"
)
