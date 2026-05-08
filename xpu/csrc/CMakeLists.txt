# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)

option(DEEPEP_USE_XPU "Build XPU migration target without CUDA kernels" ON)
option(DEEPEP_XPU_NATIVE_STAGED "Build staged native XPU module from mirrored sources" OFF)

if (DEEPEP_USE_XPU)
	project(deep_ep LANGUAGES CXX)
	set(CMAKE_VERBOSE_MAKEFILE ON)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
	set(CMAKE_CXX_STANDARD 17)

	if (NOT Python_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python")
		set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "Python executable for XPU staged build")
	endif()
	if (NOT Python_ROOT_DIR AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
		set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Python root for XPU staged build")
	endif()

	find_package(Python COMPONENTS Interpreter Development REQUIRED)
	if (NOT pybind11_DIR)
		execute_process(
			COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
			OUTPUT_VARIABLE pybind11_DIR
			RESULT_VARIABLE pybind11_DIR_RESULT
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET)
		if (pybind11_DIR_RESULT EQUAL 0 AND EXISTS "${pybind11_DIR}/pybind11Config.cmake")
			list(PREPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
		endif()
	endif()
	if (NOT Torch_DIR)
		execute_process(
			COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
			OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
			RESULT_VARIABLE TORCH_CMAKE_PREFIX_PATH_RESULT
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET)
		if (TORCH_CMAKE_PREFIX_PATH_RESULT EQUAL 0 AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}/Torch/TorchConfig.cmake")
			set(Torch_DIR "${TORCH_CMAKE_PREFIX_PATH}/Torch" CACHE PATH "Torch CMake package directory")
			list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
		endif()
	endif()

	find_package(pybind11 REQUIRED)

	if (DEEPEP_XPU_NATIVE_STAGED)
		# Staged native path: compile mirrored sources as C++ and keep kernel entrypoints as XPU stubs.
		find_package(Torch REQUIRED)
		find_library(TORCH_PYTHON_LIBRARY torch_python
			HINTS ${TORCH_INSTALL_PREFIX}/lib ${TORCH_LIBRARY_DIRS})
		if (NOT TORCH_PYTHON_LIBRARY)
			message(FATAL_ERROR "torch_python library was not found for the staged XPU build")
		endif()
		set(EP_XPU_STAGED_SOURCES
			deep_ep.cpp
			kernels/runtime.cu
			kernels/layout.cu
			kernels/intranode.cu
			kernels/internode.cu
			kernels/internode_ll.cu)
		set_source_files_properties(${EP_XPU_STAGED_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
		pybind11_add_module(deep_ep_cpp_xpu ${EP_XPU_STAGED_SOURCES})
		target_compile_definitions(deep_ep_cpp_xpu PRIVATE DEEPEP_USE_XPU)
		target_include_directories(deep_ep_cpp_xpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
		target_link_libraries(deep_ep_cpp_xpu PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
	else()
		# Build the migration stub so the xpu tree can configure/build without CUDA toolchains.
		pybind11_add_module(deep_ep_cpp_xpu deep_ep_xpu_stub.cpp)
		target_compile_definitions(deep_ep_cpp_xpu PRIVATE DEEPEP_USE_XPU)
		target_include_directories(deep_ep_cpp_xpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	endif()
	return()
endif()

project(deep_ep LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR})

add_subdirectory(kernels)

# Link CPP and CUDA together
pybind11_add_module(deep_ep_cpp_xpu deep_ep.cpp)
target_link_libraries(deep_ep_cpp_xpu PRIVATE ${EP_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
