cmake_minimum_required(VERSION 3.28)

# Set default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(TNL-MHFEM
    LANGUAGES CXX CUDA
)

# Require C++17, and disable compiler-specific extensions (if possible).
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=vla")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Set build flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")

# Enforce (more or less) warning-free builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized")

if( CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" )
   set( CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info" )
   if( NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" )
      # enforce (more or less) warning-free builds for host code
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror -Xcompiler -Wno-error=deprecated -Xcompiler -Wno-error=deprecated-declarations" )
   endif()
endif()

if( CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" )
    # enforce (more or less) warning-free builds
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version" )
    # workaround for Clang 15
    # https://github.com/llvm/llvm-project/issues/58491
    set( CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -g0" )
endif()

# warn about redundant semicolons
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi" )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt" )
   endif()
endif()

# Set the project's include path
include_directories(include)

# Add dependencies
include(FetchContent)
FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    GIT_TAG main
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(TNL)

FetchContent_Declare(PyTNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/pytnl.git
    GIT_TAG main
)
FetchContent_MakeAvailable(PyTNL)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(pybind11)

# Set the install path for Python modules
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
set(PyTNL_PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)

# Add project subdirectories
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(examples)
