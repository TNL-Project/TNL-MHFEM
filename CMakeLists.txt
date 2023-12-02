cmake_minimum_required(VERSION 3.24)

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

# Set the install path for Python modules
# FIXME: for some reason relying on find_package from libs/pytnl fails (dependent target 'Python3::Module' is not defined)
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
set(PyTNL_PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)

# Add dependencies
if(SYSTEM_TNL)
    find_package(TNL REQUIRED)
else()
    add_subdirectory(libs/tnl EXCLUDE_FROM_ALL)
    set(SYSTEM_TNL TRUE)  # avoid further inclusion of libs/tnl from pytnl
endif()
if(SYSTEM_PYTNL)
    find_package(PyTNL REQUIRED)
else()
    add_subdirectory(libs/pytnl)
endif()

# Add project subdirectories
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(examples)
