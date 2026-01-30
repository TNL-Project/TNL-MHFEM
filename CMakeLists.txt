cmake_minimum_required(VERSION 3.28)

# Set default CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(TNL-MHFEM
    LANGUAGES CXX CUDA
)

# Declare all CMake options for the project
option(TNL_MHFEM_EXPORT_INTERFACE_TARGETS "Instruct CMake to generate rules to export the interface target" ${PROJECT_IS_TOP_LEVEL})

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

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        # Enforce (more or less) warning-free builds for host code
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror -Xcompiler -Wno-error=deprecated -Xcompiler -Wno-error=deprecated-declarations")
    endif()
    # Force colorized output for nvcc's host compiler (assuming it is GCC...)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fdiagnostics-color")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # Enforce (more or less) warning-free builds
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=unknown-cuda-version")
    # workaround for Clang 15
    # https://github.com/llvm/llvm-project/issues/58491
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -Xarch_device -g0")
endif()

# Disable GCC's infamous warnings (they produce many false positives)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-stringop-overflow")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-maybe-uninitialized -Wno-stringop-overflow")
    endif()
endif()

# Warn about redundant semicolons
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt")
    endif()
endif()

# Make cache variable so it can be used in downstream projects
set(TNL_MHFEM_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "Directories where TNL-MHFEM headers are located")

# Create the exported targets
add_library(TNL_MHFEM INTERFACE)
# Create aliases to match exported targets
add_library(TNL::TNL_MHFEM ALIAS TNL_MHFEM)

if(TNL_MHFEM_EXPORT_INTERFACE_TARGETS)
    # Export the interface targets
    install(TARGETS TNL_MHFEM EXPORT "${PROJECT_NAME}Targets")
    # Install a CMake file for the interface target
    install(
        EXPORT "${PROJECT_NAME}Targets"
        NAMESPACE TNL::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
    # Install the Config.cmake file
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
endif()

# Add the include directory to the interface
if(PROJECT_IS_TOP_LEVEL)
    set(TNL_MHFEM_SYSTEM "")
else()
    set(TNL_MHFEM_SYSTEM SYSTEM)
endif()
target_include_directories(
    TNL_MHFEM ${TNL_MHFEM_SYSTEM} INTERFACE
    $<BUILD_INTERFACE:${TNL_MHFEM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Set position independent code for all targets, including dependencies
# downloaded and built by FetchContent. Otherwise, linking to static libraries
# without PIC fails.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find or fetch dependencies
include(FetchContent)
FetchContent_Declare(TNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/tnl.git
    # TNL does not have any releases
    GIT_TAG 7f10aa5be1de23dfe60383677730045842574485
    EXCLUDE_FROM_ALL
)
set(TNL_EXPORT_INTERFACE_TARGETS ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(TNL)

FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.11.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(PyTNL
    GIT_REPOSITORY https://gitlab.com/tnl-project/pytnl.git
    GIT_TAG 0.0.9
    SYSTEM
)
set(PyTNL_EXPORT_INTERFACE_TARGETS ON)  # needed for adding to the interface target
FetchContent_MakeAvailable(PyTNL)

FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 11.0.0
    SYSTEM
    FIND_PACKAGE_ARGS
)
set(tinyxml2_BUILD_TESTING OFF)
FetchContent_MakeAvailable(tinyxml2)

find_package(ZLIB REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)

# Add dependencies to the interface target
target_link_libraries(TNL_MHFEM INTERFACE TNL::TNL)
target_compile_definitions(TNL_MHFEM INTERFACE "-DHAVE_ZLIB")
target_link_libraries(TNL_MHFEM INTERFACE ZLIB::ZLIB)
target_compile_definitions(TNL_MHFEM INTERFACE "-DHAVE_TINYXML2")
target_link_libraries(TNL_MHFEM INTERFACE tinyxml2::tinyxml2)
if(OPENMP_FOUND)
    target_compile_definitions(TNL_MHFEM INTERFACE "-DHAVE_OPENMP")
    target_link_libraries(TNL_MHFEM INTERFACE OpenMP::OpenMP_CXX)
    target_compile_options(TNL_MHFEM INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: ${OpenMP_CXX_FLAGS} >)
    # nvcc needs -Xcompiler
    target_compile_options(TNL_MHFEM INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Xcompiler=-fopenmp >)
endif()
if(MPI_FOUND)
    target_compile_definitions(TNL_MHFEM INTERFACE "-DHAVE_MPI")
    target_link_libraries(TNL_MHFEM INTERFACE MPI::MPI_CXX)
endif()

# Add project subdirectories
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(examples)
