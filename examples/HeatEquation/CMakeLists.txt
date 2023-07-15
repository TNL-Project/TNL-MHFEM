# add all targets
add_executable(HeatEquation_host HeatEquation_host.cpp)
add_executable(HeatEquation_cuda HeatEquation_cuda.cu)

# add TNL to all targets
target_link_libraries(HeatEquation_host PUBLIC TNL::TNL_CXX)
target_link_libraries(HeatEquation_cuda PUBLIC TNL::TNL_CUDA)

set(TARGETS
        HeatEquation_host
        HeatEquation_cuda
)

foreach(target IN ITEMS ${TARGETS})
    # enable zlib and tinyxml2 (used by mesh readers and writers)
    find_package(ZLIB)
    find_package(tinyxml2)
    if(ZLIB_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
        target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
        target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})
    endif()
    if(tinyxml2_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
        target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
    endif()

    # enable MPI
    find_package(MPI COMPONENTS CXX)
    if(MPI_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endif()
endforeach()
