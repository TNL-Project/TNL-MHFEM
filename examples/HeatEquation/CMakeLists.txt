# add all targets
add_executable(HeatEquation_host HeatEquation_host.cpp)
add_executable(HeatEquation_cuda HeatEquation_cuda.cu)

# add TNL to all targets
target_link_libraries(HeatEquation_host PUBLIC TNL::TNL_MHFEM)
target_link_libraries(HeatEquation_cuda PUBLIC TNL::TNL_MHFEM)

set(TARGETS
        HeatEquation_host
        HeatEquation_cuda
)
