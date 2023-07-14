set( sources
        tnl_mhfem.cpp
)

# Create the module
pybind11_add_module(tnl_mhfem ${sources})

# Add TNL and PyTNL
target_link_libraries(tnl_mhfem PUBLIC TNL::TNL_CXX PyTNL::PyTNL)

# Set installation path
install(TARGETS tnl_mhfem DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
