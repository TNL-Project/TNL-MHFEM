set( sources
        tnl_mhfem.cpp
)

# Create the module
pybind11_add_module(tnl_mhfem ${sources})

# Enable link-time optimization
set_target_properties(tnl_mhfem PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# Add TNL and PyTNL
target_link_libraries(tnl_mhfem PUBLIC TNL::TNL_MHFEM PyTNL::PyTNL)

# Set installation path
install(TARGETS tnl_mhfem DESTINATION ${PyTNL_PYTHON_SITE_PACKAGES_DIR})
