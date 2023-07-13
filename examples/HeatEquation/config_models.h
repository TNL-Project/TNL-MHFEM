// heat equation model
#include "HeatEquationModel.h"
using Model = HeatEquationModel< Mesh, Real, MASS_LUMPING >;

// boundary conditions model
#include <TNL/MHFEM/BoundaryModels/Stationary.h>
using BoundaryModel = TNL::MHFEM::BoundaryModels::Stationary< Mesh >;
