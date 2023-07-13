using Real = double;
#ifdef HAVE_HYPRE
#include <TNL/Hypre.h>
using Index = HYPRE_Int;
#else
using Index = int;
#endif

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
using LocalIndexType = short int;
//using CellTopology = TNL::Meshes::Topologies::Edge;
using CellTopology = TNL::Meshes::Topologies::Triangle;
//using CellTopology = TNL::Meshes::Topologies::Quadrangle;
//using CellTopology = TNL::Meshes::Topologies::Tetrahedron;
//using CellTopology = TNL::Meshes::Topologies::Hexahedron;
using Mesh = TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig<
                            CellTopology,
                            CellTopology::dimension,
                            Real,
                            Index,
                            LocalIndexType >,
                        Device >;

#include <TNL/MHFEM/MassMatrix.h>
//static constexpr TNL::MHFEM::MassLumping MASS_LUMPING = TNL::MHFEM::MassLumping::enabled;
static constexpr TNL::MHFEM::MassLumping MASS_LUMPING = TNL::MHFEM::MassLumping::disabled;

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
template< typename Device, typename Index, typename IndexAlocator >
using EllpackSegments = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAlocator >;
template< typename Real, typename Device, typename Index >
using Ellpack = TNL::Matrices::SparseMatrix< Real,
                                             Device,
                                             Index,
                                             TNL::Matrices::GeneralMatrix,
                                             EllpackSegments
                                           >;

#include <TNL/Algorithms/Segments/CSR.h>
template< typename Device, typename Index, typename IndexAlocator >
using CSRSegments = TNL::Algorithms::Segments::CSRLight< Device, Index, IndexAlocator >;
template< typename Real, typename Device, typename Index >
using CSR = TNL::Matrices::SparseMatrix< Real,
                                             Device,
                                             Index,
                                             TNL::Matrices::GeneralMatrix,
                                             CSRSegments
                                           >;

#if defined( HAVE_HYPRE ) || defined( HAVE_GINKGO )
using Matrix = CSR< Real, Device, Index >;
#else
using Matrix = Ellpack< Real, Device, Index >;
#endif
