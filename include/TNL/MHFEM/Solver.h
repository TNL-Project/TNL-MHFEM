#pragma once

#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/LinearSolverTypeResolver.h>
#include <TNL/Timer.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Containers/HypreParVector.h>
#include <TNL/Matrices/HypreParCSRMatrix.h>
#include <TNL/Solvers/Linear/Hypre.h>
#ifdef HAVE_GINKGO
    #include <TNL/Containers/GinkgoVector.h>
    #include <TNL/Matrices/GinkgoOperator.h>
    #include <TNL/Solvers/GinkgoConvergenceLoggerMonitor.h>
#endif

#include "LinearSystem.h"
#include "BoundaryConditions.h"

namespace TNL::MHFEM
{

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
class Solver
{
    static_assert( std::is_same< typename Matrix::DeviceType, typename MeshDependentData::DeviceType >::value,
                   "Matrix::DeviceType does not match MeshDependentData::DeviceType" );
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;
    using Index2D = TNL::Containers::StaticArray< 2, IndexType >;
    using Index3D = TNL::Containers::StaticArray< 3, IndexType >;

    using MeshType = typename MeshDependentData::MeshType;
    using HostMeshType = typename HostMesh< MeshType >::type;
    using DistributedMeshType = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshType >;
    using DistributedHostMeshType = TNL::Meshes::DistributedMeshes::DistributedMesh< HostMeshType >;
    using DistributedMeshPointer = std::shared_ptr< DistributedMeshType >;
    using DistributedHostMeshPointer = std::shared_ptr< DistributedHostMeshType >;

    // TODO: avoid as many smart pointers as possible
    using MeshDependentDataType = MeshDependentData;
    using MeshDependentDataPointer = TNL::Pointers::SharedPointer< MeshDependentDataType, DeviceType >;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    using DofViewType = TNL::Containers::VectorView< RealType, DeviceType, IndexType >;
    using LinearSystem = MHFEM::LinearSystem< MeshType, MeshDependentDataType >;
    using BoundaryConditions = MHFEM::BoundaryConditions< MeshDependentDataType, BoundaryModel >;
    using BoundaryConditionsPointer = TNL::Pointers::SharedPointer< BoundaryConditions >;

    using MatrixType = Matrix;
    using DistributedMatrixType = TNL::Matrices::DistributedMatrix< Matrix >;
    using DistributedMatrixPointer = std::shared_ptr< DistributedMatrixType >;

    using FaceSynchronizerType = TNL::Meshes::DistributedMeshes::DistributedMeshSynchronizer< DistributedMeshType, MeshType::getMeshDimension() - 1 >;

    static std::string getPrologHeader();

    // initialization methods
    void setMesh( DistributedHostMeshPointer & meshPointer );

    bool setup( const TNL::Config::ParameterContainer & parameters,
                const std::string & prefix = "" );

    bool setInitialCondition( const TNL::Config::ParameterContainer & parameters );

    void setupLinearSystem();

    // getters for internal components (used e.g. for coupling)
    DistributedHostMeshPointer getHostMesh();

    DistributedMeshPointer getMesh();

    MeshDependentDataPointer& getMeshDependentData();

    BoundaryConditionsPointer& getBoundaryConditions();

    std::shared_ptr<FaceSynchronizerType>& getFaceSynchronizer();

    // index getters
    IndexType getDofsOffset() const;

    IndexType getLocalDofs() const;

    IndexType getDofs() const;

    IndexType getGlobalDofs() const;

    // main solver methods (used from the functions in control.h)
    void makeSnapshot( const RealType time,
                       const IndexType step );

    void preIterate( const RealType time,
                     const RealType tau );

    void assembleLinearSystem( const RealType time,
                               const RealType tau );

    void solveLinearSystem( TNL::Solvers::IterativeSolverMonitor< RealType, IndexType >* solverMonitor = nullptr );

    void saveLinearSystem( const Matrix & matrix,
                           DofViewType dofs,
                           DofViewType rhs ) const;

    void postIterate( const RealType time,
                      const RealType tau );

    void writeEpilog( TNL::Logger & logger ) const;

    static void estimateMemoryDemands( const DistributedHostMeshType & mesh, std::ostream & out = std::cout );

protected:
    // prefix for snapshots
    std::string outputDirectory;
    IndexType facesOffset = 0;
    IndexType globalFaces = 0;
    IndexType globalCells = 0;
    // counts without ghost entities
    IndexType localFaces = 0;
    IndexType localCells = 0;

    // output/profiling variables
    long long int allIterations = 0;
    long long int allTimeSteps = 0;
    TNL::Timer timer_preIterate, timer_assembleLinearSystem, timer_linearPreconditioner, timer_linearSolver, timer_postIterate,
               // preIterate
               timer_b, timer_R, timer_Q, timer_nonlinear, timer_upwind, timer_model_preIterate,
               // postIterate
               timer_explicit, timer_velocities, timer_model_postIterate,
               // MPI synchronization
               timer_mpi_upwind;

#ifdef HAVE_GINKGO
    // enum of implemented Ginkgo preconditioner types
    enum GKO_PRECONDITIONER_TYPE { AMGX, ILU_ISAI, PARILU_ISAI, PARILUT_ISAI };
    GKO_PRECONDITIONER_TYPE gko_preconditioner_type = AMGX;

    // Ginkgo executor
    std::shared_ptr< gko::Executor > gko_exec = nullptr;

    // Ginkgo convergence logger
    std::shared_ptr< TNL::Solvers::GinkgoConvergenceLoggerMonitor< RealType, IndexType > > gko_convergence_logger = nullptr;

    // Ginkgo stopping criteria
    std::shared_ptr< gko::stop::CriterionFactory > gko_stop_iter = nullptr;
    std::shared_ptr< gko::stop::CriterionFactory > gko_stop_tol = nullptr;

    // Ginkgo preconditioner
    std::shared_ptr< gko::LinOp > gko_preconditioner = nullptr;

    long long int gko_updated_iters = 0;
    long long int gko_last_iters = 1;
    long long int gko_setup_counter = 0;
#endif

#ifdef HAVE_HYPRE
    TNL::Timer timer_hypre_conversion, timer_hypre_setup, timer_hypre_solve, timer_hypre_synchronization;

    TNL::Matrices::HypreCSRMatrix csr_diag, csr_offd;
    TNL::Matrices::HypreParCSRMatrix parcsr_A;
    using ColMapOffdDeviceType = TNL::Containers::Array< HYPRE_Int, DeviceType, HYPRE_Int >;
    using ColMapOffdType = TNL::Containers::Array< HYPRE_Int, TNL::Devices::Host, HYPRE_Int >;
    ColMapOffdType col_map_offd;

    // Hypre solver and preconditioner
    // (we need std::unique_ptr to pass the MPI communicator later to the constructor)
    std::unique_ptr< TNL::Solvers::Linear::HypreBiCGSTAB > hypre_solver = nullptr;
    std::unique_ptr< TNL::Solvers::Linear::HypreBoomerAMG > hypre_precond = nullptr;

    long long int hypre_updated_iters = 0;
    long long int hypre_last_iters = 1;
    long long int hypre_setup_counter = 0;
#endif

    DistributedHostMeshPointer distributedHostMeshPointer = nullptr;
    DistributedMeshPointer distributedMeshPointer = nullptr;
    MeshDependentDataPointer mdd;
    BoundaryConditionsPointer boundaryConditionsPointer;

    // linear system solver and preconditioner
    using LinearSolverType = TNL::Solvers::Linear::LinearSolver< DistributedMatrixType >;
    using LinearSolverPointer = std::shared_ptr< LinearSolverType >;
    using PreconditionerType = typename LinearSolverType::PreconditionerType;
    using PreconditionerPointer = std::shared_ptr< PreconditionerType >;
    // uninitialized smart pointers (they are initialized in the setup method)
    LinearSolverPointer linearSystemSolver = nullptr;
    PreconditionerPointer preconditioner = nullptr;
    // (distributedMatrixPointer is initialized in the setupLinearSystem method)
    DistributedMatrixPointer distributedMatrixPointer = nullptr;

    // right hand side vector for the linear system
    DofVectorType rhsVector;

    // device pointers to local stuff for passing to CUDA kernels
    using LocalMeshPointer = TNL::Pointers::DevicePointer< MeshType >;
    using LocalMatrixPointer = TNL::Pointers::DevicePointer< MatrixType >;
    LocalMeshPointer localMeshPointer = nullptr;
    LocalMatrixPointer localMatrixPointer = nullptr;

    std::shared_ptr< FaceSynchronizerType > faceSynchronizer;
};

} // namespace TNL::MHFEM

#include "Solver_impl.h"
