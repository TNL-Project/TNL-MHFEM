#pragma once

#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/LinearSolverTypeResolver.h>
#include <TNL/Timer.h>
#include <TNL/Communicators/NoDistrCommunicator.h>

#include "DifferentialOperator.h"
#include "BoundaryConditions.h"
#include "RightHandSide.h"
#include "../lib_general/MeshOrdering.h"

namespace mhfem
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

    using MeshType = typename MeshDependentData::MeshType;
    using MeshPointer = TNL::Pointers::SharedPointer< MeshType, DeviceType >;
    using MeshDependentDataType = MeshDependentData;
    using MeshDependentDataPointer = TNL::Pointers::SharedPointer< MeshDependentDataType, DeviceType >;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    using DofViewType = TNL::Containers::VectorView< RealType, DeviceType, IndexType >;
    using DifferentialOperator = mhfem::DifferentialOperator< MeshType, MeshDependentDataType >;
    using DifferentialOperatorPointer = TNL::Pointers::SharedPointer< DifferentialOperator >;
    using BoundaryConditions = mhfem::BoundaryConditions< MeshDependentDataType, BoundaryModel >;
    using BoundaryConditionsPointer = TNL::Pointers::SharedPointer< BoundaryConditions >;
    using RightHandSide = mhfem::RightHandSide< MeshDependentDataType >;
    using RightHandSidePointer = TNL::Pointers::SharedPointer< RightHandSide, DeviceType >;
    using MatrixType = Matrix;
    using MatrixPointer = TNL::Pointers::SharedPointer< MatrixType >;

    static TNL::String getPrologHeader();

    void setMesh( MeshPointer & meshPointer );

    bool setup( const TNL::Config::ParameterContainer & parameters,
                const TNL::String & prefix = "" );

    bool setInitialCondition( const TNL::Config::ParameterContainer & parameters );

    void setupLinearSystem();

    void makeSnapshot( const RealType time,
                       const IndexType step );

    IndexType getDofs() const;

    MeshDependentDataPointer& getMeshDependentData();

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

protected:
    // prefix for snapshots
    TNL::String outputDirectory;
    bool doMeshOrdering;

    // output/profiling variables
    long long int allIterations = 0;
    TNL::Timer timer_preIterate, timer_assembleLinearSystem, timer_linearPreconditioner, timer_linearSolver, timer_postIterate,
               // preIterate
               timer_b, timer_R, timer_Q, timer_nonlinear, timer_upwind, timer_model_preIterate,
               // postIterate
               timer_explicit, timer_velocities, timer_model_postIterate;

    MeshPointer meshPointer = nullptr;
    // holder for mesh ordering permutations
    MeshOrdering< MeshType > meshOrdering;
    MeshDependentDataPointer mdd;

    DifferentialOperatorPointer differentialOperatorPointer;
    BoundaryConditionsPointer boundaryConditionsPointer;
    RightHandSidePointer rightHandSidePointer;

    // linear system preconditioner and solver
    using LinearSolverType = TNL::Solvers::Linear::LinearSolver< MatrixType >;
    using LinearSolverPointer = std::shared_ptr< LinearSolverType >;
    using PreconditionerType = typename LinearSolverType::PreconditionerType;
    using PreconditionerPointer = std::shared_ptr< PreconditionerType >;
    // uninitialized smart pointers (they are initialized in the setup method)
    LinearSolverPointer linearSystemSolver = nullptr;
    PreconditionerPointer preconditioner = nullptr;

    // matrix and right hand side vector for the linear system
    MatrixPointer matrixPointer;
    DofVectorType rhsVector;
};

} // namespace mhfem

#include "Solver_impl.h"
