#pragma once

#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/LinearSolverTypeResolver.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Timer.h>
#include <TNL/Communicators/NoDistrCommunicator.h>

#include "DifferentialOperator.h"
#include "RightHandSide.h"
#include "HybridizationExplicitFunction.h"
#include "Upwind.h"
#include "../lib_general/MeshOrdering.h"
#include "../lib_general/LinearSystemAssembler.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
class Solver
{
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;

    using MeshType = Mesh;
    using MeshPointer = TNL::Pointers::SharedPointer< MeshType, DeviceType >;
    using MeshDependentDataType = MeshDependentData;
    using MeshDependentDataPointer = TNL::Pointers::SharedPointer< MeshDependentDataType, DeviceType >;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    using DofViewType = TNL::Containers::VectorView< RealType, DeviceType, IndexType >;
    using DifferentialOperator = mhfem::DifferentialOperator< MeshType, MeshDependentDataType >;
    using DifferentialOperatorPointer = TNL::Pointers::SharedPointer< DifferentialOperator >;
    using BoundaryConditionsPointer = TNL::Pointers::SharedPointer< BoundaryConditions >;
    using RightHandSide = mhfem::RightHandSide< MeshType, MeshDependentDataType >;
    using RightHandSidePointer = TNL::Pointers::SharedPointer< RightHandSide, DeviceType >;
    using MatrixType = Matrix;
    using MatrixPointer = TNL::Pointers::SharedPointer< MatrixType >;

    static TNL::String getPrologHeader();

    void setMesh( MeshPointer & meshPointer );

    bool setup( const TNL::Config::ParameterContainer & parameters,
                const TNL::String & prefix = "" );

    bool setInitialCondition( const TNL::Config::ParameterContainer & parameters );

    void setupLinearSystem();

    bool makeSnapshot( const RealType & time,
                       const IndexType & step );

    IndexType getDofs() const;

    MeshDependentDataPointer& getMeshDependentData();

    void preIterate( const RealType & time,
                     const RealType & tau );

    void assembleLinearSystem( const RealType & time,
                               const RealType & tau );

    void solveLinearSystem( TNL::Solvers::IterativeSolverMonitor< RealType, IndexType >* solverMonitor = nullptr );

    void saveLinearSystem( const Matrix & matrix,
                           DofViewType dofs,
                           DofViewType rhs ) const;

    void postIterate( const RealType & time,
                      const RealType & tau );

    void writeEpilog( TNL::Logger & logger ) const;

protected:
    // prefix for snapshots
    TNL::String outputDirectory;
    bool doMeshOrdering;

    // output/profiling variables
    long long int allIterations = 0;
    TNL::Timer timer_preIterate, timer_assembleLinearSystem, timer_linearPreconditioner, timer_linearSolver, timer_postIterate,
               // preIterate
               timer_b, timer_R, timer_Q, timer_nonlinear, timer_upwind,
               // postIterate
               timer_explicit, timer_velocities;

    MeshPointer meshPointer = nullptr;
    // holder for mesh ordering permutations
    MeshOrdering< Mesh > meshOrdering;
    MeshDependentDataPointer mdd;

    DifferentialOperatorPointer differentialOperatorPointer;
    BoundaryConditionsPointer boundaryConditionsPointer;
    RightHandSidePointer rightHandSidePointer;

    // cached instance for assembleLinearSystem
    using LinearSystemAssembler = mhfem::LinearSystemAssembler
                                  < MeshType,
                                    DifferentialOperator,
                                    BoundaryConditions,
                                    RightHandSide,
                                    DofVectorType >;
    LinearSystemAssembler systemAssembler;

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


    // cached instances for postIterate method

    // output
    using ZkMeshFunction = TNL::Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), RealType, MeshDependentDataType::NumberOfEquations >;
    TNL::Pointers::SharedPointer< ZkMeshFunction, DeviceType > meshFunctionZK;
    // input
    using HybridizationFunction = HybridizationExplicitFunction< MeshType, MeshDependentDataType >;
    TNL::Pointers::SharedPointer< HybridizationFunction, DeviceType > functionZK;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< ZkMeshFunction, HybridizationFunction > evaluatorZK;

    // output
    using DofFunction = TNL::Functions::MeshFunction< Mesh, Mesh::getMeshDimension() - 1, RealType, MeshDependentDataType::NumberOfEquations >;
    TNL::Pointers::SharedPointer< DofFunction, DeviceType > upwindMeshFunction;
    // input
    using UpwindFunction = Upwind< MeshType, MeshDependentDataType, BoundaryConditions >;
    TNL::Pointers::SharedPointer< UpwindFunction, DeviceType > upwindFunction;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< DofFunction, UpwindFunction > upwindEvaluator;

    // output
    using NDofFunction = TNL::Functions::MeshFunction< Mesh, Mesh::getMeshDimension() - 1, RealType, MeshDependentDataType::NumberOfEquations * MeshDependentDataType::NumberOfEquations >;
    TNL::Pointers::SharedPointer< NDofFunction, DeviceType > upwindZMeshFunction;
    // input
    using UpwindZFunction = UpwindZ< MeshType, MeshDependentDataType, BoundaryConditions >;
    TNL::Pointers::SharedPointer< UpwindZFunction, DeviceType > upwindZFunction;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< NDofFunction, UpwindZFunction > upwindZEvaluator;
};

} // namespace mhfem

#include "Solver_impl.h"
