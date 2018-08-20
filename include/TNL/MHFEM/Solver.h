#pragma once

#include <TNL/Solvers/SolverMonitor.h>
#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/SharedPointer.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/NoTimeDiscretisation.h>
#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Timer.h>

#include "DifferentialOperator.h"
#include "RightHandSide.h"
#include "HybridizationExplicitFunction.h"
#include "Upwind.h"
#include "../lib_general/MeshOrdering.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
class Solver :
    public TNL::Problems::PDEProblem< Mesh,
                                      typename MeshDependentData::RealType,
                                      typename MeshDependentData::DeviceType,
                                      typename MeshDependentData::IndexType >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;

    using MeshType = Mesh;
    using MeshPointer = TNL::SharedPointer< MeshType, DeviceType >;
    using MeshDependentDataType = MeshDependentData;
    using MeshDependentDataPointer = TNL::SharedPointer< MeshDependentDataType, DeviceType >;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    using DofVectorPointer = TNL::SharedPointer< DofVectorType >;
    using DofFunction = TNL::Functions::MeshFunction< Mesh, Mesh::getMeshDimension() - 1, RealType, MeshDependentDataType::NumberOfEquations >;
    using DofFunctionPointer = TNL::SharedPointer< DofFunction >;
    using DifferentialOperator = mhfem::DifferentialOperator< MeshType, MeshDependentDataType >;
    using DifferentialOperatorPointer = TNL::SharedPointer< DifferentialOperator >;
    using BoundaryConditionsPointer = TNL::SharedPointer< BoundaryConditions >;
    using RightHandSide = mhfem::RightHandSide< MeshType, MeshDependentDataType >;
    using RightHandSidePointer = TNL::SharedPointer< RightHandSide, DeviceType >;
    using MatrixType = Matrix;
    using MatrixPointer = TNL::SharedPointer< MatrixType >;

    static TNL::String getPrologHeader();

    static void writeProlog( TNL::Logger & logger,
                             const TNL::Config::ParameterContainer & parameters );

    TNL::Solvers::SolverMonitor* getSolverMonitor();

    void setMesh( MeshPointer & meshPointer );

    bool setup( const TNL::Config::ParameterContainer & parameters,
                const TNL::String & prefix );

    bool setInitialCondition( const TNL::Config::ParameterContainer & parameters,
                              DofVectorPointer & dofsPointer );

    bool setupLinearSystem( MatrixPointer & matrixPointer );

    bool makeSnapshot( const RealType & time,
                       const IndexType & step,
                       DofVectorPointer & dofsPointer );

    IndexType getDofs() const;

    void bindDofs( DofVectorPointer & dofs );

    bool preIterate( const RealType & time,
                     const RealType & tau,
                     DofVectorPointer & dofsPointer );

    void assemblyLinearSystem( const RealType & time,
                               const RealType & tau,
                               DofVectorPointer & dofsPointer,
                               MatrixPointer & matrixPointer,
                               DofVectorPointer & rightHandSidePointer );

    void saveFailedLinearSystem( const Matrix & matrix,
                                 const DofVectorType & dofs,
                                 const DofVectorType & rhs ) const;

    bool postIterate( const RealType & time,
                      const RealType & tau,
                      DofVectorPointer & dofsPointer );

    bool writeEpilog( TNL::Logger & logger );

protected:
    // prefix for snapshots
    TNL::String outputPrefix;
    bool doMeshOrdering;

    // timers for profiling
    TNL::Timer timer_b, timer_R, timer_Q, timer_explicit, timer_nonlinear, timer_velocities, timer_upwind;

    MeshPointer meshPointer;
    // holder for mesh ordering permutations
    MeshOrdering< Mesh > meshOrdering;
    MeshDependentDataPointer mdd;

    DofFunctionPointer dofFunctionPointer;
    DifferentialOperatorPointer differentialOperatorPointer;
    BoundaryConditionsPointer boundaryConditionsPointer;
    RightHandSidePointer rightHandSidePointer;

    // cached instance for assemblyLinearSystem
    using LinearSystemAssembler = TNL::Solvers::PDE::LinearSystemAssembler
                                  < MeshType,
                                    DofFunction,
                                    DifferentialOperator,
                                    BoundaryConditions,
                                    RightHandSide,
                                    TNL::Solvers::PDE::NoTimeDiscretisation,
                                    DofVectorType >;
    LinearSystemAssembler systemAssembler;


    // cached instances for postIterate method

    // output
    using ZkMeshFunction = TNL::Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), RealType, MeshDependentDataType::NumberOfEquations >;
    TNL::SharedPointer< ZkMeshFunction, DeviceType > meshFunctionZK;
    // input
    using HybridizationFunction = HybridizationExplicitFunction< MeshType, MeshDependentDataType >;
    TNL::SharedPointer< HybridizationFunction, DeviceType > functionZK;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< ZkMeshFunction, HybridizationFunction > evaluatorZK;

    // output
    TNL::SharedPointer< DofFunction, DeviceType > upwindMeshFunction;
    // input
    using UpwindFunction = Upwind< MeshType, MeshDependentDataType, BoundaryConditions >;
    TNL::SharedPointer< UpwindFunction, DeviceType > upwindFunction;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< DofFunction, UpwindFunction > upwindEvaluator;

    // output
    using NDofFunction = TNL::Functions::MeshFunction< Mesh, Mesh::getMeshDimension() - 1, RealType, MeshDependentDataType::NumberOfEquations * MeshDependentDataType::NumberOfEquations >;
    TNL::SharedPointer< NDofFunction, DeviceType > upwindZMeshFunction;
    // input
    using UpwindZFunction = UpwindZ< MeshType, MeshDependentDataType, BoundaryConditions >;
    TNL::SharedPointer< UpwindZFunction, DeviceType > upwindZFunction;
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< NDofFunction, UpwindZFunction > upwindZEvaluator;
};

} // namespace mhfem

#include "Solver_impl.h"
