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

#include "HybridizationExplicitFunction.h"
#include "Upwind.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
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
    using DifferentialOperatorPointer = TNL::SharedPointer< DifferentialOperator >;
    using BoundaryConditionsPointer = TNL::SharedPointer< BoundaryConditions >;
    using RightHandSidePointer = TNL::SharedPointer< RightHandSide, DeviceType >;
    using MatrixType = Matrix;
    using MatrixPointer = TNL::SharedPointer< MatrixType >;

    static TNL::String getTypeStatic();

    TNL::String getPrologHeader() const;

    void writeProlog( TNL::Logger & logger,
                      const TNL::Config::ParameterContainer & parameters ) const;

    TNL::Solvers::SolverMonitor* getSolverMonitor();

    bool setup( const MeshPointer & meshPointer,
                const TNL::Config::ParameterContainer & parameters,
                const TNL::String & prefix );

    bool setInitialCondition( const TNL::Config::ParameterContainer & parameters,
                              const MeshPointer & meshPointer,
                              DofVectorPointer & dofsPointer,
                              MeshDependentDataPointer & mdd );

    bool setupLinearSystem( const MeshPointer & meshPointer,
                            MatrixPointer & matrixPointer );

    bool makeSnapshot( const RealType & time,
                       const IndexType & step,
                       const MeshPointer & meshPointer,
                       DofVectorPointer & dofsPointer,
                       MeshDependentDataPointer & mdd );

    IndexType getDofs( const MeshPointer & mesh ) const;

    void bindDofs( const MeshPointer & meshPointer,
                   DofVectorPointer & dofs );

    bool setMeshDependentData( const MeshPointer & meshPointer,
                               MeshDependentDataPointer & mdd );

    void bindMeshDependentData( const MeshPointer & meshPointer,
                                MeshDependentDataPointer & mdd );

    bool preIterate( const RealType & time,
                     const RealType & tau,
                     const MeshPointer & meshPointer,
                     DofVectorPointer & dofsPointer,
                     MeshDependentDataPointer & mdd );

    void assemblyLinearSystem( const RealType & time,
                               const RealType & tau,
                               const MeshPointer & meshPointer,
                               DofVectorPointer & dofsPointer,
                               MatrixPointer & matrixPointer,
                               DofVectorPointer & rightHandSidePointer,
                               MeshDependentDataPointer & mdd );

    bool postIterate( const RealType & time,
                      const RealType & tau,
                      const MeshPointer & meshPointer,
                      DofVectorPointer & dofsPointer,
                      MeshDependentDataPointer & mdd );

    bool writeEpilog( TNL::Logger & logger );

protected:
    // prefix for snapshots
    TNL::String outputPrefix;

    // for condition in preIterate
    RealType initialTime;

    // timers for profiling
    TNL::Timer timer_b, timer_R, timer_Q, timer_explicit, timer_nonlinear, timer_upwind;

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
                                    MatrixType,
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
};

} // namespace mhfem

#include "Solver_impl.h"
