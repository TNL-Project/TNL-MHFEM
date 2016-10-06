#pragma once

#include <TNL/Solvers/SolverMonitor.h>
#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/SharedPointer.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Timer.h>

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
    typedef typename MeshDependentData::RealType RealType;
    typedef typename MeshDependentData::DeviceType DeviceType;
    typedef typename MeshDependentData::IndexType IndexType;

    typedef Mesh MeshType;
    typedef TNL::SharedPointer< MeshType, DeviceType > MeshPointer;
    typedef MeshDependentData MeshDependentDataType;
    typedef TNL::SharedPointer< MeshDependentDataType, DeviceType > MeshDependentDataPointer;
    typedef TNL::Containers::Vector< RealType, DeviceType, IndexType > DofVectorType;
    typedef TNL::SharedPointer< DofVectorType > DofVectorPointer;
    typedef TNL::Functions::MeshFunction< Mesh, Mesh::meshDimensions - 1, RealType, MeshDependentDataType::NumberOfEquations > DofFunction;
    typedef TNL::SharedPointer< DofFunction > DofFunctionPointer;
    typedef TNL::SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
    typedef TNL::SharedPointer< BoundaryConditions > BoundaryConditionsPointer;
    typedef TNL::SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
    typedef Matrix MatrixType;
    typedef TNL::SharedPointer< MatrixType > MatrixPointer;

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

    DofFunctionPointer dofFunctionPointer;

    DifferentialOperatorPointer differentialOperatorPointer;

    BoundaryConditionsPointer boundaryConditionsPointer;

    RightHandSidePointer rightHandSidePointer;

    TNL::Timer timer_R, timer_Q, timer_explicit, timer_nonlinear, timer_upwind;
};

} // namespace mhfem

#include "Solver_impl.h"
