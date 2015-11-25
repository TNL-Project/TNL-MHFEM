#pragma once

#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <problems/tnlPDEProblem.h>
#include <core/tnlTimerRT.h>

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
class Solver :
    public tnlPDEProblem< Mesh,
                          typename MeshDependentData::RealType,
                          typename MeshDependentData::DeviceType,
                          typename MeshDependentData::IndexType >
{
public:
    typedef Mesh MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType > DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef Matrix MatrixType;
    typedef typename MatrixType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;

    static tnlString getTypeStatic();

    tnlString getPrologHeader() const;

    void writeProlog( tnlLogger & logger,
                      const tnlParameterContainer & parameters ) const;

    tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();

    bool setup( const tnlParameterContainer & parameters );

    bool setInitialCondition( const tnlParameterContainer & parameters,
                              const MeshType & mesh,
                              DofVectorType & dofs,
                              MeshDependentDataType & mdd );

    bool setupLinearSystem( const MeshType & mesh,
                            MatrixType & matrix );

    bool makeSnapshot( const RealType & time,
                       const IndexType & step,
                       const MeshType & mesh,
                       DofVectorType & dofs,
                       MeshDependentDataType & mdd );

    IndexType getDofs( const MeshType & mesh ) const;

    void bindDofs( const MeshType & mesh,
                   DofVectorType & dofs );

    bool setMeshDependentData( const MeshType& mesh,
                               MeshDependentDataType & mdd );

    void bindMeshDependentData( const MeshType& mesh,
                                MeshDependentDataType & mdd );

    bool preIterate( const RealType & time,
                     const RealType & tau,
                     const MeshType & mesh,
                     DofVectorType & dofs,
                     MeshDependentDataType & mdd );

    void assemblyLinearSystem( const RealType & time,
                               const RealType & tau,
                               const MeshType & mesh,
                               DofVectorType & dofs,
                               MatrixType & matrix,
                               DofVectorType & rightHandSide,
                               MeshDependentDataType & mdd );

    bool postIterate( const RealType & time,
                      const RealType & tau,
                      const MeshType & mesh,
                      DofVectorType & dofs,
                      MeshDependentDataType & mdd );

    bool writeEpilog( tnlLogger & logger );

protected:
    // prefix for snapshots
    ::tnlString outputPrefix;

    DifferentialOperator differentialOperator;

    BoundaryConditions boundaryConditions;

    RightHandSide rightHandSide;

    tnlTimerRT timer_R, timer_Q, timer_explicit, timer_nonlinear, timer_upwind;
};

} // namespace mhfem

#include "Solver_impl.h"
