#pragma once

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <functors/tnlFunctionEnumerator.h>
#include <solvers/pde/tnlNoTimeDiscretisation.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/GenericEnumerator.h"
#include "../lib_general/FaceAverageFunction.h"
#include "../lib_general/device_ptr.h"

#include "Solver.h"
#include "QRupdater.h"
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
tnlString
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getTypeStatic()
{
    return tnlString( "Solver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
tnlString
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getPrologHeader() const
{
    // TODO
//    return tnlString( "Single-phase flow" );
    return tnlString();
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
writeProlog( tnlLogger & logger, const tnlParameterContainer & parameters ) const
{
    logger.writeParameter< tnlString >( "Output prefix:", parameters.getParameter< tnlString >( "output-prefix" ) );
    // TODO: let models write their parameters
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
tnlSolverMonitor< typename Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::RealType,
                  typename Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::IndexType >*
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getSolverMonitor()
{
    return 0;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
setup( const tnlParameterContainer & parameters )
{
    // prefix for snapshots
    outputPrefix = parameters.getParameter< tnlString >( "output-prefix" ) + tnlString("-");

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
typename Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::IndexType
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getDofs( const MeshType & mesh ) const
{
    return MeshDependentDataType::NumberOfEquations * FacesCounter< MeshType >::getNumberOfFaces( mesh );
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
bindDofs( const MeshType & mesh,
          DofVectorType & dofVector )
{ }

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
setMeshDependentData( const MeshType & mesh,
                      MeshDependentDataType & mdd )
{
    return mdd.allocate( mesh );
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
bindMeshDependentData( const MeshType & mesh,
                       MeshDependentDataType & mdd )
{ }


template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
setInitialCondition( const tnlParameterContainer & parameters,
                     const MeshType & mesh,
                     DofVectorType & dofVector,
                     MeshDependentDataType & mdd )
{
    if( ! boundaryConditions.init( parameters, mesh ) )
        return false;

    if( ! mdd.init( parameters, mesh, boundaryConditions ) )
        return false;

    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );
    device_ptr< BoundaryConditions, DeviceType > bcDevicePtr( boundaryConditions );

    // initialize dofVector as an average of mdd.Z on neighbouring cells
    FaceAverageFunctionWithBoundary< MeshType, MeshDependentDataType, BoundaryConditions > faceAverageFunction;
    faceAverageFunction.bind( mddDevicePtr.get(), bcDevicePtr.get(), mdd.Z );
    tnlFunctionEnumerator< MeshType, FaceAverageFunctionWithBoundary< MeshType, MeshDependentDataType, BoundaryConditions >, DofVectorType > faceAverageEnumerator;
    faceAverageEnumerator.template enumerate< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            mesh,
            faceAverageFunction,
            dofVector );

    timer_R.reset();
    timer_Q.reset();
    timer_explicit.reset();
    timer_nonlinear.reset();

    timer_R.stop();
    timer_Q.stop();
    timer_explicit.stop();
    timer_nonlinear.stop();

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
setupLinearSystem( const MeshType & mesh,
                   MatrixType & matrix )
{
    const IndexType dofs = this->getDofs( mesh );
    CompressedRowsLengthsVectorType rowsLengths;
    if( ! rowsLengths.setSize( dofs ) )
        return false;

    tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVectorType > matrixSetter;
    matrixSetter.template getCompressedRowsLengths< Mesh::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            mesh,
            differentialOperator,
            boundaryConditions,
            rowsLengths );

    // sanity check (doesn't happen if the traverser works, but this is pretty
    // hard to debug and the check does not cost us much in initialization)
    if( rowsLengths.min() <= 0 ) {
        cerr << "Attempted to set invalid rowsLengths vector:" << endl << rowsLengths << endl;
        return false;
    }

    if( ! matrix.setDimensions( dofs, dofs ) )
        return false;
    if( ! matrix.setCompressedRowsLengths( rowsLengths ) )
        return false;
    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
makeSnapshot( const RealType & time,
              const IndexType & step,
              const MeshType & mesh,
              DofVectorType & dofVector,
              MeshDependentDataType & mdd )
{
    cout << endl << "Writing output at time " << time << " step " << step << endl;

    const IndexType cells = mesh.getNumberOfCells();

    if( ! mdd.makeSnapshot( time, step, mesh, outputPrefix ) )
        return false;

    // FIXME: TwoPhaseModel::makeSnapshotOnFaces does not work in 2D
//    if( ! mdd.makeSnapshotOnFaces( time, step, mesh, dofVector, outputPrefix ) )
//        return false;

//    cout << "solution (Z_iE): " << endl << dofVector << endl;
//    cout << "solution (Z_iK): " << endl << mdd.Z << endl;
//    cout << "mobility (m_iK): " << endl << mdd.m << endl;

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
preIterate( const RealType & time,
            const RealType & tau,
            const MeshType & mesh,
            DofVectorType & dofVector,
            MeshDependentDataType & mdd )
{
    // FIXME: nasty hack to pass tau to QRupdater
    mdd.current_tau = tau;

//    tnlTraverser< MeshType, MeshType::Dimensions > traverser;
//    traverser.template processInteriorEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );
//    traverser.template processBoundaryEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );

    tnlTraverser< MeshType, MeshType::Dimensions, MeshDependentDataType::NumberOfEquations > traverserND;
    timer_R.start();
//    traverserND.template processInteriorEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( mesh, mdd );
//    traverserND.template processBoundaryEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( mesh, mdd );
    traverserND.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( mesh, mdd );
    timer_R.stop();

    tnlTraverser< MeshType, MeshType::Dimensions > traverser;
    timer_Q.start();
//    traverser.template processInteriorEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( mesh, mdd );
//    traverser.template processBoundaryEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( mesh, mdd );
    traverser.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( mesh, mdd );
    timer_Q.stop();

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData,  DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
assemblyLinearSystem( const RealType & time,
                      const RealType & tau,
                      const MeshType & mesh,
                      DofVectorType & dofVector,
                      MatrixType & matrix,
                      DofVectorType & b,
                      MeshDependentDataType & mdd )
{
    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );

    // bind mesh-dependent data
    this->differentialOperator.bindMeshDependentData( mddDevicePtr.get() );
    this->boundaryConditions.bindMeshDependentData( mddDevicePtr.get() );
    this->rightHandSide.bindMeshDependentData( mddDevicePtr.get() );

    // initialize system assembler for stationary problem
    tnlLinearSystemAssembler< MeshType, DofVectorType, DifferentialOperator, BoundaryConditions, RightHandSide, tnlNoTimeDiscretisation, MatrixType > systemAssembler;
    systemAssembler.template assembly< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            time,
            tau,
            mesh,
            this->differentialOperator,
            this->boundaryConditions,
            this->rightHandSide,
            dofVector,
            matrix,
            b );

//    matrix.print( cout );
//    cout << b << endl;
//    if( time > tau )
//        abort();

//    tnlString matrixFileName, dofFileName, rhsFileName;
//    FileNameBaseNumberEnding( outputPrefix.getString(), time / tau, 5, "-matrix.tnl", matrixFileName );
//    FileNameBaseNumberEnding( outputPrefix.getString(), time / tau, 5, "-dof.vec.tnl", dofFileName );
//    FileNameBaseNumberEnding( outputPrefix.getString(), time / tau, 5, "-rhs.vec.tnl", rhsFileName );
//    matrix.save( matrixFileName );
//    dofVector.save( dofFileName );
//    b.save( rhsFileName );

//    // print matrix elements
//    for( IndexType i = 0; i < dofs; i++ ) {
//        for( IndexType j = 0; j < dofs; j++ ) {
//            RealType value = matrix.getElement( i, j );
//            if( value == 0 || value == 1 ) cout << value << " ";
//            else if( value > 0 ) cout << "+ ";
//            else if( value < 0 ) cout << "- ";
//        }
//        cout << "| " << b[ i ] << endl;
//    }
//    cout << matrix;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
postIterate( const RealType & time,
             const RealType & tau,
             const MeshType & mesh,
             DofVectorType & dofVector,
             MeshDependentDataType & mdd )
{
    // TODO: copying the objects to GPU takes as long as the rest of this method!
    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );
    device_ptr< BoundaryConditions, DeviceType > bcDevicePtr( boundaryConditions );

    this->boundaryConditions.bindMeshDependentData( mddDevicePtr.get() );

    timer_explicit.start();
    HybridizationExplicitFunction< MeshType, MeshDependentDataType > functionZK;
    functionZK.bind( mddDevicePtr.get(), dofVector );
    tnlFunctionEnumerator< MeshType, HybridizationExplicitFunction< MeshType, MeshDependentDataType >, DofVectorType > enumeratorZK;
    enumeratorZK.template enumerate< MeshType::Dimensions, MeshDependentDataType::NumberOfEquations >(
            mesh,
            functionZK,
            mdd.Z );
    timer_explicit.stop();

    // update non-linear terms
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType > genericEnumerator;
    genericEnumerator.template enumerate< &MeshDependentDataType::updateNonLinearTerms, MeshType::Dimensions >( mesh, mdd );
    timer_nonlinear.stop();

    // update upwind density values
    timer_upwind.start();
    Upwind< MeshType, MeshDependentDataType, BoundaryConditions > upwindFunction;
    upwindFunction.bind( mddDevicePtr.get(), bcDevicePtr.get(), dofVector );
    tnlFunctionEnumerator< MeshType, Upwind< MeshType, MeshDependentDataType, BoundaryConditions >, DofVectorType > upwindEnumerator;
    upwindEnumerator.template enumerate< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            mesh,
            upwindFunction,
            mdd.m_upw );
    timer_upwind.stop();

    // TODO
//    FaceAverageFunction< MeshType, RealType, IndexType > faceAverageFunction;
//    faceAverageFunction.bind( mdd.m );
//    tnlFunctionEnumerator< MeshType, FaceAverageFunction< MeshType, RealType, IndexType >, DofVectorType > faceAverageEnumerator;
//    faceAverageEnumerator.template enumerate< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
//            mesh,
//            faceAverageFunction,
//            mdd.m_upw );

//    cout << "solution (Z_iE): " << endl << dofVector << endl;
//    cout << "solution (Z_iK): " << endl << mdd.Z << endl;

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
writeEpilog( tnlLogger & logger )
{
    logger.writeParameter< double >( "update_R time:", timer_R.getTime() );
    logger.writeParameter< double >( "update_Q time:", timer_Q.getTime() );
    logger.writeParameter< double >( "Z_iF -> Z_iK update time:", timer_explicit.getTime() );
    logger.writeParameter< double >( "nonlinear update time:", timer_nonlinear.getTime() );
    logger.writeParameter< double >( "upwind update time:", timer_upwind.getTime() );
    return true;
}

} // namespace mhfem
