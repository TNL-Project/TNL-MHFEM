#pragma once

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include "Solver.h"

#include "../mesh_helpers.h"
#include "../GenericEnumerator.h"
#include "../FaceAverageFunction.h"
#include "../device_ptr.h"

#include <functors/tnlFunctionEnumerator.h>
#include <solvers/pde/tnlNoTimeDiscretisation.h>
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
    return tnlString( "Single-phase flow" );
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
    // TODO: porosity
    logger.writeParameter< double >( "Permeability:", parameters.getParameter< double >( "permeability" ) );
    logger.writeParameter< double >( "Fluid viscosity:", parameters.getParameter< double >( "viscosity" ) );
    logger.writeParameter< double >( "Fluid molar mass:", parameters.getParameter< double >( "molar-mass" ) );
    logger.writeParameter< double >( "Gas constant:", parameters.getParameter< double >( "gas-constant" ) );
    logger.writeParameter< double >( "Temperature:", parameters.getParameter< double >( "temperature" ) );
    logger.writeParameter< double >( "Gravity:", parameters.getParameter< double >( "y-gravity" ) );
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
    // save value of initial time (defined by tnlPDESolver)
    initialTime = parameters.getParameter< double >( "initial-time" );

    // prefix for snapshots
    outputPrefix = parameters.getParameter< tnlString >( "output-prefix" ) + tnlString("-");

    // TODO: load boundary conditions from file
    return true;
//    return boundaryConditions.setup( parameters );
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
    return MeshDependentDataType::NumberOfEquations * mesh.template getNumberOfFaces< 1, 1 >();
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
{
    const IndexType dofs = this->getDofs( mesh );
    this->ptrace.bind( dofVector.getData(), dofs );
}

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
{
    // TODO: does not work for arbitrary object !!!
    //       but should not be needed anymore (mdd can be passed by reference along with mesh
//    this->pressure.bind( mdd.getData(), mdd.getSize() );
}


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
    // TODO: load boundary conditions from file
    if( ! boundaryConditions.init( parameters, mesh ) )
        return false;

    if( ! mdd.init( parameters, mesh ) )
        return false;

    // initialize dofVector as an average of mdd.Z on neighbouring cells
    FaceAverageFunction< MeshType, RealType, IndexType > faceAverageFunction;
    faceAverageFunction.bind( mdd.Z );
    tnlFunctionEnumerator< MeshType, FaceAverageFunction< MeshType, RealType, IndexType >, DofVectorType > faceAverageEnumerator;
    faceAverageEnumerator.template enumerate< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            mesh,
            faceAverageFunction,
            dofVector );

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
    matrix.setDimensions( dofs, dofs );
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
    bindDofs( mesh, dofVector );
//    bindMeshDependentData( mesh, mdd );

    cout << endl << "Writing output at time " << time << " step " << step << endl;

    // TODO: move everything to mdd.makeSnapshot (and some command-line parameter should select whether to save dofVector or auxiliaryDofVector (from mdd.makeSnapshot))

    const IndexType cells = mesh.getNumberOfCells();

    // NOTE: depends on the indexation of mdd.Z vector
    for( int i = 0; i < mdd.n; i++ ) {
        tnlString fileName;
        FileNameBaseNumberEnding( (outputPrefix + "Z" + convertToString( i ) + "-").getString(), step, 5, ".tnl", fileName );
        SharedVectorType dofPhase( &mdd.Z[ 0 ] + i * cells, cells );
        if( ! dofPhase.save( fileName ) )
           return false;
    }

    if( ! mdd.makeSnapshot( time, step, mesh, outputPrefix ) )
        return false;

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
preIterate( const RealType & time,
            const RealType & tau,
            const MeshType & mesh,
            DofVectorType & dofVector,
            MeshDependentDataType & mdd )
{
    bindDofs( mesh, dofVector );
//    bindMeshDependentData( mesh, mdd );

    // FIXME: nasty hack to pass tau to QRupdater
    mdd.current_tau = tau;

    tnlTraverser< MeshType, MeshType::Dimensions > traverser;
    traverser.template processInteriorEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );
    traverser.template processBoundaryEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );

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
    bindDofs( mesh, dofVector );
//    bindMeshDependentData( mesh, mdd );

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
            dofVector,   // ptrace as DofVectorType
            matrix,
            b );

//    matrix.print( cout );
//    cout << b << endl;
//    if( time > 0 )
//        abort();

//    tnlString matrixFileName, rhsFileName;
//    FileNameBaseNumberEnding( outputPrefix.getString(), time / tau, 5, ".csr.tnl", matrixFileName );
//    FileNameBaseNumberEnding( outputPrefix.getString(), time / tau, 5, "-rhs.vec.tnl", rhsFileName );
//    matrix.save( matrixFileName );
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
    bindDofs( mesh, dofVector );
//    bindMeshDependentData( mesh, mdd );

    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );

    HybridizationExplicitFunction< MeshType, MeshDependentDataType > functionZK;
    functionZK.bind( mddDevicePtr.get(), dofVector );
    tnlFunctionEnumerator< MeshType, HybridizationExplicitFunction< MeshType, MeshDependentDataType >, DofVectorType > enumeratorZK;
    enumeratorZK.template enumerate< MeshType::Dimensions, MeshDependentDataType::NumberOfEquations >(
            mesh,
            functionZK,
            mdd.Z );

    // update non-linear terms
    GenericEnumerator< MeshType, MeshDependentDataType > genericEnumerator;
    genericEnumerator.template enumerate< &MeshDependentDataType::updateNonLinearTerms, MeshType::Dimensions >( mesh, mdd );

    // update upwind density values
    Upwind< MeshType, MeshDependentDataType > upwindFunction;
    upwindFunction.bind( mddDevicePtr.get(), dofVector );
    tnlFunctionEnumerator< MeshType, Upwind< MeshType, MeshDependentDataType >, DofVectorType > upwindEnumerator;
    upwindEnumerator.template enumerate< MeshType::Dimensions - 1, MeshDependentDataType::NumberOfEquations >(
            mesh,
            upwindFunction,
            mdd.m_upw );

//    cout << "solution (Z_iE): " << endl << dofVector << endl;
//    cout << "solution (Z_iK): " << endl << mdd.Z << endl;

    return true;
}

} // namespace mhfem
