#pragma once

#include <core/mfilename.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Solvers/PDE/NoTimeDiscretisation.h>

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
TNL::String
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getTypeStatic()
{
    return TNL::String( "Solver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
TNL::String
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
getPrologHeader() const
{
    // TODO
//    return TNL::String( "Single-phase flow" );
    return TNL::String();
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
writeProlog( TNL::Logger & logger, const TNL::Config::ParameterContainer & parameters ) const
{
    logger.writeParameter< TNL::String >( "Output prefix:", parameters.getParameter< TNL::String >( "output-prefix" ) );
    // TODO: let models write their parameters
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
TNL::Solvers::SolverMonitor< typename Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::RealType,
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
setup( const TNL::Config::ParameterContainer & parameters )
{
    // prefix for snapshots
    outputPrefix = parameters.getParameter< TNL::String >( "output-prefix" ) + TNL::String("-");

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
getDofs( const MeshPointer & meshPointer ) const
{
    return MeshDependentDataType::NumberOfEquations * meshPointer->template getEntitiesCount< typename MeshType::Face >();
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
bindDofs( const MeshPointer & meshPointer,
          DofVectorPointer & dofVectorPointer )
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
bindMeshDependentData( const MeshPointer & mesh,
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
setInitialCondition( const TNL::Config::ParameterContainer & parameters,
                     const MeshPointer & meshPointer,
                     DofVectorPointer & dofVectorPointer,
                     MeshDependentDataType & mdd )
{
    if( ! boundaryConditionsPointer->init( parameters, *meshPointer ) )
        return false;

    if( ! mdd.init( parameters, meshPointer, boundaryConditionsPointer ) )
        return false;

    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );

    // initialize dofVector as an average of mdd.Z on neighbouring cells
    FaceAverageFunctionWithBoundary< MeshType, MeshDependentDataType, BoundaryConditions > faceAverageFunction;
    faceAverageFunction.bind( mddDevicePtr.get(), boundaryConditionsPointer, mdd.Z );
    TNL::Functions::MeshFunctionEvaluator< MeshType, FaceAverageFunctionWithBoundary< MeshType, MeshDependentDataType, BoundaryConditions >, DofVectorType > faceAverageEnumerator;
    faceAverageEnumerator.template enumerate< MeshType::meshDimensions - 1, MeshDependentDataType::NumberOfEquations >(
            meshPointer,
            faceAverageFunction,
            dofVectorPointer );

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
setupLinearSystem( const MeshPointer & meshPointer,
                   MatrixPointer & matrixPointer )
{
    typedef typename MatrixType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;

    const IndexType dofs = this->getDofs( meshPointer );
    TNL::SharedPointer< CompressedRowsLengthsVectorType > rowLengthsPointer;
    if( ! rowLengthsPointer->setSize( dofs ) )
        return false;

    TNL::Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVectorType > matrixSetter;
    matrixSetter.template getCompressedRowsLengths< typename Mesh::Face, MeshDependentDataType::NumberOfEquations >(
            meshPointer,
            differentialOperatorPointer,
            boundaryConditionsPointer,
            rowLengthsPointer );

    // sanity check (doesn't happen if the traverser works, but this is pretty
    // hard to debug and the check does not cost us much in initialization)
    if( rowLengthsPointer->min() <= 0 ) {
        std::cerr << "Attempted to set invalid rowsLengths vector:" << std::endl << *rowLengthsPointer << std::endl;
        return false;
    }

    if( ! matrixPointer->setDimensions( dofs, dofs ) )
        return false;
    if( ! matrixPointer->setCompressedRowsLengths( *rowLengthsPointer ) )
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
              const MeshPointer & meshPointer,
              DofVectorPointer & dofVectorPointer,
              MeshDependentDataType & mdd )
{
    std::cout << std::endl << "Writing output at time " << time << " step " << step << std::endl;

    const IndexType cells = meshPointer->template getEntitiesCount< typename Mesh::Cell >();

    if( ! mdd.makeSnapshot( time, step, *meshPointer, outputPrefix ) )
        return false;

    // FIXME: TwoPhaseModel::makeSnapshotOnFaces does not work in 2D
//    if( ! mdd.makeSnapshotOnFaces( time, step, mesh, dofVector, outputPrefix ) )
//        return false;

//    std::cout << "solution (Z_iE): " << std::endl << dofVector << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd.Z << std::endl;
//    std::cout << "mobility (m_iK): " << std::endl << mdd.m << std::endl;

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
            const MeshPointer & meshPointer,
            DofVectorPointer & dofVectorPointer,
            MeshDependentDataType & mdd )
{
    // FIXME: nasty hack to pass tau to QRupdater
    mdd.current_tau = tau;

//    TNL::Meshes::Traverser< MeshType, MeshType::meshDimensions > traverser;
//    traverser.template processInteriorEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );
//    traverser.template processBoundaryEntities< MeshDependentDataType, QRupdater< MeshType, MeshDependentDataType > >( mesh, mdd );

    TNL::Meshes::Traverser< MeshType, MeshType::meshDimensions, MeshDependentDataType::NumberOfEquations > traverserND;
    timer_R.start();
//    traverserND.template processInteriorEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( mesh, mdd );
//    traverserND.template processBoundaryEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( mesh, mdd );
    traverserND.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( *meshPointer, mdd );
    timer_R.stop();

    TNL::Meshes::Traverser< MeshType, MeshType::meshDimensions > traverser;
    timer_Q.start();
//    traverser.template processInteriorEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( mesh, mdd );
//    traverser.template processBoundaryEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( mesh, mdd );
    traverser.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( *meshPointer, mdd );
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
                      const MeshPointer & meshPointer,
                      DofVectorPointer & dofVectorPointer,
                      MatrixPointer & matrixPointer,
                      DofVectorPointer & bPointer,
                      MeshDependentDataType & mdd )
{
    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );

    // bind mesh-dependent data
    this->differentialOperatorPointer->bindMeshDependentData( mddDevicePtr.get() );
    this->boundaryConditionsPointer->bindMeshDependentData( mddDevicePtr.get() );
    this->rightHandSidePointer->bindMeshDependentData( mddDevicePtr.get() );

    // initialize system assembler for stationary problem
    TNL::Solvers::PDE::LinearSystemAssembler< MeshType, MeshFunctionType, DifferentialOperator, BoundaryConditions, RightHandSide, TNL::Solvers::PDE::NoTimeDiscretisation, MatrixType, DofVectorType > systemAssembler;
    systemAssembler.template assembly< MeshType::meshDimensions - 1, MeshDependentDataType::NumberOfEquations >(
            time,
            tau,
            meshPointer,
            this->differentialOperatorPointer,
            this->boundaryConditionsPointer,
            this->rightHandSidePointer,
            dofVectorPointer,
            matrixPointer,
            bPointer );

//    matrix.print( std::cout );
//    std::cout << b << std::endl;
//    if( time > tau )
//        abort();

//    TNL::String matrixFileName, dofFileName, rhsFileName;
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
//            if( value == 0 || value == 1 ) std::cout << value << " ";
//            else if( value > 0 ) std::cout << "+ ";
//            else if( value < 0 ) std::cout << "- ";
//        }
//        std::cout << "| " << b[ i ] << std::endl;
//    }
//    std::cout << matrix;
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
             const MeshPointer & meshPointer,
             DofVectorPointer & dofVectorPointer,
             MeshDependentDataType & mdd )
{
    // TODO: copying the objects to GPU takes as long as the rest of this method!
    device_ptr< MeshDependentDataType, DeviceType > mddDevicePtr( mdd );

    this->boundaryConditionsPointer->bindMeshDependentData( mddDevicePtr.get() );

    timer_explicit.start();
    HybridizationExplicitFunction< MeshType, MeshDependentDataType > functionZK;
    functionZK.bind( mddDevicePtr.get(), dofVectorPointer );
    tnlFunctionEvaluator< MeshType, HybridizationExplicitFunction< MeshType, MeshDependentDataType >, DofVectorType > enumeratorZK;
    enumeratorZK.template enumerate< MeshType::meshDimensions, MeshDependentDataType::NumberOfEquations >(
            meshPointer,
            functionZK,
            mdd.Z );
    timer_explicit.stop();

    // update non-linear terms
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType > genericEnumerator;
    genericEnumerator.template enumerate< &MeshDependentDataType::updateNonLinearTerms, MeshType::meshDimensions >( meshPointer, mdd );
    timer_nonlinear.stop();

    // update upwind density values
    timer_upwind.start();
    Upwind< MeshType, MeshDependentDataType, BoundaryConditions > upwindFunction;
    upwindFunction.bind( mddDevicePtr.get(), boundaryConditionsPointer, dofVectorPointer );
    tnlFunctionEvaluator< MeshType, Upwind< MeshType, MeshDependentDataType, BoundaryConditions >, DofVectorType > upwindEnumerator;
    upwindEnumerator.template enumerate< MeshType::meshDimensions - 1, MeshDependentDataType::NumberOfEquations >(
            meshPointer,
            upwindFunction,
            mdd.m_upw );
    timer_upwind.stop();

    // TODO
//    FaceAverageFunction< MeshType, RealType, IndexType > faceAverageFunction;
//    faceAverageFunction.bind( mdd.m );
//    tnlFunctionEvaluator< MeshType, FaceAverageFunction< MeshType, RealType, IndexType >, DofVectorType > faceAverageEnumerator;
//    faceAverageEnumerator.template enumerate< MeshType::meshDimensions - 1, MeshDependentDataType::NumberOfEquations >(
//            mesh,
//            faceAverageFunction,
//            mdd.m_upw );

//    std::cout << "solution (Z_iE): " << std::endl << dofVector << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd.Z << std::endl;

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
writeEpilog( TNL::Logger & logger )
{
    logger.writeParameter< double >( "update_R time:", timer_R.getTime() );
    logger.writeParameter< double >( "update_Q time:", timer_Q.getTime() );
    logger.writeParameter< double >( "Z_iF -> Z_iK update time:", timer_explicit.getTime() );
    logger.writeParameter< double >( "nonlinear update time:", timer_nonlinear.getTime() );
    logger.writeParameter< double >( "upwind update time:", timer_upwind.getTime() );
    return true;
}

} // namespace mhfem
