#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Functions/MeshFunction.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/GenericEnumerator.h"
#include "../lib_general/FaceAverageFunction.h"

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
TNL::Solvers::SolverMonitor*
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
setup( const MeshPointer & meshPointer,
       const TNL::Config::ParameterContainer & parameters,
       const TNL::String & prefix )
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
{
    dofFunctionPointer->bind( meshPointer, dofVectorPointer );
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
setMeshDependentData( const MeshPointer & meshPointer,
                      MeshDependentDataPointer & mdd )
{
    return mdd->allocate( *meshPointer );
}

template< typename Mesh,
          typename MeshDependentData,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, DifferentialOperator, BoundaryConditions, RightHandSide, Matrix >::
bindMeshDependentData( const MeshPointer & meshPointer,
                       MeshDependentDataPointer & mdd )
{
    this->differentialOperatorPointer->bindMeshDependentData( mdd );
    this->boundaryConditionsPointer->bindMeshDependentData( mdd );
    this->rightHandSidePointer->bindMeshDependentData( mdd );
}


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
                     MeshDependentDataPointer & mdd )
{
    bindDofs( meshPointer, dofVectorPointer );
    bindMeshDependentData( meshPointer, mdd );

    if( ! boundaryConditionsPointer->init( parameters, *meshPointer ) )
        return false;

    if( ! mdd->init( parameters, meshPointer, boundaryConditionsPointer, mdd ) )
        return false;

    // initialize dofVector as an average of mdd.Z on neighbouring cells
    using FaceAverageFunction = FaceAverageFunctionWithBoundary< MeshType, MeshDependentDataType, BoundaryConditions >;
    // TODO: this might as well be a class attribute
    TNL::SharedPointer< FaceAverageFunction, DeviceType > faceAverageFunction;
    faceAverageFunction->bind( mdd, boundaryConditionsPointer, mdd->Z );
    TNL::Functions::MeshFunctionEvaluator< DofFunction, FaceAverageFunction > faceAverageEvaluator;
    faceAverageEvaluator.evaluate(
            dofFunctionPointer,     // out
            faceAverageFunction );  // in

    timer_R.reset();
    timer_Q.reset();
    timer_explicit.reset();
    timer_nonlinear.reset();

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
    matrixSetter.template getCompressedRowsLengths< typename Mesh::Face >(
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
              MeshDependentDataPointer & mdd )
{
    bindDofs( meshPointer, dofVectorPointer );
    bindMeshDependentData( meshPointer, mdd );

    std::cout << std::endl << "Writing output at time " << time << " step " << step << std::endl;

    const IndexType cells = meshPointer->template getEntitiesCount< typename Mesh::Cell >();

    if( ! mdd->makeSnapshot( time, step, *meshPointer, outputPrefix ) )
        return false;

    // FIXME: TwoPhaseModel::makeSnapshotOnFaces does not work in 2D
//    if( ! mdd->makeSnapshotOnFaces( time, step, mesh, dofVector, outputPrefix ) )
//        return false;

//    std::cout << "solution (Z_iE): " << std::endl << dofVector << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z << std::endl;
//    std::cout << "mobility (m_iK): " << std::endl << mdd->m << std::endl;

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
            MeshDependentDataPointer & mdd )
{
    bindDofs( meshPointer, dofVectorPointer );
    bindMeshDependentData( meshPointer, mdd );

    // FIXME: nasty hack to pass tau to QRupdater
    mdd->current_tau = tau;

    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell, MeshDependentDataType::NumberOfEquations > traverserND;
    timer_R.start();
    traverserND.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_R >( meshPointer, mdd );
    timer_R.stop();

    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell > traverser;
    timer_Q.start();
    traverser.template processAllEntities< MeshDependentDataType, typename QRupdater< MeshType, MeshDependentDataType >::update_Q >( meshPointer, mdd );
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
                      MeshDependentDataPointer & mdd )
{
    bindDofs( meshPointer, dofVectorPointer );
    bindMeshDependentData( meshPointer, mdd );

    // initialize system assembler for stationary problem
    systemAssembler.template assembly< typename MeshType::Face >(
            time,
            tau,
            meshPointer,
            this->differentialOperatorPointer,
            this->boundaryConditionsPointer,
            this->rightHandSidePointer,
            dofFunctionPointer,
            matrixPointer,
            bPointer );

//    (*matrixPointer).print( std::cout );
//    std::cout << *bPointer << std::endl;
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
             MeshDependentDataPointer & mdd )
{
    bindDofs( meshPointer, dofVectorPointer );
    bindMeshDependentData( meshPointer, mdd );

    timer_explicit.start();
    // output
    using ZkMeshFunction = TNL::Functions::MeshFunction< MeshType, MeshType::meshDimensions, RealType, MeshDependentDataType::NumberOfEquations >;
    // TODO: might be class attribute
    TNL::SharedPointer< ZkMeshFunction, DeviceType > meshFunctionZK;
    meshFunctionZK->bind( meshPointer, mdd->Z );
    // input
    using HybridizationFunction = HybridizationExplicitFunction< MeshType, MeshDependentDataType >;
    TNL::SharedPointer< HybridizationFunction, DeviceType > functionZK;
    functionZK->bind( mdd, *dofVectorPointer );
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< ZkMeshFunction, HybridizationFunction > evaluatorZK;
    evaluatorZK.evaluate( meshFunctionZK, functionZK );
    timer_explicit.stop();

    // update non-linear terms
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType > genericEnumerator;
    genericEnumerator.template enumerate< &MeshDependentDataType::updateNonLinearTerms, typename MeshType::Cell >( meshPointer, mdd );
    timer_nonlinear.stop();

    // update upwind density values
    timer_upwind.start();
    // output
    TNL::SharedPointer< DofFunction, DeviceType > upwindMeshFunction;
    upwindMeshFunction->bind( meshPointer, mdd->m_upw );
    // input
    using UpwindFunction = Upwind< MeshType, MeshDependentDataType, BoundaryConditions >;
    TNL::SharedPointer< UpwindFunction, DeviceType > upwindFunction;
    upwindFunction->bind( mdd, boundaryConditionsPointer, *dofVectorPointer );
    // evaluator
    TNL::Functions::MeshFunctionEvaluator< DofFunction, UpwindFunction > upwindEvaluator;
    upwindEvaluator.evaluate( upwindMeshFunction, upwindFunction );
    timer_upwind.stop();

    // TODO
//    FaceAverageFunction< MeshType, RealType, IndexType > faceAverageFunction;
//    faceAverageFunction.bind( mdd->m );
//    tnlFunctionEvaluator< MeshType, FaceAverageFunction< MeshType, RealType, IndexType >, DofVectorType > faceAverageEnumerator;
//    faceAverageEnumerator.template enumerate< MeshType::meshDimensions - 1, MeshDependentDataType::NumberOfEquations >(
//            mesh,
//            faceAverageFunction,
//            mdd->m_upw );

//    std::cout << "solution (Z_iE): " << std::endl << dofVector << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z << std::endl;

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
    logger.writeParameter< double >( "update_R time:", timer_R.getRealTime() );
    logger.writeParameter< double >( "update_Q time:", timer_Q.getRealTime() );
    logger.writeParameter< double >( "Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    logger.writeParameter< double >( "nonlinear update time:", timer_nonlinear.getRealTime() );
    logger.writeParameter< double >( "upwind update time:", timer_upwind.getRealTime() );
    return true;
}

} // namespace mhfem
