#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Functions/MeshFunction.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/GenericEnumerator.h"
#include "../lib_general/FaceAverageFunction.h"

#include "Solver.h"
#include "LocalUpdaters.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
TNL::String
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
getPrologHeader()
{
    return TNL::String( "NumDwarf solver" );
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
writeProlog( TNL::Logger & logger, const TNL::Config::ParameterContainer & parameters )
{
    logger.writeParameter< TNL::String >( "Output prefix:", parameters.getParameter< TNL::String >( "output-prefix" ) );
    logger.writeParameter< bool >( "Mesh ordering enabled:", parameters.getParameter< bool >( "reorder-mesh" ) );
    logger.writeSeparator();
    MeshDependentDataType::writeProlog( logger, parameters );
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
TNL::Solvers::SolverMonitor*
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
getSolverMonitor()
{
    return 0;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setMesh( MeshPointer & meshPointer )
{
    this->meshPointer = meshPointer;
    mdd->allocate( *meshPointer );
    differentialOperatorPointer->bind( meshPointer, mdd );
    boundaryConditionsPointer->bind( meshPointer, mdd );
    rightHandSidePointer->bind( meshPointer, mdd );
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setup( const TNL::Config::ParameterContainer & parameters,
       const TNL::String & prefix )
{
    // prefix for snapshots
    outputPrefix = parameters.getParameter< TNL::String >( "output-prefix" ) + TNL::String("-");
    doMeshOrdering = parameters.getParameter< bool >( "reorder-mesh" );

    // Our kernels for LocalUpdaters and DifferentialOperator have many local memory spills,
    // so this helps a lot. It does not affect TNL's reduction and multireduction algorithms,
    // which set cudaFuncCachePreferShared manually per kernel.
#ifdef HAVE_CUDA
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif

    if( doMeshOrdering )
        meshOrdering.reorder( *meshPointer );

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
typename Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::IndexType
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
getDofs() const
{
    // TODO: avoid the re-allocation on re-binding
    return MeshDependentDataType::NumberOfEquations * meshPointer->template getEntitiesCount< typename MeshType::Face >();
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
bindDofs( DofVectorPointer & dofVectorPointer )
{
    dofFunctionPointer->bind( meshPointer, dofVectorPointer );
}


template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setInitialCondition( const TNL::Config::ParameterContainer & parameters,
                     DofVectorPointer & dofVectorPointer )
{
    // HACK: we allocate DOFs as an NDArray (mdd->Z_iF) and bind the TNL's
    // DofVector to the underlying 1D array
    // (not related to setting initial condition, but this is the first place
    // where we have both dofVector and mdd together)
    dofVectorPointer->bind( mdd->Z_iF.getStorageArray() );

    bindDofs( dofVectorPointer );

    if( ! boundaryConditionsPointer->init( parameters, *meshPointer ) )
        return false;

    if( ! mdd->init( parameters, meshPointer ) )
        return false;

    if( doMeshOrdering ) {
        boundaryConditionsPointer->reorderBoundaryConditions( meshOrdering );
        mdd->reorderDofs( meshOrdering, false );
        meshOrdering.reset_vertices();
        meshOrdering.reset_faces();
    }

    // initialize dofVector as an average of mdd.Z on neighbouring cells
    // (this is not strictly necessary, we just provide an initial guess for
    // the iterative linear solver)
        // bind input
        using FaceAverageFunction = FaceAverageFunction< MeshType, MeshDependentDataType >;
        TNL::SharedPointer< FaceAverageFunction, DeviceType > faceAverageFunction;
        faceAverageFunction->bind( meshPointer, mdd );
        // evaluator
        TNL::Functions::MeshFunctionEvaluator< DofFunction, FaceAverageFunction > faceAverageEvaluator;
        // TODO: correctness of the evaluator depends on the specific layout of Z_iF
        faceAverageEvaluator.evaluate(
                dofFunctionPointer,     // out
                faceAverageFunction );  // in

    mdd->v_iKe.setValue( 0.0 );

    timer_b.reset();
    timer_R.reset();
    timer_Q.reset();
    timer_explicit.reset();
    timer_nonlinear.reset();
    timer_velocities.reset();
    timer_upwind.reset();

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setupLinearSystem( MatrixPointer & matrixPointer )
{
    using CompressedRowLengthsVectorType = typename MatrixType::CompressedRowLengthsVector;

    const IndexType dofs = this->getDofs();
    TNL::SharedPointer< CompressedRowLengthsVectorType > rowLengthsPointer;
    rowLengthsPointer->setSize( dofs );

    TNL::Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryConditions, CompressedRowLengthsVectorType > matrixSetter;
    matrixSetter.template getCompressedRowLengths< typename Mesh::Face >(
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

    matrixPointer->setDimensions( dofs, dofs );
    matrixPointer->setCompressedRowLengths( *rowLengthsPointer );
    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
makeSnapshot( const RealType & time,
              const IndexType & step,
              DofVectorPointer & dofVectorPointer )
{
    std::cout << std::endl << "Writing output at time " << time << " step " << step << std::endl;

    // reorder DOFs back to original numbering before snapshot
    if( doMeshOrdering )
        mdd->reorderDofs( meshOrdering, true );

    if( ! mdd->makeSnapshot( time, step, *meshPointer, outputPrefix ) )
        return false;

    // reorder DOFs back to the special numbering after snapshot
    if( doMeshOrdering )
        mdd->reorderDofs( meshOrdering, false );

    // FIXME: TwoPhaseModel::makeSnapshotOnFaces does not work in 2D
//    if( ! mdd->makeSnapshotOnFaces( time, step, mesh, dofVector, outputPrefix ) )
//        return false;

//    std::cout << "solution (Z_iE): " << std::endl << mdd->Z_iF.getStorageArray() << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z_iK.getStorageArray() << std::endl;
//    std::cout << "mobility (m_iK): " << std::endl << mdd->m << std::endl;

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
preIterate( const RealType & time,
            const RealType & tau,
            DofVectorPointer & dofVectorPointer )
{
    // FIXME
    mdd->current_time = time;

    // FIXME: nasty hack to pass tau to LocalUpdaters
    mdd->current_tau = tau;

    // not necessary for correctness, but for correct timings
    #ifdef HAVE_CUDA
    TNL::Devices::Cuda::synchronizeDevice();
    #endif

    // update non-linear terms
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType >::
        template enumerate< &MeshDependentDataType::updateNonLinearTerms, typename MeshType::Cell >( meshPointer, mdd );
    timer_nonlinear.stop();

    // update coefficients b_ijKEF
    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell, MeshDependentDataType::NumberOfEquations > traverser_Ki;
    timer_b.start();
    traverser_Ki.template processAllEntities< MeshDependentDataType, typename LocalUpdaters< MeshType, MeshDependentDataType >::update_b >( meshPointer, mdd );
    timer_b.stop();

    // update vector coefficients (u, w, a), whose projection into the RTN_0(K) space
    // generally depends on the b_ijKEF coefficients
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType >::
        template enumerate< &MeshDependentDataType::updateVectorCoefficients,
                            typename MeshType::Cell,
                            MeshDependentDataType::NumberOfEquations >( meshPointer, mdd );
    timer_nonlinear.stop();

    // update upwinded mobility values
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_upwind.start();
        // bind output
        upwindMeshFunction->bind( meshPointer, mdd->m_iE_upw.getStorageArray() );
        // bind inputs
        upwindFunction->bind( meshPointer, mdd, boundaryConditionsPointer );
        // evaluate
        upwindEvaluator.evaluate(
                upwindMeshFunction,     // out
                upwindFunction );       // in

        // upwind Z_ijE_upw (this needs the a_ij and u_ij coefficients)
        timer_upwind.start();
        // bind output
        upwindZMeshFunction->bind( meshPointer, mdd->Z_ijE_upw.getStorageArray() );
        // bind inputs
        upwindZFunction->bind( meshPointer, mdd, *dofVectorPointer );
        // evaluate
        upwindZEvaluator.evaluate(
                upwindZMeshFunction,     // out
                upwindZFunction );       // in
    timer_upwind.stop();

    timer_R.start();
    traverser_Ki.template processAllEntities< MeshDependentDataType, typename LocalUpdaters< MeshType, MeshDependentDataType >::update_R >( meshPointer, mdd );
    timer_R.stop();

    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell > traverser_K;
    timer_Q.start();
    traverser_K.template processAllEntities< MeshDependentDataType, typename LocalUpdaters< MeshType, MeshDependentDataType >::update_Q >( meshPointer, mdd );
    timer_Q.stop();

//    std::cout << "N = " << mdd->N << std::endl;
//    std::cout << "u = " << mdd->u << std::endl;
//    std::cout << "m = " << mdd->m << std::endl;
//    std::cout << "D = " << mdd->D << std::endl;
//    std::cout << "w = " << mdd->w << std::endl;
//    std::cout << "a = " << mdd->a << std::endl;
//    std::cout << "r = " << mdd->r << std::endl;
//    std::cout << "f = " << mdd->f << std::endl;

//    std::cout << "b = " << mdd->b << std::endl;
//    std::cout << "m_upw = " << mdd->m_iE_upw.getStorageArray() << std::endl;
//    std::cout << "R_ijKF = " << mdd->R1 << std::endl;
//    std::cout << "R_iK = " << mdd->R2 << std::endl;

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
assemblyLinearSystem( const RealType & time,
                      const RealType & tau,
                      DofVectorPointer & dofVectorPointer,
                      MatrixPointer & matrixPointer,
                      DofVectorPointer & bPointer )
{
    // Setting this here instead of some setup method ensures that
    // the systemAssembler always has the correct operator etc.
    systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
    systemAssembler.setBoundaryConditions( this->boundaryConditionsPointer );
    systemAssembler.setRightHandSide( this->rightHandSidePointer );

    // initialize system assembler for stationary problem
    systemAssembler.template assembly< typename MeshType::Face >(
            time,
            tau,
            meshPointer,
            dofFunctionPointer,
            matrixPointer,
            bPointer );

//    (*matrixPointer).print( std::cout );
//    std::cout << *bPointer << std::endl;
//    if( time > tau )
//        abort();

//    static IndexType iter = 0;
//    TNL::String matrixFileName = outputPrefix + "matrix.tnl";
//    TNL::String dofFileName = outputPrefix + "dof.vec.tnl";
//    TNL::String rhsFileName = outputPrefix + "rhs.vec.tnl";
//    if( iter == 10 ) {
//        matrixPointer->save( matrixFileName );
//        dofVectorPointer->save( dofFileName );
//        bPointer->save( rhsFileName );
//    }
//    iter++;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
saveFailedLinearSystem( const Matrix & matrix,
                        const DofVectorType & dofs,
                        const DofVectorType & rhs ) const
{
    matrix.save( outputPrefix + "matrix.tnl" );
    dofs.save( outputPrefix + "dof.vec.tnl" );
    rhs.save( outputPrefix + "rhs.vec.tnl" );
    std::cerr << "The linear system has been saved to " << outputPrefix << "{matrix,dof.vec,rhs.vec}.tnl" << std::endl;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
postIterate( const RealType & time,
             const RealType & tau,
             DofVectorPointer & dofVectorPointer )
{
    timer_explicit.start();
    // bind output
    meshFunctionZK->bind( meshPointer, mdd->Z_iK.getStorageArray() );
    // bind inputs
    functionZK->bind( meshPointer, mdd );
    // evaluate
    evaluatorZK.evaluate( meshFunctionZK, functionZK );
    timer_explicit.stop();

    // update coefficients of the conservative velocities
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_velocities.start();
    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell, MeshDependentDataType::NumberOfEquations > traverser_Ki;
    traverser_Ki.template processAllEntities< MeshDependentDataType, typename LocalUpdaters< MeshType, MeshDependentDataType >::update_v >( meshPointer, mdd );
    timer_velocities.stop();

//    std::cout << "solution (Z_iE): " << std::endl << *dofVectorPointer << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z_iK.getStorageArray() << std::endl;
//    std::cin.get();

    return true;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
writeEpilog( TNL::Logger & logger )
{
    logger.writeParameter< double >( "nonlinear update time:", timer_nonlinear.getRealTime() );
    logger.writeParameter< double >( "update_b time:", timer_b.getRealTime() );
    logger.writeParameter< double >( "upwind update time:", timer_upwind.getRealTime() );
    logger.writeParameter< double >( "update_R time:", timer_R.getRealTime() );
    logger.writeParameter< double >( "update_Q time:", timer_Q.getRealTime() );
    logger.writeParameter< double >( "Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    logger.writeParameter< double >( "velocities update time:", timer_velocities.getRealTime() );
    return true;
}

} // namespace mhfem
