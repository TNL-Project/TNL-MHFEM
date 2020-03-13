#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/GenericEnumerator.h"

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
    if( ! meshPointer ) {
        std::cerr << "The meshPointer is NULL, the setMesh method must be called first." << std::endl;
        return false;
    }

    // prefix for snapshots
    outputDirectory = parameters.getParameter< TNL::String >( "output-directory" );
    doMeshOrdering = parameters.getParameter< bool >( "reorder-mesh" );

    // set up the linear solver
    const TNL::String& linearSolverName = parameters.getParameter< TNL::String >( "linear-solver" );
    linearSystemSolver = TNL::Solvers::getLinearSolver< MatrixType >( linearSolverName );
    if( ! linearSystemSolver )
        return false;
    if( ! linearSystemSolver->setup( parameters ) )
        return false;

    // set up the preconditioner
    const TNL::String& preconditionerName = parameters.getParameter< TNL::String >( "preconditioner" );
    preconditioner = TNL::Solvers::getPreconditioner< MatrixType >( preconditionerName );
    if( preconditioner ) {
        linearSystemSolver->setPreconditioner( preconditioner );
        if( ! preconditioner->setup( parameters ) )
            return false;
    }

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
    return MeshDependentDataType::NumberOfEquations * meshPointer->template getEntitiesCount< typename MeshType::Face >();
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
typename Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::MeshDependentDataPointer&
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
getMeshDependentData()
{
    return mdd;
}


template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setInitialCondition( const TNL::Config::ParameterContainer & parameters )
{
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

    #ifdef HAVE_CUDA
    // make sure that meshPointer and mdd are synchronized
    TNL::Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
    #endif

    // initialize mdd.Z_iF as an average of mdd.Z on neighbouring cells
    // (this is not strictly necessary, we just provide an initial guess for
    // the iterative linear solver)
    const Mesh* _mesh = &meshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    auto faceAverageKernel = [_mesh, _mdd] __cuda_callable__ ( int i, IndexType E )
    {
        IndexType cellIndexes[ 2 ] = {0, 0};
        const auto & entity = _mesh->template getEntity< typename Mesh::Face >( E );
        int numCells = getCellsForFace( *_mesh, entity, cellIndexes );

        // NOTE: using the boundary condition is too much work, because it might be
        // uninitialized at this point, and it does not help that much to be worth it
        if( numCells == 1 ) {
            _mdd->Z_iF( i, E ) = _mdd->Z_iK( i, cellIndexes[ 0 ] );
        }
        else {
            _mdd->Z_iF( i, E ) = 0.5 * ( _mdd->Z_iK( i, cellIndexes[ 0 ] )
                                       + _mdd->Z_iK( i, cellIndexes[ 1 ] ) );
        }
    };
    const IndexType faces = meshPointer->template getEntitiesCount< typename Mesh::Face >();
    TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                        MeshDependentDataType::NumberOfEquations, faces,
                                                        faceAverageKernel );

    mdd->v_iKe.setValue( 0.0 );

    // reset output/profiling variables
    allIterations = 0;
    timer_preIterate.reset();
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
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
setupLinearSystem()
{
    using CompressedRowLengthsVectorType = typename MatrixType::CompressedRowLengthsVector;

    const IndexType dofs = this->getDofs();
    TNL::Pointers::SharedPointer< CompressedRowLengthsVectorType > rowLengthsPointer;
    rowLengthsPointer->setSize( dofs );
    rowLengthsPointer->setValue( -1 );

    TNL::Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryConditions, CompressedRowLengthsVectorType > matrixSetter;
    matrixSetter.template getCompressedRowLengths< typename Mesh::Face >(
            meshPointer,
            differentialOperatorPointer,
            boundaryConditionsPointer,
            rowLengthsPointer );

    // sanity check (doesn't happen if the traverser works, but this is pretty
    // hard to debug and the check does not cost us much in initialization)
    if( TNL::min( *rowLengthsPointer ) <= 0 ) {
        std::stringstream ss;
        ss << "MHFEM error: attempted to set invalid rowsLengths vector:\n" << *rowLengthsPointer << std::endl;
        throw std::runtime_error( ss.str() );
    }

    // initialize matrix
    matrixPointer->setDimensions( dofs, dofs );
    matrixPointer->setCompressedRowLengths( *rowLengthsPointer );

    // initialize the right hand side vector
    rhsVector.setSize( dofs );

    // set the matrix for the linear solver
    linearSystemSolver->setMatrix( matrixPointer );
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
bool
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
makeSnapshot( const RealType & time,
              const IndexType & step )
{
    // TODO: write only into the log file
//    std::cout << std::endl << "Writing output at time " << time << " step " << step << std::endl;

    // reorder DOFs back to original numbering before snapshot
    if( doMeshOrdering )
        mdd->reorderDofs( meshOrdering, true );

    if( ! mdd->makeSnapshot( time, step, *meshPointer, outputDirectory + "/" ) )
        return false;

    // reorder DOFs back to the special numbering after snapshot
    if( doMeshOrdering )
        mdd->reorderDofs( meshOrdering, false );

    // FIXME: TwoPhaseModel::makeSnapshotOnFaces does not work in 2D
//    if( ! mdd->makeSnapshotOnFaces( time, step, mesh, dofVector, outputDirectory + "/" ) )
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
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
preIterate( const RealType & time,
            const RealType & tau )
{
    timer_preIterate.start();

    // FIXME
    mdd->current_time = time;

    // FIXME: nasty hack to pass tau to LocalUpdaters
    mdd->current_tau = tau;

    // not necessary for correctness, but for correct timings
    #ifdef HAVE_CUDA
    TNL::Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
    #endif

    auto* mdd_device = &mdd.template modifyData< DeviceType >();

    // update non-linear terms
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType >::
        template enumerate< &MeshDependentDataType::updateNonLinearTerms, typename MeshType::Cell >( meshPointer, mdd_device );
    timer_nonlinear.stop();

    // update coefficients b_ijKEF
    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell, MeshDependentDataType::NumberOfEquations > traverser_Ki;
    timer_b.start();
    traverser_Ki.template processAllEntities< typename LocalUpdaters< MeshType, MeshDependentDataType >::update_b >( meshPointer, mdd_device );
    timer_b.stop();

    // update vector coefficients (u, w, a), whose projection into the RTN_0(K) space
    // generally depends on the b_ijKEF coefficients
    timer_nonlinear.start();
    GenericEnumerator< MeshType, MeshDependentDataType >::
        template enumerate< &MeshDependentDataType::updateVectorCoefficients,
                            typename MeshType::Cell,
                            MeshDependentDataType::NumberOfEquations >( meshPointer, mdd_device );
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
        upwindZFunction->bind( meshPointer, mdd );
        // evaluate
        upwindZEvaluator.evaluate(
                upwindZMeshFunction,     // out
                upwindZFunction );       // in
    timer_upwind.stop();

    timer_R.start();
    traverser_Ki.template processAllEntities< typename LocalUpdaters< MeshType, MeshDependentDataType >::update_R >( meshPointer, mdd_device );
    timer_R.stop();

    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell > traverser_K;
    timer_Q.start();
    traverser_K.template processAllEntities< typename LocalUpdaters< MeshType, MeshDependentDataType >::update_Q >( meshPointer, mdd_device );
    timer_Q.stop();

    timer_preIterate.stop();

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
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
assembleLinearSystem( const RealType & time,
                      const RealType & tau )
{
    timer_assembleLinearSystem.start();

    // Setting this here instead of some setup method ensures that
    // the systemAssembler always has the correct operator etc.
    systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
    systemAssembler.setBoundaryConditions( this->boundaryConditionsPointer );
    systemAssembler.setRightHandSide( this->rightHandSidePointer );

    // initialize system assembler for stationary problem
    systemAssembler.template assemble< typename MeshType::Face >(
            time,
            tau,
            meshPointer,
            matrixPointer,
            rhsVector );

    timer_assembleLinearSystem.stop();

//    (*matrixPointer).print( std::cout );
//    std::cout << rhsVector << std::endl;
//    if( time > tau )
//        abort();

//    static IndexType iter = 0;
//    TNL::String matrixFileName = outputDirectory + "/matrix.tnl";
//    TNL::String dofFileName = outputDirectory + "/dof.vec.tnl";
//    TNL::String rhsFileName = outputDirectory + "/rhs.vec.tnl";
//    if( iter == 10 ) {
//        saveLinearSystem( *matrixPointer, *dofVectorPointer, rhsVector );
//    }
//    iter++;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
saveLinearSystem( const Matrix & matrix,
                  DofViewType dofs,
                  DofViewType rhs ) const
{
    matrix.save( outputDirectory + "/matrix.tnl" );
    dofs.save( outputDirectory + "/dof.vec.tnl" );
    rhs.save( outputDirectory + "/rhs.vec.tnl" );
    std::cerr << "The linear system has been saved to " << outputDirectory << "/{matrix,dof.vec,rhs.vec}.tnl" << std::endl;
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
solveLinearSystem( TNL::Solvers::IterativeSolverMonitor< RealType, IndexType >* solverMonitor )
{
    if( solverMonitor )
        linearSystemSolver->setSolverMonitor( *solverMonitor );

    if( preconditioner )
    {
        timer_linearPreconditioner.start();
        preconditioner->update( matrixPointer );
        timer_linearPreconditioner.stop();
    }

    timer_linearSolver.start();
    DofViewType dofs = mdd->Z_iF.getStorageArray().getView();
    const bool converged = linearSystemSolver->solve( rhsVector, dofs );
    allIterations += linearSystemSolver->getIterations();
    timer_linearSolver.stop();

    if( ! converged ) {
        // save the linear system for debugging
        saveLinearSystem( *matrixPointer, dofs, rhsVector );
        throw std::runtime_error( "MHFEM error: the linear system solver did not converge." );
    }
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
postIterate( const RealType & time,
             const RealType & tau )
{
    timer_postIterate.start();

    timer_explicit.start();
    {
        using coeff = SecondaryCoefficients< MeshDependentDataType >;

        const Mesh* _mesh = &meshPointer.template getData< DeviceType >();
        MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();

        auto kernel = [_mdd, _mesh] __cuda_callable__ ( int i, IndexType K ) mutable
        {
            const auto faceIndexes = getFacesForCell( *_mesh, K );
            _mdd->Z_iK( i, K ) = coeff::Z_iK( *_mdd, faceIndexes, i, K );
        };
        const IndexType cells = meshPointer->template getEntitiesCount< typename Mesh::Cell >();
        TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                            MeshDependentDataType::NumberOfEquations, cells,
                                                            kernel );
    }
    timer_explicit.stop();

    // update coefficients of the conservative velocities
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_velocities.start();
    TNL::Meshes::Traverser< MeshType, typename MeshType::Cell, MeshDependentDataType::NumberOfEquations > traverser_Ki;
    auto* mdd_device = &mdd.template modifyData< DeviceType >();
    traverser_Ki.template processAllEntities< typename LocalUpdaters< MeshType, MeshDependentDataType >::update_v >( meshPointer, mdd_device );
    timer_velocities.stop();

    timer_postIterate.stop();

//    std::cout << "solution (Z_iE): " << std::endl << *dofVectorPointer << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z_iK.getStorageArray() << std::endl;
//    std::cin.get();
}

template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions,
          typename Matrix >
void
Solver< Mesh, MeshDependentData, BoundaryConditions, Matrix >::
writeEpilog( TNL::Logger & logger ) const
{
    logger.writeParameter< long long int >( "Total count of linear solver iterations:", allIterations );
    logger.writeParameter< double >( "Pre-iterate time:", timer_preIterate.getRealTime() );
    logger.writeParameter< double >( "  nonlinear update time:", timer_nonlinear.getRealTime() );
    logger.writeParameter< double >( "  update_b time:", timer_b.getRealTime() );
    logger.writeParameter< double >( "  upwind update time:", timer_upwind.getRealTime() );
    logger.writeParameter< double >( "  update_R time:", timer_R.getRealTime() );
    logger.writeParameter< double >( "  update_Q time:", timer_Q.getRealTime() );
    logger.writeParameter< double >( "Linear system assembler time:", timer_assembleLinearSystem.getRealTime() );
    logger.writeParameter< double >( "Linear preconditioner update time:", timer_linearPreconditioner.getRealTime() );
    logger.writeParameter< double >( "Linear system solver time:", timer_linearPreconditioner.getRealTime() );
    logger.writeParameter< double >( "Post-iterate time:", timer_postIterate.getRealTime() );
    logger.writeParameter< double >( "  Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    logger.writeParameter< double >( "  velocities update time:", timer_velocities.getRealTime() );
}

} // namespace mhfem
