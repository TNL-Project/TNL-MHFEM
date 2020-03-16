#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>

#include "../lib_general/mesh_helpers.h"

#include "Solver.h"

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

    // Our kernels for preIterate, postIterate and DifferentialOperator have many local memory spills,
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

    // initialize mdd->Z_iF as an average of mdd->Z on neighbouring cells
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
    mdd->Z_iF.forAll( faceAverageKernel );

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

    // not necessary for correctness, but for correct timings
    #ifdef HAVE_CUDA
    TNL::Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
    #endif

    using coeff = SecondaryCoefficients< MeshDependentDataType >;
    const Mesh* _mesh = &meshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    const IndexType cells = meshPointer->template getEntitiesCount< typename Mesh::Cell >();

    // update non-linear terms
    timer_nonlinear.start();
    {
        auto kernel = [_mdd, _mesh, time] __cuda_callable__ ( IndexType K ) mutable
        {
            _mdd->updateNonLinearTerms( *_mesh, K, time );
        };
        TNL::Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, cells, kernel );
    }
    timer_nonlinear.stop();

    // update coefficients b_ijKEF
    timer_b.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( IndexType K, int i, int j ) mutable
        {
            using MassMatrix = typename MeshDependentDataType::MassMatrix;
            MassMatrix::update( *_mesh, *_mdd, K, i, j );
        };
        TNL::Algorithms::ParallelFor3D< DeviceType >::exec( (IndexType) 0, (IndexType) 0, (IndexType) 0,
                                                            cells, MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations,
                                                            kernel );
    }
    timer_b.stop();

    // update vector coefficients (u, w, a), whose projection into the RTN_0(K) space
    // generally depends on the b_ijKEF coefficients
    timer_nonlinear.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( IndexType K, int i ) mutable
        {
            _mdd->updateVectorCoefficients( *_mesh, K, i );
        };
        TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                            cells, MeshDependentDataType::NumberOfEquations,
                                                            kernel );
    }
    timer_nonlinear.stop();

    // update upwinded mobility values
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_upwind.start();
    {
        const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
        const IndexType faces = meshPointer->template getEntitiesCount< typename Mesh::Face >();

        auto kernel_m_iE = [_mdd, _mesh, _bc, time, tau] __cuda_callable__ ( int i, IndexType E ) mutable
        {
            const auto& entity = _mesh->template getEntity< typename MeshType::Face >( E );

            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( *_mesh, entity, cellIndexes );

            // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
            const IndexType & K1 = cellIndexes[ 0 ];

            // find local index of face E
            const auto faceIndexes = getFacesForCell( *_mesh, K1 );
            const int e = getLocalIndex( faceIndexes, E );

            RealType m_iE_upw;

            if( numCells == 1 ) {
                // We need to check inflow of ALL phases!
                // FIXME: this assumes two-phase model, general system might be coupled differently or even decoupled
                // TODO: check the BoundaryConditionsType value
                bool inflow = false;
                for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                    // Taking the boundary value increases the error, for example in the mcwh3d problem
                    // on cubes, so we need to use _mdd->v_iKe instead of _bc->getNeumannValue
                    if( _mdd->v_iKe( j, K1, e ) < 0 ) {
                        inflow = true;
                        break;
                    }

                if( inflow )
                    // The velocity might be negative even on faces with 0 Neumann condition (probably
                    // due to rounding errors), so the model must check if the value is available and
                    // otherwise return m_iK( i, K1 ).
                    m_iE_upw = _mdd->getBoundaryMobility( *_mesh, *_bc, i, E, K1, time, tau );
                else
                    m_iE_upw = _mdd->m_iK( i, K1 );
            }
            else {
                const IndexType & K2 = cellIndexes[ 1 ];
                // Theoretically, v_iKE is conservative so one might expect that `vel = _mdd->v_iKe( i, K1, e )`
                // is enough, but there might be numerical errors. Perhaps more importantly, the main equation
                // might not be based on balancing v_iKE, but some other quantity. We also use a dummy equation
                // if Q_K is singular, so this has significant effect on the error.
                const RealType vel = _mdd->v_iKe( i, K1, e ) - _mdd->v_iKe( i, K2, e );

                if( vel >= 0.0 )
                    m_iE_upw = _mdd->m_iK( i, K1 );
                else
                    m_iE_upw = _mdd->m_iK( i, K2 );
            }

            // write into the global memory after all branches have converged
            _mdd->m_iE_upw( i, E ) = m_iE_upw;
        };
        mdd->m_iE_upw.forAll( kernel_m_iE );

        auto kernel_Z_ijE = [_mdd, _mesh] __cuda_callable__ ( int i, int j, IndexType E ) mutable
        {
            const auto& entity = _mesh->template getEntity< typename MeshType::Face >( E );

            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( *_mesh, entity, cellIndexes );

            // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
            const IndexType & K1 = cellIndexes[ 0 ];

            // find local index of face E
            const auto faceIndexes = getFacesForCell( *_mesh, K1 );
            const int e = getLocalIndex( faceIndexes, E );

            const RealType a_plus_u = _mdd->a_ijKe( i, j, K1, e ) + _mdd->u_ijKe( i, j, K1, e );

            RealType Z_ijE_upw;

            if( a_plus_u > 0.0 )
                Z_ijE_upw = _mdd->Z_iK( j, K1 );
            else if( a_plus_u == 0.0 )
                Z_ijE_upw = 0;
            else if( numCells == 2 ) {
                const IndexType & K2 = cellIndexes[ 1 ];
                Z_ijE_upw = _mdd->Z_iK( j, K2 );
            }
            else {
                // TODO: this matches the Dirichlet condition, but what happens on Neumann boundary?
                // TODO: at time=0 the value on Neumann boundary is indeterminate
                Z_ijE_upw = _mdd->Z_iF( j, E );
            }

            // write into the global memory after all branches have converged
            _mdd->Z_ijE_upw( i, j, E ) = Z_ijE_upw;
        };
        mdd->Z_ijE_upw.forAll( kernel_Z_ijE );
    }
    timer_upwind.stop();

    timer_R.start();
    {
        auto kernel = [_mdd, _mesh, tau] __cuda_callable__ ( IndexType K, int i ) mutable
        {
            // get face indexes
            const auto faceIndexes = getFacesForCell( *_mesh, K );

            // update coefficients R_ijKE
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ )
                    _mdd->R_ijKe( i, j, K, e ) = coeff::R_ijKe( *_mdd, faceIndexes, i, j, K, e, tau );

            // update coefficient R_iK
            const auto& entity = _mesh->template getEntity< typename MeshType::Cell >( K );
            _mdd->R_iK( i, K ) = coeff::R_iK( *_mdd, *_mesh, entity, faceIndexes, i, K, tau );
        };
        TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                            cells, MeshDependentDataType::NumberOfEquations,
                                                            kernel );
    }
    timer_R.stop();

    timer_Q.start();
    {
        auto kernel = [_mdd, _mesh, tau] __cuda_callable__ ( IndexType K ) mutable
        {
            // get face indexes
            const auto faceIndexes = getFacesForCell( *_mesh, K );
            const auto& entity = _mesh->template getEntity< typename MeshType::Cell >( K );

            using LocalMatrixType = TNL::Containers::StaticMatrix< RealType, MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations >;
#ifndef __CUDA_ARCH__
            LocalMatrixType Q;
//            RealType rhs[ MeshDependentDataType::NumberOfEquations ];
#else
            // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
            // (we'll need to pass custom launch configuration to the ParallelFor)
            // Now we just assume that the ParallelFor kernel uses 256 threads per block.
            __shared__ LocalMatrixType __Qs[ 256 ];
            LocalMatrixType& Q = __Qs[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];

            // TODO: this limits the kernel to 1 block on Fermi - maybe cudaFuncCachePreferShared specifically for this kernel would help
//            __shared__ RealType __rhss[ MeshDependentDataType::NumberOfEquations * 256 ];
//            RealType* rhs = &__rhss[ MeshDependentDataType::NumberOfEquations * (
//                                        ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x
//                                    ) ];
#endif

            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                    const RealType value = coeff::Q_ijK( *_mdd, *_mesh, entity, faceIndexes, i, j, K, tau );
                    Q( i, j ) = value;

                    // update singularity state
                    if( value != 0.0 )
                        singular = false;
                }

                // check for singularity
                if( singular ) {
                    Q( i, i ) = 1.0;
                    _mdd->R_iK( i, K ) += _mdd->Z_iK( i, K );
                }
            }

            LU_factorize( Q );

            RealType rhs[ MeshDependentDataType::NumberOfEquations ];

            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                rhs[ i ] = _mdd->R_iK( i, K );
            LU_solve_inplace( Q, rhs );
            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                _mdd->R_iK( i, K ) = rhs[ i ];

            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ ) {
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        rhs[ i ] = _mdd->R_ijKe( i, j, K, e );
                    LU_solve_inplace( Q, rhs );
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        _mdd->R_ijKe( i, j, K, e ) = rhs[ i ];
                }
        };
        TNL::Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, cells, kernel );
    }
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

    using coeff = SecondaryCoefficients< MeshDependentDataType >;
    const Mesh* _mesh = &meshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    const IndexType cells = meshPointer->template getEntitiesCount< typename Mesh::Cell >();

    timer_explicit.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( int i, IndexType K ) mutable
        {
            const auto faceIndexes = getFacesForCell( *_mesh, K );
            _mdd->Z_iK( i, K ) = coeff::Z_iK( *_mdd, faceIndexes, i, K );
        };
        mdd->Z_iK.forAll( kernel );
    }
    timer_explicit.stop();

    // update coefficients of the conservative velocities
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_velocities.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( IndexType K, int i ) mutable
        {
            const auto faceIndexes = getFacesForCell( *_mesh, K );
            for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ )
                _mdd->v_iKe( i, K, e ) = coeff::v_iKE( *_mdd, faceIndexes, i, K, faceIndexes[ e ], e );
        };
        TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                            cells, MeshDependentDataType::NumberOfEquations,
                                                            kernel );
    }
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
    logger.writeParameter< double >( "Linear system solver time:", timer_linearSolver.getRealTime() );
    logger.writeParameter< double >( "Post-iterate time:", timer_postIterate.getRealTime() );
    logger.writeParameter< double >( "  Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    logger.writeParameter< double >( "  velocities update time:", timer_velocities.getRealTime() );
}

} // namespace mhfem
