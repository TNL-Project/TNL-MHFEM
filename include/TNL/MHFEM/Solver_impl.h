#pragma once

#include <experimental/filesystem>

#include <TNL/Meshes/Readers/getMeshReader.h>
#include <TNL/Matrices/StaticMatrix.h>
#include <TNL/MPI/Wrappers.h>

#include "../lib_general/mesh_helpers.h"
#include "Solver.h"

namespace mhfem
{

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
TNL::String
Solver< MeshDependentData, BoundaryModel, Matrix >::
getPrologHeader()
{
    return TNL::String( "NumDwarf solver" );
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
setMesh( DistributedHostMeshPointer & meshPointer )
{
    // copy to device, initialize other mesh pointers
    this->distributedHostMeshPointer = meshPointer;
    if( std::is_same< MeshType, HostMeshType >::value ) {
        // in C++17 we would use constexpr if, before that we have to hack with static_cast
        //this->distributedMeshPointer = this->distributedHostMeshPointer;
        this->distributedMeshPointer = *(DistributedMeshPointer*)&this->distributedHostMeshPointer;
    }
    else {
        this->distributedMeshPointer = std::make_shared< DistributedMeshType >();
        *this->distributedMeshPointer = *this->distributedHostMeshPointer;
    }
    this->localMeshPointer = LocalMeshPointer( distributedMeshPointer->getLocalMesh() );

    // allocate mesh dependent data
    mdd->allocate( *localMeshPointer );

    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL ) {
        localFaces = localMeshPointer->template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >();
        localCells = localMeshPointer->template getGhostEntitiesOffset< MeshType::getMeshDimension() >();
    }
    else {
        localFaces = localCells = 0;
    }

    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        // initialize the synchronizer
        faceSynchronizer = std::make_shared< FaceSynchronizerType >();
        faceSynchronizer->initialize( *distributedMeshPointer );

        facesOffset = distributedMeshPointer->template getGlobalIndices< MeshType::getMeshDimension() - 1 >().getElement( 0 );

        const IndexType maxFaceIndex = distributedMeshPointer->template getGlobalIndices< MeshType::getMeshDimension() - 1 >().getElement( localMeshPointer->template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >() - 1 );
        TNL::MPI::Allreduce( &maxFaceIndex, &this->globalFaces, 1, MPI_MAX, distributedMeshPointer->getCommunicator() );
        ++this->globalFaces;

        const IndexType maxCellIndex = distributedMeshPointer->template getGlobalIndices< MeshType::getMeshDimension() >().getElement( localMeshPointer->template getGhostEntitiesOffset< MeshType::getMeshDimension() >() - 1 );
        TNL::MPI::Allreduce( &maxCellIndex, &this->globalCells, 1, MPI_MAX, distributedMeshPointer->getCommunicator() );
        ++this->globalCells;
    }
    else {
        facesOffset = 0;
        globalFaces = localFaces;
        globalCells = localCells;
    }
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
bool
Solver< MeshDependentData, BoundaryModel, Matrix >::
setup( const TNL::Config::ParameterContainer & parameters,
       const TNL::String & prefix )
{
    if( ! distributedMeshPointer ) {
        std::cerr << "The distributedMeshPointer is NULL, the setMesh method must be called first." << std::endl;
        return false;
    }

    // prefix for snapshots
    outputDirectory = parameters.getParameter< TNL::String >( "output-directory" );

    // set up the linear solver
    const TNL::String& linearSolverName = parameters.getParameter< TNL::String >( "linear-solver" );
    linearSystemSolver = TNL::Solvers::getLinearSolver< DistributedMatrixType >( linearSolverName );
    if( ! linearSystemSolver )
        return false;
    if( ! linearSystemSolver->setup( parameters ) )
        return false;

    // set up the preconditioner
    const TNL::String& preconditionerName = parameters.getParameter< TNL::String >( "preconditioner" );
    preconditioner = TNL::Solvers::getPreconditioner< DistributedMatrixType >( preconditionerName );
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

    // reset output/profiling variables
    allIterations = 0;
    timer_preIterate.reset();
    timer_assembleLinearSystem.reset();
    timer_linearPreconditioner.reset();
    timer_linearSolver.reset();
    timer_postIterate.reset();
    timer_b.reset();
    timer_R.reset();
    timer_Q.reset();
    timer_nonlinear.reset();
    timer_upwind.reset();
    timer_model_preIterate.reset();
    timer_explicit.reset();
    timer_velocities.reset();
    timer_model_postIterate.reset();
    timer_mpi_upwind.reset();

    TNL::MPI::getTimerAllreduce().reset();
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        faceSynchronizer->async_ops_count = 0;
        faceSynchronizer->async_wait_before_start_timer.reset();
        faceSynchronizer->async_start_timer.reset();
        faceSynchronizer->async_wait_timer.reset();
    }

    return true;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
bool
Solver< MeshDependentData, BoundaryModel, Matrix >::
setInitialCondition( const TNL::Config::ParameterContainer & parameters )
{
    if( parameters.checkParameter( "boundary-conditions-file" ) ) {
        std::string boundaryConditionsFile = parameters.getParameter< std::string >( "boundary-conditions-file" );
        if( TNL::MPI::GetSize() > 1 ) {
            namespace fs = std::experimental::filesystem;
            fs::path path( boundaryConditionsFile );
            std::string ext = path.extension();
            ext = "." + std::to_string( TNL::MPI::GetRank() ) + ext;
            path.replace_extension(ext);
            boundaryConditionsFile = path.string();
        }
        BoundaryConditionsStorage< RealType > storage;
        storage.load( boundaryConditionsFile );
        if( storage.dofSize != getDofs() ) {
            std::cerr << "Wrong dofSize in BoundaryConditionsStorage loaded from file " << boundaryConditionsFile << ". "
                      << "Expected " << getDofs() << " elements, got " << storage.dofSize << "." << std::endl;
            return false;
        }
        boundaryConditionsPointer->init( storage );
    }
    else {
        // default boundary conditions (zero Dirichlet), without this calling setupLinearSystem would fail
        // (e.g. the coupled solver calls setInitialCondition first without boundary-conditions-file, but
        // then sets different boundary conditions properly)
        BoundaryConditionsStorage< RealType > storage;
        storage.dofSize = getDofs();
        storage.tags.setSize( getDofs() );
        storage.values.setSize( getDofs() );
        storage.dirichletValues.setSize( getDofs() );
        storage.tags.setValue( BoundaryConditionsType::FixedValue );
        storage.values.setValue( 0 );
        storage.dirichletValues.setValue( 0 );
        boundaryConditionsPointer->init( storage );
    }

    if( parameters.checkParameter( "initial-condition" ) ) {
        const TNL::String & initialConditionFile = parameters.getParameter< TNL::String >( "initial-condition" );
        auto reader = TNL::Meshes::Readers::getMeshReader( initialConditionFile, "auto" );
        reader->detectMesh();
        for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ ) {
            const std::string name = "InitialCondition[Z" + std::to_string(i) + "]";
            const auto variant_vec = reader->readCellData( name );
            using mpark::visit;
            visit( [this, i](auto&& vector) { this->mdd->setInitialCondition( i, vector ); }, variant_vec );
        }
    }
    else {
        mdd->Z_iK.setValue( 0 );
    }

    if( ! mdd->init( parameters ) )
        return false;

    #ifdef HAVE_CUDA
    // make sure that TNL smart pointers are synchronized
    TNL::Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
    #endif

    // initialize mdd->Z_iF as an average of mdd->Z on neighbouring cells
    // (this is not strictly necessary, we just provide an initial guess for
    // the iterative linear solver)
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    auto faceAverageKernel = [_mesh, _mdd] __cuda_callable__ ( int i, IndexType E )
    {
        IndexType cellIndexes[ 2 ] = {0, 0};
        const int numCells = getCellsForFace( *_mesh, E, cellIndexes );

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

    mdd->v_iKe.setValue( 0 );

    return true;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
setupLinearSystem()
{
    if( distributedMeshPointer->getCommunicator() == MPI_COMM_NULL )
        return;

    using CompressedRowLengths =
        TNL::Containers::NDArray< IndexType,
                                  TNL::Containers::SizesHolder< IndexType, MeshDependentDataType::NumberOfEquations, 0 >,  // i, F
                                  std::index_sequence< 1, 0 >,  // F, i  (must match the permutation for mdd.Z_iF for all devices
                                  DeviceType >;

    CompressedRowLengths rowLengths;
    rowLengths.setSizes( 0, localFaces );
    rowLengths.setValue( -1 );

    auto rowLengths_view = rowLengths.getView();
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _op = &differentialOperatorPointer.template getData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    auto kernel = [rowLengths_view, _mesh, _op, _bc] __cuda_callable__ ( int i, IndexType E ) mutable
    {
        IndexType length;
        if( isBoundaryFace( *_mesh, E ) )
            length = _bc->getLinearSystemRowLength( *_mesh, E, i );
        else
            length = _op->getLinearSystemRowLength( *_mesh, E, i );
        rowLengths_view( i, E ) = length;
    };
    rowLengths_view.forAll( kernel );

    // sanity check (doesn't happen if the traverser works, but this is pretty
    // hard to debug and the check does not cost us much in initialization)
    TNL::Containers::VectorView< IndexType, DeviceType, IndexType > rowLengths_vector( rowLengths.getStorageArray().getView() );
    if( TNL::min( rowLengths_vector ) <= 0 ) {
        std::stringstream ss;
        ss << "MHFEM error: attempted to set invalid rowsLengths vector:\n" << rowLengths_vector << std::endl;
        throw std::runtime_error( ss.str() );
    }

    // initialize the matrix
    const IndexType offset = this->getDofsOffset();
    const IndexType localDofs = this->getLocalDofs();
    const IndexType dofs = this->getDofs();
    const IndexType globalDofs = this->getGlobalDofs();
    distributedMatrixPointer = std::make_shared< DistributedMatrixType >();
    distributedMatrixPointer->setDistribution( {offset, offset + localDofs}, globalDofs, dofs, distributedMeshPointer->getCommunicator() );
    TNL::Containers::DistributedVectorView< IndexType, DeviceType, IndexType > dist_rowLengths( {offset, offset + localDofs}, 0, globalDofs, distributedMatrixPointer->getCommunicator(), rowLengths_vector );
    distributedMatrixPointer->setRowCapacities( dist_rowLengths );

    // initialize the right hand side vector
    rhsVector.setSize( dofs );

    // set the matrix for the linear solver
    linearSystemSolver->setMatrix( distributedMatrixPointer );

    // bind device pointer to the local matrix
    this->localMatrixPointer = LocalMatrixPointer( distributedMatrixPointer->getLocalMatrix() );
}


template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::DistributedHostMeshPointer
Solver< MeshDependentData, BoundaryModel, Matrix >::
getHostMesh()
{
    return distributedHostMeshPointer;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::DistributedMeshPointer
Solver< MeshDependentData, BoundaryModel, Matrix >::
getMesh()
{
    return distributedMeshPointer;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::MeshDependentDataPointer&
Solver< MeshDependentData, BoundaryModel, Matrix >::
getMeshDependentData()
{
    return mdd;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::BoundaryConditionsPointer&
Solver< MeshDependentData, BoundaryModel, Matrix >::
getBoundaryConditions()
{
    return boundaryConditionsPointer;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
std::shared_ptr< typename Solver< MeshDependentData, BoundaryModel, Matrix >::FaceSynchronizerType >&
Solver< MeshDependentData, BoundaryModel, Matrix >::
getFaceSynchronizer()
{
    return faceSynchronizer;
}


template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::IndexType
Solver< MeshDependentData, BoundaryModel, Matrix >::
getDofsOffset() const
{
    return MeshDependentDataType::NumberOfEquations * facesOffset;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::IndexType
Solver< MeshDependentData, BoundaryModel, Matrix >::
getLocalDofs() const
{
    // exclude ghost entities
    return MeshDependentDataType::NumberOfEquations * localFaces;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::IndexType
Solver< MeshDependentData, BoundaryModel, Matrix >::
getDofs() const
{
    // include ghost entities
    return MeshDependentDataType::NumberOfEquations * localMeshPointer->template getEntitiesCount< typename MeshType::Face >();
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
typename Solver< MeshDependentData, BoundaryModel, Matrix >::IndexType
Solver< MeshDependentData, BoundaryModel, Matrix >::
getGlobalDofs() const
{
    return MeshDependentDataType::NumberOfEquations * globalFaces;
}


template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
makeSnapshot( const RealType time,
              const IndexType step )
{
    // TODO: write only into the log file
//    std::cout << std::endl << "Writing output at time " << time << " step " << step << std::endl;

    mdd->makeSnapshot( time, step, *distributedHostMeshPointer, outputDirectory + "/" );
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
preIterate( const RealType time,
            const RealType tau )
{
    timer_preIterate.start();

    // not necessary for correctness, but for correct timings
    #ifdef HAVE_CUDA
    TNL::Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
    #endif

    using coeff = SecondaryCoefficients< MeshDependentDataType >;
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    const IndexType cells = localMeshPointer->template getEntitiesCount< typename MeshType::Cell >();

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
                                                            cells, (IndexType) MeshDependentDataType::NumberOfEquations, (IndexType) MeshDependentDataType::NumberOfEquations,
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
                                                            cells, (IndexType) MeshDependentDataType::NumberOfEquations,
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

        auto kernel_m_iE = [_mdd, _mesh, _bc, time, tau] __cuda_callable__ ( IndexType E, int i ) mutable
        {
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( *_mesh, E, cellIndexes );

            // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
            const IndexType & K1 = cellIndexes[ 0 ];

            // find local index of face E
            const auto faceIndexesK1 = getFacesForCell( *_mesh, K1 );
            const int e1 = getLocalIndex( faceIndexesK1, E );

            RealType m_iE_upw;

            if( numCells == 1 ) {
                // We need to check inflow of ALL phases!
                // FIXME: this assumes two-phase model, general system might be coupled differently or even decoupled
                // TODO: check the BoundaryConditionsType value
                bool inflow = false;
                for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                    // Taking the boundary value increases the error, for example in the mcwh3d problem
                    // on cubes, so we need to use _mdd->v_iKe instead of _bc->getNeumannValue
                    if( _mdd->v_iKe( j, K1, e1 ) < 0 ) {
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
                const auto faceIndexesK2 = getFacesForCell( *_mesh, K2 );
                const int e2 = getLocalIndex( faceIndexesK2, E );
                // Theoretically, v_iKE is conservative so one might expect that `vel = _mdd->v_iKe( i, K1, e )`
                // is enough, but there might be numerical errors. Perhaps more importantly, the main equation
                // might not be based on balancing v_iKE, but some other quantity. We also use a dummy equation
                // if Q_K is singular, so this has significant effect on the error.
                const RealType vel = _mdd->v_iKe( i, K1, e1 ) - _mdd->v_iKe( i, K2, e2 );

                if( vel == 0 )
                    // symmetrized (upwinding should not depend on the order of elements K1 and K2)
                    m_iE_upw = 0.5 * ( _mdd->m_iK( i, K1 ) + _mdd->m_iK( i, K2 ) );
                else if( vel > 0 )
                    m_iE_upw = _mdd->m_iK( i, K1 );
                else
                    m_iE_upw = _mdd->m_iK( i, K2 );
            }

            // write into the global memory after all branches have converged
            _mdd->m_iE_upw( i, E ) = m_iE_upw;
        };
        // mdd->m_iE_upw.forAll does not skip ghosts, so we use ParallelFor2D manually for the specific permutation of indices
        TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                            localFaces, (IndexType) MeshDependentDataType::NumberOfEquations,
                                                            kernel_m_iE );

        auto kernel_Z_ijE = [_mdd, _mesh] __cuda_callable__ ( IndexType E, int j, int i ) mutable
        {
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( *_mesh, E, cellIndexes );

            // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
            const IndexType & K1 = cellIndexes[ 0 ];

            // find local index of face E
            const auto faceIndexesK1 = getFacesForCell( *_mesh, K1 );
            const int e1 = getLocalIndex( faceIndexesK1, E );

            const RealType a_plus_u = _mdd->a_ijKe( i, j, K1, e1 ) + _mdd->u_ijKe( i, j, K1, e1 );

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
        // mdd->Z_ijE_upw.forAll does not skip ghosts, so we use ParallelFor3D manually for the specific permutation of indices
        TNL::Algorithms::ParallelFor3D< DeviceType >::exec( (IndexType) 0, (IndexType) 0, (IndexType) 0,
                                                            localFaces, (IndexType) MeshDependentDataType::NumberOfEquations, (IndexType) MeshDependentDataType::NumberOfEquations,
                                                            kernel_Z_ijE );
    }
    timer_upwind.stop();

    // synchronize the upwinded quantities
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        timer_mpi_upwind.start();

        // NOTE: this is specific to how the ndarrays are ordered
        auto m_upw_view = mdd->m_iE_upw.getStorageArray().getView();
        faceSynchronizer->synchronizeArray( m_upw_view, MeshDependentDataType::NumberOfEquations );

        auto Z_upw_view = mdd->Z_ijE_upw.getStorageArray().getView();
        faceSynchronizer->synchronizeArray( Z_upw_view, MeshDependentDataType::NumberOfEquations * MeshDependentDataType::NumberOfEquations );

        timer_mpi_upwind.stop();
    }

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
                                                            cells, (IndexType) MeshDependentDataType::NumberOfEquations,
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

            using LocalMatrixType = TNL::Matrices::StaticMatrix< RealType, MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations >;
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

    timer_model_preIterate.start();
    mdd->preIterate( time, tau );
    timer_model_preIterate.stop();

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

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
assembleLinearSystem( const RealType time,
                      const RealType tau )
{
    if( distributedMeshPointer->getCommunicator() == MPI_COMM_NULL )
        return;

    timer_assembleLinearSystem.start();

    const auto* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _mdd = &mdd.template modifyData< DeviceType >();
    const auto* _op = &differentialOperatorPointer.template getData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    const auto* _rhs = &rightHandSidePointer.template getData< DeviceType >();
    auto* _matrix = &localMatrixPointer.template modifyData< DeviceType >();
    auto _b = rhsVector.getView();
    auto kernel = [_mesh, _mdd, _op, _bc, _rhs, _matrix, _b, time, tau] __cuda_callable__ ( IndexType E, int i ) mutable
    {
        TNL_ASSERT_FALSE( _mesh->template isGhostEntity< MeshType::getMeshDimension() - 1 >( E ),
                          "A ghost face encountered while assembling the linear system." );
        const IndexType rowIndex = _mdd->getRowIndex( i, E );
        TNL_ASSERT_LT( rowIndex, _matrix->getRows(), "bug in getRowIndex" );
        if( isBoundaryFace( *_mesh, E ) )
            _bc->setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, *_matrix, _b );
        else {
            _op->setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, *_matrix, _b );
            _b[ rowIndex ] = (*_rhs)( *_mesh, *_mdd, E, i );
        }
    };
    TNL_ASSERT_EQ( localMatrixPointer->getRows(), MeshDependentDataType::NumberOfEquations * localFaces, "BUG: wrong matrix size" );
    // mdd->Z_iF.forAll does not skip ghosts, so we use ParallelFor2D manually for the specific permutation of indices
    TNL::Algorithms::ParallelFor2D< DeviceType >::exec( (IndexType) 0, (IndexType) 0,
                                                        localFaces, (IndexType) MeshDependentDataType::NumberOfEquations,
                                                        kernel );

    timer_assembleLinearSystem.stop();
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
saveLinearSystem( const Matrix & matrix,
                  DofViewType dofs,
                  DofViewType rhs ) const
{
    matrix.save( outputDirectory + "/matrix.tnl" );
    dofs.save( outputDirectory + "/dof.vec.tnl" );
    rhs.save( outputDirectory + "/rhs.vec.tnl" );
    std::cerr << "The linear system has been saved to " << outputDirectory << "/{matrix,dof.vec,rhs.vec}.tnl" << std::endl;
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
solveLinearSystem( TNL::Solvers::IterativeSolverMonitor< RealType, IndexType >* solverMonitor )
{
    if( distributedMeshPointer->getCommunicator() == MPI_COMM_NULL )
        return;

    if( solverMonitor )
        linearSystemSolver->setSolverMonitor( *solverMonitor );

    if( preconditioner )
    {
        timer_linearPreconditioner.start();
        preconditioner->update( distributedMatrixPointer );
        timer_linearPreconditioner.stop();
    }

    timer_linearSolver.start();
    const IndexType offset = this->getDofsOffset();
    const IndexType localDofs = this->getLocalDofs();
    const IndexType dofs = this->getDofs();
    const IndexType globalDofs = this->getGlobalDofs();

    DofViewType dofs_view = mdd->Z_iF.getStorageArray().getView();
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_dofs( {offset, offset + localDofs}, dofs - localDofs, globalDofs, distributedMatrixPointer->getCommunicator(), dofs_view );
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_rhs( {offset, offset + localDofs}, dofs - localDofs, globalDofs, distributedMatrixPointer->getCommunicator(), rhsVector.getView() );
    dist_dofs.setSynchronizer( faceSynchronizer, MeshDependentDataType::NumberOfEquations );
    dist_rhs.setSynchronizer( faceSynchronizer, MeshDependentDataType::NumberOfEquations );
    dist_rhs.startSynchronization();

    // NOTE: the dist_dofs vector will be synchronized by the solver, so we do not have to use faceSynchronizer again
    const bool converged = linearSystemSolver->solve( dist_rhs, dist_dofs );
    allIterations += linearSystemSolver->getIterations();
    timer_linearSolver.stop();

    if( ! converged ) {
        // save the linear system for debugging
        // TODO: save the distributed system
        saveLinearSystem( distributedMatrixPointer->getLocalMatrix(), dofs_view, rhsVector );
        throw std::runtime_error( "MHFEM error: the linear system solver did not converge (" + std::to_string(linearSystemSolver->getIterations()) + " iterations)." );
    }
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
postIterate( const RealType time,
             const RealType tau )
{
    timer_postIterate.start();

    using coeff = SecondaryCoefficients< MeshDependentDataType >;
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    MeshDependentDataType* _mdd = &mdd.template modifyData< DeviceType >();
    const IndexType cells = localMeshPointer->template getEntitiesCount< typename MeshType::Cell >();

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
                                                            cells, (IndexType) MeshDependentDataType::NumberOfEquations,
                                                            kernel );
    }
    timer_velocities.stop();

    timer_model_postIterate.start();
    mdd->postIterate( time, tau );
    timer_model_postIterate.stop();

    timer_postIterate.stop();

//    std::cout << "solution (Z_iE): " << std::endl << mdd->Z_iF.getStorageArray() << std::endl;
//    std::cout << "solution (Z_iK): " << std::endl << mdd->Z_iK.getStorageArray() << std::endl;
//    std::cin.get();
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
writeEpilog( TNL::Logger & logger ) const
{
    logger.writeParameter< long long int >( "Total count of linear solver iterations:", allIterations );
    logger.writeParameter< double >( "Pre-iterate time:", timer_preIterate.getRealTime() );
    logger.writeParameter< double >( "  nonlinear update time:", timer_nonlinear.getRealTime() );
    logger.writeParameter< double >( "  update_b time:", timer_b.getRealTime() );
    logger.writeParameter< double >( "  upwind update time:", timer_upwind.getRealTime() );
    logger.writeParameter< double >( "  upwind MPI synchronization time:", timer_mpi_upwind.getRealTime() );
    logger.writeParameter< double >( "  update_R time:", timer_R.getRealTime() );
    logger.writeParameter< double >( "  update_Q time:", timer_Q.getRealTime() );
    logger.writeParameter< double >( "  model pre-iterate time:", timer_model_preIterate.getRealTime() );
    logger.writeParameter< double >( "Linear system assembler time:", timer_assembleLinearSystem.getRealTime() );
    logger.writeParameter< double >( "Linear preconditioner update time:", timer_linearPreconditioner.getRealTime() );
    logger.writeParameter< double >( "Linear system solver time:", timer_linearSolver.getRealTime() );
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        const double total_mpi_time = faceSynchronizer->async_wait_before_start_timer.getRealTime()
                                    + faceSynchronizer->async_start_timer.getRealTime()
                                    + faceSynchronizer->async_wait_timer.getRealTime();
        logger.writeParameter< std::size_t >( "  MPI synchronizations count:", faceSynchronizer->async_ops_count );
        logger.writeParameter< double >( "  MPI synchronization time:", total_mpi_time );
        logger.writeParameter< double >( "    async wait before start time:", faceSynchronizer->async_wait_before_start_timer.getRealTime() );
        logger.writeParameter< double >( "    async start time:", faceSynchronizer->async_start_timer.getRealTime() );
        logger.writeParameter< double >( "    async wait time:", faceSynchronizer->async_wait_timer.getRealTime() );
    }
    logger.writeParameter< double >( "Post-iterate time:", timer_postIterate.getRealTime() );
    logger.writeParameter< double >( "  Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    logger.writeParameter< double >( "  velocities update time:", timer_velocities.getRealTime() );
    logger.writeParameter< double >( "  model post-iterate time:", timer_model_postIterate.getRealTime() );
    if( TNL::MPI::GetSize() > 1 ) {
        logger.writeParameter< std::string >( "MPI operations (included in the previous phases):", "" );
        logger.writeParameter< double >( "  MPI_Allreduce time:", TNL::MPI::getTimerAllreduce().getRealTime() );
    }
}

} // namespace mhfem
