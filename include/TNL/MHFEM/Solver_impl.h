#pragma once

#include <filesystem>
#include <variant>

#include <TNL/Meshes/Readers/getMeshReader.h>
#include <TNL/Matrices/StaticMatrix.h>
#include <TNL/MPI/Wrappers.h>

#include "mesh_helpers.h"
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

    // deallocate unnecessary superentity matrix on the device mesh
    // (it is needed only for distributeSubentities and the EntityTags layer initialization)
    this->localMeshPointer->template getSuperentitiesMatrix< 0, MeshType::getMeshDimension() - 1 >().reset();

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

    const std::string solver_name = parameters.getParameter< std::string >( "linear-solver" );
    const std::string preconditioner_name = parameters.getParameter< std::string >( "preconditioner" );

#if defined( HAVE_GINKGO )
    // check the dynamic configuration
    if( solver_name != "bicgstab" )
        throw std::logic_error( "invalid solver name for Ginkgo: " + solver_name );
    if( preconditioner_name == "AMGX" )
        gko_preconditioner_type = AMGX;
    else if( preconditioner_name == "ILU_ISAI" )
        gko_preconditioner_type = ILU_ISAI;
    else if( preconditioner_name == "PARILU_ISAI" )
        gko_preconditioner_type = PARILU_ISAI;
    else if( preconditioner_name == "PARILUT_ISAI" )
        gko_preconditioner_type = PARILUT_ISAI;
    else
        throw std::logic_error( "invalid preconditioner name for Ginkgo: " + preconditioner_name );

    if( gko_exec == nullptr ) {
        #ifdef HAVE_CUDA
        // NOTE: false here disables device reset in the executor's destructor
        gko_exec = gko::CudaExecutor::create( 0, gko::OmpExecutor::create(), false );
        #else
        gko_exec = gko::OmpExecutor::create();
        #endif
    }

    // Create the convergence logger
    gko_convergence_logger = gko::share( TNL::Solvers::GinkgoConvergenceLoggerMonitor< RealType, IndexType >::create() );

    // Create stopping criteria
    gko_stop_iter = gko::share(
            gko::stop::Iteration::build()
                .with_max_iters( 1000 )
                .on( gko_exec ) );
    const double convergence_residue = parameters.getParameter< double >( "convergence-residue" );
    gko_stop_tol = gko::share(
            gko::stop::ResidualNorm< RealType >::build()
                .with_baseline( gko::stop::mode::rhs_norm )
                .with_reduction_factor( convergence_residue )
                .on( gko_exec ) );
    // missing chaining member:  https://github.com/ginkgo-project/ginkgo/discussions/1099#discussioncomment-3439954
    gko_stop_iter->add_logger( gko_convergence_logger );
    gko_stop_tol->add_logger( gko_convergence_logger );
#elif defined( HAVE_HYPRE )
    // check the dynamic configuration
    if( solver_name != "bicgstab" )
        throw std::logic_error( "invalid solver name for Hypre: " + solver_name );
    if( preconditioner_name != "BoomerAMG" )
        throw std::logic_error( "invalid preconditioner name for Hypre: " + preconditioner_name );

    // NOTE: Hypre uses right-preconditioning in all solvers, which means that
    // *original* (i.e. unpreconditioned) residuals are used in the stopping
    // criteria. Hence, the stopping threshold is fundamentally different from
    // left-preconditioned solvers in TNL. But so far we used only the Jacobi
    // preconditioner in TNL and the MHFEM matrix assembler was updated to apply
    // the diagonal scaling manually, so the threshold value should be on the
    // same scale now.
    const double convergence_residue = parameters.getParameter< double >( "convergence-residue" );

    // create and configure the solver
    hypre_solver = std::make_unique< TNL::Solvers::Linear::HypreBiCGSTAB >( distributedMeshPointer->getCommunicator() );
    // Set some parameters (See Reference Manual for more parameters)
    HYPRE_BiCGSTABSetMaxIter( *hypre_solver, 1000 );  // max iterations
    HYPRE_BiCGSTABSetTol( *hypre_solver, convergence_residue );
//    HYPRE_BiCGSTABSetPrintLevel( *hypre_solver, 2 );  // print solve info

    // create the preconditioner and set it to the solver
    hypre_precond = std::make_unique< TNL::Solvers::Linear::HypreBoomerAMG >();
    hypre_solver->setPreconditioner( *hypre_precond );

    // Set some parameters (See Reference Manual for more parameters)
    HYPRE_BoomerAMGSetPrintLevel( *hypre_precond, 1 );    // Print setup info + parameters
    hypre_precond->setSystemsOptions( MeshDependentData::NumberOfEquations, false );
    // setSystemsOptions also disables aggressive coarsening and sets the strong threshold to 0.5
    HYPRE_BoomerAMGSetAggNumLevels( *hypre_precond, 1 );
    // for MHFEM, 0.25 seems to be a good choice regardless of the dimension (2D and 3D)
    HYPRE_BoomerAMGSetStrongThreshold( *hypre_precond, 0.25 );
//    if( MeshType::getMeshDimension() == 2 )
//        HYPRE_BoomerAMGSetStrongThreshold( *hypre_precond, 0.25 );
//    if( MeshType::getMeshDimension() == 3 )
//        HYPRE_BoomerAMGSetStrongThreshold( *hypre_precond, 0.5 );
#ifndef HYPRE_USING_GPU
    // truncation factor for the interpolation (higher value means more truncation, default is 0)
    // (on GPUs this may destroy convergence - value tuned for HMIS coarsening on CPU)
    HYPRE_BoomerAMGSetTruncFactor( *hypre_precond, 0.3 );
#endif
    // AMG coarsening algorithm: 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
//    HYPRE_BoomerAMGSetCoarsenType( *hypre_precond, 8 );
    // AMG relaxation algorithm: 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi, 16 = Chebyshev
//    HYPRE_BoomerAMGSetRelaxType( *hypre_precond, 18 );
#else
    // set up the linear solver
    linearSystemSolver = TNL::Solvers::getLinearSolver< DistributedMatrixType >( solver_name );
    if( ! linearSystemSolver )
        return false;
    if( ! linearSystemSolver->setup( parameters ) )
        return false;

    // set up the preconditioner
    preconditioner = TNL::Solvers::getPreconditioner< DistributedMatrixType >( preconditioner_name );
    if( preconditioner ) {
        linearSystemSolver->setPreconditioner( preconditioner );
        if( ! preconditioner->setup( parameters ) )
            return false;
    }
#endif

    // Our kernels for preIterate, postIterate and DifferentialOperator have many local memory spills,
    // so this helps a lot. It does not affect TNL's reduction and multireduction algorithms,
    // which set cudaFuncCachePreferShared manually per kernel.
#ifdef HAVE_CUDA
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
#endif

    // reset output/profiling variables
    allIterations = 0;
    allTimeSteps = 0;
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
#ifdef HAVE_HYPRE
    timer_hypre_conversion.reset();
    timer_hypre_setup.reset();
    timer_hypre_solve.reset();
    timer_hypre_synchronization.reset();
    hypre_updated_iters = 0;
    hypre_last_iters = 1;
    hypre_setup_counter = 0;
#endif

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
        if( TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 ) {
            namespace fs = std::filesystem;
            fs::path path( boundaryConditionsFile );
            std::string ext = path.extension();
            ext = "." + std::to_string( TNL::MPI::GetRank( distributedMeshPointer->getCommunicator() ) ) + ext;
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
            using std::visit;
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

    const IndexType offset = this->getDofsOffset();
    const IndexType localDofs = this->getLocalDofs();
    const IndexType dofs = this->getDofs();
    const IndexType globalDofs = this->getGlobalDofs();
    const TNL::Containers::Subrange< IndexType > localRange( offset, offset + localDofs );

    // initialize the right hand side vector
    rhsVector.setSize( dofs );

    using CompressedRowLengths =
        TNL::Containers::NDArray< IndexType,
                                  TNL::Containers::SizesHolder< IndexType, MeshDependentDataType::NumberOfEquations, 0 >,  // i, F
                                  std::index_sequence< 1, 0 >,  // F, i  (must match the permutation for mdd.Z_iF for all devices
                                  DeviceType >;

#ifdef HAVE_HYPRE
    CompressedRowLengths rowLengths_diag, rowLengths_offd;
    rowLengths_diag.setSizes( 0, localFaces );
    rowLengths_offd.setSizes( 0, localFaces );
    rowLengths_diag.setValue( -1 );
    rowLengths_offd.setValue( -1 );

    auto rowLengths_diag_view = rowLengths_diag.getView();
    auto rowLengths_offd_view = rowLengths_offd.getView();
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    auto kernel = [rowLengths_diag_view, rowLengths_offd_view, _mesh, _bc] __cuda_callable__ ( int i, IndexType E ) mutable
    {
        if( isBoundaryFace( *_mesh, E ) ) {
            const IndexType diag_count = _bc->getLinearSystemRowLengthDiag( *_mesh, E, i );
            rowLengths_diag_view( i, E ) = diag_count;
            rowLengths_offd_view( i, E ) = _bc->getLinearSystemRowLength( *_mesh, E, i ) - diag_count;
        }
        else {
            const IndexType diag_count = LinearSystem::getRowLengthDiag( *_mesh, E, i );
            rowLengths_diag_view( i, E ) = diag_count;
            rowLengths_offd_view( i, E ) = LinearSystem::getRowLength( *_mesh, E, i ) - diag_count;
        }
    };
    rowLengths_diag_view.forAll( kernel );

    // sanity check (doesn't happen if the traverser works, but this is pretty
    // hard to debug and the check does not cost us much in initialization)
    using IndexVectorViewType = TNL::Containers::VectorView< IndexType, DeviceType, IndexType >;
    IndexVectorViewType rowLengths_diag_vector( rowLengths_diag.getStorageArray().getView() );
    IndexVectorViewType rowLengths_offd_vector( rowLengths_offd.getStorageArray().getView() );
    if( TNL::min( rowLengths_diag_vector ) <= 0 ) {
        std::stringstream ss;
        ss << "MHFEM error: attempted to set invalid rowsLengths vector for the diag block:\n" << rowLengths_diag_vector << std::endl;
        throw std::runtime_error( ss.str() );
    }
    if( TNL::min( rowLengths_offd_vector ) < 0 ) {
        std::stringstream ss;
        ss << "MHFEM error: attempted to set invalid rowsLengths vector for the offd block:\n" << rowLengths_offd_vector << std::endl;
        throw std::runtime_error( ss.str() );
    }

    // initialize the diag block
    csr_diag.setDimensions( localDofs, localDofs );
    csr_diag.setRowCapacities( rowLengths_diag_vector );

    // initialize the offd block
    csr_offd.setDimensions( localDofs, dofs - localDofs );
    csr_offd.setRowCapacities( rowLengths_offd_vector );

    // initialize the offd col map
    CompressedRowLengths col_map_device;
    const IndexType faces = localMeshPointer->template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
    col_map_device.setSizes( 0, faces - localFaces );
    auto col_map_view = col_map_device.getView();
    const auto* _mdd = &mdd.template getData< DeviceType >();
    const auto& globalFaceIndices = distributedMeshPointer->template getGlobalIndices< MeshType::getMeshDimension() - 1 >().getConstView();
    const IndexType localFaces = this->localFaces;
    auto kernel_col_map = [col_map_view, _mdd, globalFaceIndices, localFaces] __cuda_callable__ ( IndexType E ) mutable
    {
        for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ ) {
            const IndexType globalDof = i + globalFaceIndices[ E ] * MeshDependentDataType::NumberOfEquations;
            col_map_view( i, E - localFaces ) = globalDof;
        }
    };
    TNL::Algorithms::parallelFor< DeviceType >( localFaces, faces, kernel_col_map );
    col_map_offd = col_map_device.getStorageArray();

    // initialize the parcsr matrix
    parcsr_A.bind( distributedMeshPointer->getCommunicator(),
                   globalDofs,
                   globalDofs,
                   localRange,
                   localRange,
                   csr_diag,
                   csr_offd,
                   col_map_offd.getData() );
#else
    CompressedRowLengths rowLengths;
    rowLengths.setSizes( 0, localFaces );
    rowLengths.setValue( -1 );

    auto rowLengths_view = rowLengths.getView();
    const MeshType* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    auto kernel = [rowLengths_view, _mesh, _bc] __cuda_callable__ ( int i, IndexType E ) mutable
    {
        IndexType length;
        if( isBoundaryFace( *_mesh, E ) )
            length = _bc->getLinearSystemRowLength( *_mesh, E, i );
        else
            length = LinearSystem::getRowLength( *_mesh, E, i );
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
    distributedMatrixPointer = std::make_shared< DistributedMatrixType >();
    distributedMatrixPointer->setDistribution( localRange, globalDofs, dofs, distributedMeshPointer->getCommunicator() );
    TNL::Containers::DistributedVectorView< IndexType, DeviceType, IndexType > dist_rowLengths( localRange, 0, globalDofs, distributedMatrixPointer->getCommunicator(), rowLengths_vector );
    distributedMatrixPointer->setRowCapacities( dist_rowLengths );

    // bind device pointer to the local matrix
    this->localMatrixPointer = LocalMatrixPointer( distributedMatrixPointer->getLocalMatrix() );
#endif
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
        TNL::Algorithms::parallelFor< DeviceType >( 0, cells, kernel );
    }
    timer_nonlinear.stop();

    // update coefficients b_ijKEF
    timer_b.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( const Index3D& idx ) mutable
        {
            using MassMatrix = typename MeshDependentDataType::MassMatrix;
            MassMatrix::update( *_mesh, *_mdd, idx[ 0 ], idx[ 1 ], idx[ 2 ] );
        };
        const Index3D begin( 0, 0, 0 );
        const Index3D end( cells, MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations );
        TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
    }
    timer_b.stop();

    // update vector coefficients (u, w, a), whose projection into the RTN_0(K) space
    // generally depends on the b_ijKEF coefficients
    timer_nonlinear.start();
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( const Index2D& idx ) mutable
        {
            _mdd->updateVectorCoefficients( *_mesh, idx[ 0 ], idx[ 1 ] );
        };
        const Index2D begin( 0, 0 );
        const Index2D end( cells, (IndexType) MeshDependentDataType::NumberOfEquations );
        TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
    }
    timer_nonlinear.stop();

    // update upwinded mobility values
    // NOTE: Upwinding is done based on v_{i,K,E}, which is computed from the "old" b_{i,j,K,E,F} and w_{i,K,E}
    //       coefficients, but "new" Z_{j,K} and Z_{j,F}. From the semi-implicit approach it follows that
    //       velocity calculated this way is conservative, which is very important for upwinding.
    timer_upwind.start();
    {
        const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();

        auto kernel_m_iE = [_mdd, _mesh, _bc, time, tau] __cuda_callable__ ( const Index2D& idx ) mutable
        {
            const IndexType& E = idx[ 0 ];
            const IndexType& i = idx[ 1 ];

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
        if constexpr( MeshDependentDataType::do_mobility_upwind ) {
            // mdd->m_iE_upw.forAll does not skip ghosts, so we use parallelFor manually for the specific permutation of indices
            const Index2D begin( 0, 0 );
            const Index2D end( localFaces, MeshDependentDataType::NumberOfEquations );
            TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel_m_iE );
        }

        auto kernel_Z_ijE = [_mdd, _mesh] __cuda_callable__ ( const Index3D& idx ) mutable
        {
            const IndexType& E = idx[ 0 ];
            const IndexType& j = idx[ 1 ];
            const IndexType& i = idx[ 2 ];

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
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind ) {
            // mdd->Z_ijE_upw.forAll does not skip ghosts, so we use parallelFor manually for the specific permutation of indices
            const Index3D begin( 0, 0, 0 );
            const Index3D end( localFaces, MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations );
            TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel_Z_ijE );
        }
    }
    timer_upwind.stop();

    // synchronize the upwinded quantities
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        timer_mpi_upwind.start();

        // NOTE: this is specific to how the ndarrays are ordered
        if constexpr( MeshDependentDataType::do_mobility_upwind ) {
            auto m_upw_view = mdd->m_iE_upw.getStorageArray().getView();
            faceSynchronizer->synchronizeArray( m_upw_view, MeshDependentDataType::NumberOfEquations );
        }

        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind ) {
            auto Z_upw_view = mdd->Z_ijE_upw.getStorageArray().getView();
            faceSynchronizer->synchronizeArray( Z_upw_view, MeshDependentDataType::NumberOfEquations * MeshDependentDataType::NumberOfEquations );
        }

        timer_mpi_upwind.stop();
    }

    timer_R.start();
    {
        auto kernel = [_mdd, _mesh, tau] __cuda_callable__ ( const Index2D& idx ) mutable
        {
            const IndexType& K = idx[ 0 ];
            const IndexType& i = idx[ 1 ];

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
        const Index2D begin( 0, 0 );
        const Index2D end( cells, MeshDependentDataType::NumberOfEquations );
        TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
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
            // (we'll need to pass custom launch configuration to the parallelFor)
            // Now we just assume that the parallelFor kernel uses 256 threads per block.
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

            using namespace TNL::Matrices::Factorization;
            LU_sequential_factorize( Q );

            RealType rhs[ MeshDependentDataType::NumberOfEquations ];

            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                rhs[ i ] = _mdd->R_iK( i, K );
            LU_sequential_solve_inplace( Q, rhs );
            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                _mdd->R_iK( i, K ) = rhs[ i ];

            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ ) {
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        rhs[ i ] = _mdd->R_ijKe( i, j, K, e );
                    LU_sequential_solve_inplace( Q, rhs );
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        _mdd->R_ijKe( i, j, K, e ) = rhs[ i ];
                }
        };
        TNL::Algorithms::parallelFor< DeviceType >( 0, cells, kernel );
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

#ifdef HAVE_HYPRE
    const auto* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _mdd = &mdd.template modifyData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    auto diag_view = csr_diag.getView();
    auto offd_view = csr_offd.getView();
    auto _b = rhsVector.getView();
    auto kernel = [_mesh, _mdd, _bc, diag_view, offd_view, _b, time, tau] __cuda_callable__ ( const Index2D& idx ) mutable
    {
        const IndexType& E = idx[ 0 ];
        const IndexType& i = idx[ 1 ];

        TNL_ASSERT_FALSE( _mesh->template isGhostEntity< MeshType::getMeshDimension() - 1 >( E ),
                          "A ghost face encountered while assembling the linear system." );
        const IndexType rowIndex = _mdd->getRowIndex( i, E );
        TNL_ASSERT_LT( rowIndex, diag_view.getRows(), "bug in getRowIndex" );
        if( isBoundaryFace( *_mesh, E ) )
            _bc->setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, diag_view, offd_view, _b );
        else {
            const RealType diagonalValue = LinearSystem::setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, diag_view, offd_view, _b );
            _b[ rowIndex ] = LinearSystem::RHS::getValue( *_mesh, *_mdd, E, i ) / diagonalValue;
        }
    };
    // mdd->Z_iF.forAll does not skip ghosts, so we use parallelFor manually for the specific permutation of indices
    const Index2D begin( 0, 0 );
    const Index2D end( localFaces, MeshDependentDataType::NumberOfEquations );
    TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
#else
    const auto* _mesh = &localMeshPointer.template getData< DeviceType >();
    const auto* _mdd = &mdd.template modifyData< DeviceType >();
    const auto* _bc = &boundaryConditionsPointer.template getData< DeviceType >();
    auto* _matrix = &localMatrixPointer.template modifyData< DeviceType >();
    auto _b = rhsVector.getView();
    auto kernel = [_mesh, _mdd, _bc, _matrix, _b, time, tau] __cuda_callable__ ( const Index2D& idx ) mutable
    {
        const IndexType& E = idx[ 0 ];
        const IndexType& i = idx[ 1 ];

        TNL_ASSERT_FALSE( _mesh->template isGhostEntity< MeshType::getMeshDimension() - 1 >( E ),
                          "A ghost face encountered while assembling the linear system." );
        const IndexType rowIndex = _mdd->getRowIndex( i, E );
        TNL_ASSERT_LT( rowIndex, _matrix->getRows(), "bug in getRowIndex" );
        if( isBoundaryFace( *_mesh, E ) )
            _bc->setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, *_matrix, _b );
        else {
            const RealType diagonalValue = LinearSystem::setMatrixElements( *_mesh, *_mdd, rowIndex, E, i, time + tau, tau, *_matrix, _b );
            _b[ rowIndex ] = LinearSystem::RHS::getValue( *_mesh, *_mdd, E, i ) / diagonalValue;
        }
    };
    TNL_ASSERT_EQ( localMatrixPointer->getRows(), MeshDependentDataType::NumberOfEquations * localFaces, "BUG: wrong matrix size" );
    // mdd->Z_iF.forAll does not skip ghosts, so we use parallelFor manually for the specific permutation of indices
    const Index2D begin( 0, 0 );
    const Index2D end( localFaces, MeshDependentDataType::NumberOfEquations );
    TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
#endif

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

    allTimeSteps++;

    const IndexType offset = this->getDofsOffset();
    const IndexType localDofs = this->getLocalDofs();
    const IndexType globalDofs = this->getGlobalDofs();
    const TNL::Containers::Subrange< IndexType > localRange( offset, offset + localDofs );

#if defined( HAVE_GINKGO )
    // Avoid compiler warning since Ginkgo is non-MPI
    (void) globalDofs;

    // Create a Ginkgo Csr view
    auto gko_A = gko::share( TNL::Matrices::getGinkgoMatrixCsrView( gko_exec, distributedMatrixPointer->getLocalMatrix() ) );

    // Wrap the vectors
    DofViewType dofs_view = mdd->Z_iF.getStorageArray().getView();
    auto gko_b = TNL::Containers::GinkgoVector< RealType, DeviceType >::create( gko_exec, rhsVector.getView() );
    auto gko_x = TNL::Containers::GinkgoVector< RealType, DeviceType >::create( gko_exec, dofs_view );

    // Output the initial guess (NOTE: debug only)
//    gko::write( std::ofstream( outputDirectory + "/x0.mtx" ), gko::lend( gko_x ) );

    // Pass the solver monitor to the logger
    gko_convergence_logger->set_solver_monitor( solverMonitor );

    // Check if the preconditioner should be reused
    // TODO: figure out logging and better parametrization
//    const bool reuse_preconditioner = gko_last_iters <= 2 * gko_updated_iters;
    const bool reuse_preconditioner = gko_last_iters <= TNL::min( 2 * gko_updated_iters, gko_updated_iters + 5 );
//    std::cout << "reuse preconditioner: " << reuse_preconditioner << "  last iters: " << gko_last_iters << " updated iters: " << gko_updated_iters << std::endl;

    if( ! reuse_preconditioner ) {
        timer_linearPreconditioner.start();
        // clear the previous preconditioner before allocating storage for the new one
        gko_preconditioner = nullptr;
        switch( gko_preconditioner_type ) {
            case AMGX:
            {
                // Create smoother factory (ir with bj)
                auto inner_solver_gen = gko::share(
                    gko::preconditioner::Jacobi< RealType, IndexType >::build()
                        .with_max_block_size( MeshDependentDataType::NumberOfEquations )
                        .on( gko_exec )
                );
                auto smoother_gen = gko::share(
                    gko::solver::Ir< RealType >::build()
                        .with_solver( inner_solver_gen )
                        .with_relaxation_factor( 0.9 )
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters( 2 ).on( gko_exec ) )
                        .on( gko_exec )
                );
                // Create MultigridLevel factory
                auto mg_level_gen = gko::share(
                    gko::multigrid::Pgm< RealType, IndexType >::build()
                        .with_deterministic( true )
                        .with_max_iterations( 15 )  // default: 15
                        .with_max_unassigned_ratio( 0.1 )  // default: 0.05
                        .on( gko_exec )
                );
                // Create CoarsestSolver factory
                auto coarsest_gen = gko::share(
                    gko::solver::Ir< RealType >::build()
                        .with_solver( inner_solver_gen )
                        .with_relaxation_factor( 0.9 )
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters( 4 ).on( gko_exec ) )
                        .on( gko_exec )
                );
                // Create multigrid factory
                auto multigrid_gen = gko::share(
                    gko::solver::Multigrid::build()
                        .with_max_levels( 25 )
                        .with_min_coarse_rows( 4 )
                        .with_pre_smoother( smoother_gen )
                        .with_post_uses_pre( true )
                        .with_mg_level( mg_level_gen )
                        .with_coarsest_solver( coarsest_gen )
                        // convergence only with zero initial guess
                        .with_default_initial_guess( gko::solver::initial_guess_mode::zero )
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters( 1 ).on( gko_exec ) )
                        .on( gko_exec )
                );
                gko_preconditioner = gko::share( multigrid_gen->generate( gko_A ) );
                break;
            }
            case ILU_ISAI:
            case PARILU_ISAI:
            case PARILUT_ISAI:
            {
                std::shared_ptr< gko::LinOpFactory > fact_factory = nullptr;
                if( gko_preconditioner_type == ILU_ISAI )
                    // Generate incomplete factors using ILU
                    fact_factory = gko::share(
                        gko::factorization::Ilu< RealType, IndexType >::build()
                            .on( gko_exec )
                    );
                else if( gko_preconditioner_type == PARILU_ISAI )
                    // Generate incomplete factors using ParILU
                    fact_factory = gko::share(
                        gko::factorization::ParIlu< RealType, IndexType >::build()
                            // TODO: parameters: iterations
                            .on( gko_exec )
                    );
                else if( gko_preconditioner_type == PARILUT_ISAI )
                    // Generate incomplete factors using ParILUT
                    fact_factory = gko::share(
                        gko::factorization::ParIlut< RealType, IndexType >::build()
                            // TODO: parameters: fill-in limit, iterations
                            .on( gko_exec )
                    );

                // Generate an ILU preconditioner factory by setting lower and upper triangular solver
#if 0
                // exact triangular solves
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu< gko::solver::LowerTrs< RealType, IndexType >,
                                              gko::solver::UpperTrs< RealType, IndexType > >::build()
                        .on( gko_exec );
#endif
                // incomplete sparse approximate inverse
                const int sparsity_power = 2; // TODO: parametrize
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu< gko::preconditioner::LowerIsai< RealType, IndexType >,
                                              gko::preconditioner::UpperIsai< RealType, IndexType > >::build()
                        .with_factorization_factory( fact_factory )
                        .with_l_solver_factory(
                            gko::preconditioner::LowerIsai< RealType, IndexType >::build()
                                .with_sparsity_power( sparsity_power )
                                .on( gko_exec )
                            )
                        .with_u_solver_factory(
                            gko::preconditioner::UpperIsai< RealType, IndexType >::build()
                                .with_sparsity_power( sparsity_power )
                                .on( gko_exec )
                            )
                        .on( gko_exec );

                // Use incomplete factors to generate ILU preconditioner
                gko_preconditioner = gko::share( ilu_pre_factory->generate( gko_A ) );
                break;
            }
        }
        timer_linearPreconditioner.stop();
    }

    timer_linearSolver.start();

    // Create the solver
    auto solver_factory =
            gko::solver::Bicgstab< RealType >::build()
//                .with_preconditioner(
//                    gko::preconditioner::Jacobi< RealType, IndexType >::build()
//                        .on( gko_exec ) )
                .with_generated_preconditioner( gko_preconditioner )
                .with_criteria( gko_stop_iter, gko_stop_tol )
                .on( gko_exec );
    auto solver = solver_factory->generate( gko_A );

    // Solve the system
    solver->apply( lend( gko_b ), lend( gko_x ) );
    timer_linearSolver.stop();

    // Check the result
    const bool converged = gko_convergence_logger->has_converged();
    const std::size_t num_iterations = gko_convergence_logger->get_num_iterations();
    allIterations += num_iterations;

    gko_last_iters = num_iterations;
    if( ! reuse_preconditioner || gko_updated_iters == 0 )
        gko_updated_iters = num_iterations;
    if( ! reuse_preconditioner )
        gko_setup_counter++;

    if( ! converged ) {
        // save the linear system for debugging
        // TODO: save the distributed system
        gko::write( std::ofstream( outputDirectory + "/A.mtx" ), gko::lend( gko_A ) );
        gko::write( std::ofstream( outputDirectory + "/x.mtx" ), gko::lend( gko_x ) );
        gko::write( std::ofstream( outputDirectory + "/b.mtx" ), gko::lend( gko_b ) );
        std::cerr << "The linear system has been saved to " << outputDirectory << "/{A,x,b}.mtx" << std::endl;
        throw std::runtime_error( "MHFEM error: the linear system solver did not converge (" + std::to_string(num_iterations) + " iterations)." );
    }
#elif defined( HAVE_HYPRE )
    // Check if the preconditioner should be reused
    // TODO: figure out logging and better parametrization
//    const bool reuse_preconditioner = hypre_last_iters <= 2 * hypre_updated_iters;
    const bool reuse_preconditioner = hypre_last_iters <= TNL::min( 2 * hypre_updated_iters, hypre_updated_iters + 5 );
//    std::cout << "reuse preconditioner: " << reuse_preconditioner << "  last iters: " << hypre_last_iters << " updated iters: " << hypre_updated_iters << std::endl;

    // Set the matrix of the linear system
    hypre_solver->setMatrix( parcsr_A, reuse_preconditioner );

    // Prepare parallel vectors
    DofViewType dofs_view = mdd->Z_iF.getStorageArray().getView( 0, localDofs );
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_dofs( localRange, 0, globalDofs, distributedMeshPointer->getCommunicator(), dofs_view );
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_rhs( localRange, 0, globalDofs, distributedMeshPointer->getCommunicator(), rhsVector.getView( 0, localDofs ) );

    TNL::Containers::HypreParVector par_b;
    TNL::Containers::HypreParVector par_x;
    par_b.bind( dist_rhs );
    par_x.bind( dist_dofs );

    // Solve the linear system
    if( ! reuse_preconditioner ) {
        timer_hypre_setup.start();
        hypre_solver->setup( par_b, par_x );
        timer_hypre_setup.stop();
    }
    timer_hypre_solve.start();
    hypre_solver->solve( par_b, par_x );
    timer_hypre_solve.stop();

    // synchronize the solution
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        timer_hypre_synchronization.start();
        auto dofs_view_with_ghosts = mdd->Z_iF.getStorageArray().getView();
        faceSynchronizer->synchronizeArray( dofs_view_with_ghosts, MeshDependentDataType::NumberOfEquations );
        timer_hypre_synchronization.stop();
    }

    const long long int num_iterations = hypre_solver->getNumIterations();
    double final_res_norm;
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm( *hypre_solver, &final_res_norm );
    if( solverMonitor ) {
        solverMonitor->setIterations( num_iterations );
        solverMonitor->setResidue( final_res_norm );
    }
    allIterations += num_iterations;

    hypre_last_iters = num_iterations;
    if( ! reuse_preconditioner || hypre_updated_iters == 0 )
        hypre_updated_iters = num_iterations;
    if( ! reuse_preconditioner )
        hypre_setup_counter++;

    // disable printing setup info + parameters after the first time step
    HYPRE_BoomerAMGSetPrintLevel( *hypre_precond, 0 );
#else
    if( solverMonitor )
        linearSystemSolver->setSolverMonitor( *solverMonitor );

    // set the matrix for the linear solver
    linearSystemSolver->setMatrix( distributedMatrixPointer );

    if( preconditioner )
    {
        timer_linearPreconditioner.start();
        preconditioner->update( distributedMatrixPointer );
        timer_linearPreconditioner.stop();
    }

    timer_linearSolver.start();

    const IndexType dofs = this->getDofs();
    DofViewType dofs_view = mdd->Z_iF.getStorageArray().getView();
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_dofs( localRange, dofs - localDofs, globalDofs, distributedMatrixPointer->getCommunicator(), dofs_view );
    TNL::Containers::DistributedVectorView< RealType, DeviceType, IndexType > dist_rhs( localRange, dofs - localDofs, globalDofs, distributedMatrixPointer->getCommunicator(), rhsVector.getView() );
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
#endif
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
    if( MeshDependentDataType::do_mobility_upwind )
    {
        auto kernel = [_mdd, _mesh] __cuda_callable__ ( const Index2D idx ) mutable
        {
            const IndexType& K = idx[ 0 ];
            const IndexType& i = idx[ 1 ];

            const auto faceIndexes = getFacesForCell( *_mesh, K );
            for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ )
                _mdd->v_iKe( i, K, e ) = coeff::v_iKE( *_mdd, faceIndexes, i, K, faceIndexes[ e ], e );
        };
        const Index2D begin( 0, 0 );
        const Index2D end( cells, MeshDependentDataType::NumberOfEquations );
        TNL::Algorithms::parallelFor< DeviceType >( begin, end, kernel );
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
    const TNL::MPI::Comm& communicator = distributedMeshPointer->getCommunicator();
    auto log_mpi_value = [&logger, &communicator] ( std::string prefix, auto local_value )
    {
        const auto avg = TNL::MPI::reduce( local_value, MPI_SUM, communicator ) / communicator.size();
        const double stddev = std::sqrt( TNL::MPI::reduce( TNL::pow( local_value - avg, 2 ), MPI_SUM, communicator ) / communicator.size() );
        const auto min = TNL::MPI::reduce( local_value, MPI_MIN, communicator );
        const auto max = TNL::MPI::reduce( local_value, MPI_MAX, communicator );

        std::stringstream str;
        str << std::scientific << "avg: " << avg << " stddev: " << stddev << " min: " << min << " max: " << max;
        logger.writeParameter< std::string >( prefix + " ", str.str() );
    };

    logger.writeParameter< long long int >( "Total number of linear solver iterations:", allIterations );
    logger.writeParameter< long long int >( "Total number of time steps:", allTimeSteps );
#if defined( HAVE_GINKGO )
    logger.writeParameter< long long int >( "Number of time steps in which the preconditioner was updated:", gko_setup_counter );
#elif defined( HAVE_HYPRE )
    logger.writeParameter< long long int >( "Number of time steps in which the preconditioner was updated:", hypre_setup_counter );
#endif
    log_mpi_value( "Pre-iterate time:", timer_preIterate.getRealTime() );
    log_mpi_value( "  nonlinear update time:", timer_nonlinear.getRealTime() );
    log_mpi_value( "  update_b time:", timer_b.getRealTime() );
    log_mpi_value( "  upwind update time:", timer_upwind.getRealTime() );
    log_mpi_value( "  upwind MPI synchronization time:", timer_mpi_upwind.getRealTime() );
    log_mpi_value( "  update_R time:", timer_R.getRealTime() );
    log_mpi_value( "  update_Q time:", timer_Q.getRealTime() );
    log_mpi_value( "  model pre-iterate time:", timer_model_preIterate.getRealTime() );
    log_mpi_value( "Linear system assembler time:", timer_assembleLinearSystem.getRealTime() );
#ifdef HAVE_HYPRE
    log_mpi_value( "Hypre matrix conversion time:", timer_hypre_conversion.getRealTime() );
    log_mpi_value( "Hypre setup time:", timer_hypre_setup.getRealTime() );
    log_mpi_value( "Hypre solve time:", timer_hypre_solve.getRealTime() );
    log_mpi_value( "Hypre result MPI synchronization time:", timer_hypre_synchronization.getRealTime() );
#else
    log_mpi_value( "Linear preconditioner update time:", timer_linearPreconditioner.getRealTime() );
    log_mpi_value( "Linear system solver time:", timer_linearSolver.getRealTime() );
    if( distributedMeshPointer->getCommunicator() != MPI_COMM_NULL
        && TNL::MPI::GetSize( distributedMeshPointer->getCommunicator() ) > 1 )
    {
        const double total_mpi_time = faceSynchronizer->async_wait_before_start_timer.getRealTime()
                                    + faceSynchronizer->async_start_timer.getRealTime()
                                    + faceSynchronizer->async_wait_timer.getRealTime();
        logger.writeParameter< std::size_t >( "  faceSynchronizer async operations count:", faceSynchronizer->async_ops_count );
        log_mpi_value( "  faceSynchronizer async operations time:", total_mpi_time );
        log_mpi_value( "    async wait before start time:", faceSynchronizer->async_wait_before_start_timer.getRealTime() );
        log_mpi_value( "    async start time:", faceSynchronizer->async_start_timer.getRealTime() );
        log_mpi_value( "    async wait time:", faceSynchronizer->async_wait_timer.getRealTime() );
    }
#endif
    log_mpi_value( "Post-iterate time:", timer_postIterate.getRealTime() );
    log_mpi_value( "  Z_iF -> Z_iK update time:", timer_explicit.getRealTime() );
    log_mpi_value( "  velocities update time:", timer_velocities.getRealTime() );
    log_mpi_value( "  model post-iterate time:", timer_model_postIterate.getRealTime() );
    if( communicator.size() > 1 ) {
        logger.writeParameter< std::string >( "MPI operations (included in the previous phases):", "" );
        log_mpi_value( "  MPI_Allreduce time:", TNL::MPI::getTimerAllreduce().getRealTime() );
    }
}

template< typename MeshDependentData,
          typename BoundaryModel,
          typename Matrix >
void
Solver< MeshDependentData, BoundaryModel, Matrix >::
estimateMemoryDemands( const DistributedHostMeshType & mesh, std::ostream & out )
{
    if( mesh.getCommunicator() == MPI_COMM_NULL ) {
        out << "The mesh is on MPI_COMM_NULL, no memory estimate." << std::endl;
        return;
    }

    const HostMeshType & localMesh = mesh.getLocalMesh();
    const std::size_t cells = localMesh.template getEntitiesCount< typename MeshType::Cell >();
    const std::size_t faces = localMesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
    const std::size_t localFaces = localMesh.template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >();

    constexpr int FacesPerCell = MeshDependentDataType::FacesPerCell;
    constexpr int n = MeshDependentDataType::NumberOfEquations;

    // TODO: report exact size of the current mesh

    // mesh/model-dependent data
    const std::size_t mdd_size = MeshDependentDataType::estimateMemoryDemands( mesh );

    // boundary conditions
    const std::size_t bc_size =
        // tags
        + n * faces * sizeof(std::uint8_t)
        // values
        + n * faces * sizeof(RealType)
        // dirichletValues
        + n * faces * sizeof(RealType)
    ;

    // sparse matrix (only upper bound)
    const std::size_t matrix_size = n * localFaces * ( 2 * FacesPerCell - 1 ) * n * sizeof(RealType);

    // DOF and RHS vectors
    const std::size_t dof_size = n * faces * sizeof(RealType);
    const std::size_t rhs_size = n * localFaces * sizeof(RealType);

    // linear solver and preconditioner
    const std::size_t solver_size = 7 * dof_size;
    const std::size_t preconditioner_size = dof_size;

    // total
    const std::size_t total_size = mdd_size + bc_size + matrix_size + dof_size + rhs_size + solver_size + preconditioner_size;

    // GPU memory usage
    std::size_t gpu_total = 0;
    std::size_t gpu_free = 0;
    #ifdef HAVE_CUDA
    {
        int gpu_id;
        cudaGetDevice(&gpu_id);
        cudaMemGetInfo(&gpu_free, &gpu_total);
    }
    #endif

    auto format = []( std::size_t value )
    {
        const std::size_t MiB = value / 1024.0 / 1024.0;
        return std::to_string(MiB) + " MiB";
    };

    out << "Mesh size:\n"
        << "- cells count:\t" << cells << "\n"
        << "- faces count:\t" << faces << "\n"
        << "- local faces count:\t" << localFaces << "\n"
    ;
    out << "(Estimated) memory demands:\n"
        << "- mesh-dependent data:\t" << format(mdd_size) << "\n"
        << "- boundary conditions:\t" << format(bc_size) << "\n"
        << "- sparse matrix:\t" << format(matrix_size) << "\n"
        << "- DOF vector:\t" << format(dof_size) << "\n"
        << "- RHS vector:\t" << format(rhs_size) << "\n"
        << "- linear system solver:\t" << format(solver_size) << "\n"
        << "  (N * DOF vector size, where e.g. N = 7 for BiCGstab)\n"
        << "- preconditioner:\t" << format(preconditioner_size) << "\n"
        << "  (DOF vector size for Jacobi/diagonal, sparse matrix size for ILU(0))\n"
        << "Total GPU memory needed: " << format(total_size) << " (" << 100.0 * total_size / (gpu_total + 1) << "%)\n"
        << "Current GPU memory: available " << format(gpu_free) << ", total " << format(gpu_total) << "\n"
    ;
    out.flush();
}

} // namespace mhfem
