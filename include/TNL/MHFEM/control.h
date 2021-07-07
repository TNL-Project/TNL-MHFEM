#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Pointers/SmartPointersRegister.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#ifdef HAVE_MPI
#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>
#endif

#include "MassMatrix.h"

namespace mhfem {

#ifdef HAVE_MPI
template< typename DistributedMesh,
          std::enable_if_t< (DistributedMesh::getMeshDimension() > 1), bool > = true >
void distributeFaces( DistributedMesh& mesh )
{
    TNL::Meshes::DistributedMeshes::distributeSubentities< DistributedMesh::getMeshDimension() - 1 >( mesh );
}

template< typename DistributedMesh,
          std::enable_if_t< DistributedMesh::getMeshDimension() == 1, bool > = true >
void distributeFaces( DistributedMesh& mesh )
{
}
#endif

template< typename Problem >
void init( Problem& problem,
           typename Problem::DistributedHostMeshPointer& meshPointer,
           const TNL::Config::ParameterContainer& parameters )
{
    const TNL::String meshFile = parameters.getParameter< TNL::String >( "mesh" );
    const TNL::String meshFileFormat = parameters.getParameter< TNL::String >( "mesh-format" );
#ifdef HAVE_MPI
    if( ! TNL::Meshes::loadDistributedMesh( *meshPointer, meshFile, meshFileFormat ) )
        throw std::runtime_error( "failed to load the distributed mesh from file " + meshFile );

    if( meshPointer->getCommunicationGroup() != TNL::MPI::NullGroup() )
        // distribute faces
        distributeFaces( *meshPointer );
#else
    if( ! TNL::Meshes::loadMesh( meshPointer->getLocalMesh(), meshFile, meshFileFormat ) )
        throw std::runtime_error( "failed to load the mesh from file " + meshFile );
    meshPointer->setCommunicationGroup( TNL::MPI::AllGroup() );
#endif

    problem.setMesh( meshPointer );

    if( ! problem.setup( parameters ) )
        throw std::runtime_error( "Failed to set up the MHFEM solver." );

    // set the initial condition
    if( ! problem.setInitialCondition( parameters ) )
        throw std::runtime_error( "Failed to set up the initial condition." );

    problem.setupLinearSystem();
}

template< typename Problem >
void solve( Problem& problem,
            double startTime,
            double stopTime,
            double timeStep,
            TNL::Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType >* solverMonitor = nullptr )
{
    double t = startTime;

    // ignore very small steps at the end, most likely caused by truncation errors
    while( stopTime - t > timeStep * 1e-6 )
    {
        double currentTau = TNL::min( timeStep, stopTime - t );

        if( solverMonitor ) {
            solverMonitor->setTime( t );
            solverMonitor->setStage( "Preiteration" );
        }

        problem.preIterate( t, currentTau );

        if( solverMonitor )
            solverMonitor->setStage( "Assembling the linear system" );

        problem.assembleLinearSystem( t, currentTau );

        if( solverMonitor )
            solverMonitor->setStage( "Solving the linear system" );

        problem.solveLinearSystem( solverMonitor );

        if( solverMonitor )
            solverMonitor->setStage( "Postiteration" );

        problem.postIterate( t, currentTau );

        t += currentTau;
   }
}

template< typename Problem >
void solve( Problem& problem,
            const TNL::Config::ParameterContainer& parameters,
            TNL::Timer& computeTimer,
            TNL::Timer& ioTimer,
            TNL::Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType >* solverMonitor = nullptr )
{
    const double finalTime       = parameters.getParameter< double >( "final-time" );
    const double initialTime     = parameters.getParameter< double >( "initial-time" );
    const double snapshotPeriod  = parameters.getParameter< double >( "snapshot-period" );
    const double timeStep        = parameters.getParameter< double >( "time-step");

    if( finalTime <= initialTime )
        throw TNL::Exceptions::ConfigError( "Final time must larger than the initial time." );
    if( snapshotPeriod <= 0 )
        throw TNL::Exceptions::ConfigError( "Snapshot period must be positive value." );
    if( timeStep <= 0 )
        throw TNL::Exceptions::ConfigError( "Time step must be positive value." );

    double t = initialTime;
    std::size_t step = 0;
    std::size_t allSteps = std::ceil( ( finalTime - initialTime ) / snapshotPeriod );

    ioTimer.start();
    problem.makeSnapshot( t, step );
    ioTimer.stop();
    computeTimer.start();

    while( step < allSteps ) {
        const double tau = std::min( snapshotPeriod, finalTime - t );
        solve( problem, t, t + tau, timeStep, solverMonitor );
        step++;
        t += tau;

        computeTimer.stop();
        ioTimer.start();
        if( solverMonitor ) {
            solverMonitor->setTime( t );
            solverMonitor->setStage( "Making snapshot" );
        }
        problem.makeSnapshot( t, step );
        ioTimer.stop();
        computeTimer.start();
    }
    computeTimer.stop();
}

template< typename Problem >
void writeProlog( TNL::Logger& logger, bool writeSystemInformation = true )
{
    const bool printGPUs = std::is_same< typename Problem::DeviceType, TNL::Devices::Cuda >::value;

    logger.writeHeader( Problem::getPrologHeader() );
    if( TNL::MPI::isInitialized() )
        logger.writeParameter( "MPI processes:", TNL::MPI::GetSize() );
    logger.writeParameter< TNL::String >( "Device type:",   TNL::getType< typename Problem::DeviceType >() );
    if( ! printGPUs ) {
        if( TNL::Devices::Host::isOMPEnabled() ) {
            logger.writeParameter< TNL::String >( "OMP enabled:", "yes", 1 );
            logger.writeParameter< int >( "OMP threads:", TNL::Devices::Host::getMaxThreadsCount(), 1 );
        }
        else
            logger.writeParameter< TNL::String >( "OMP enabled:", "no", 1 );
    }
    logger.writeParameter< TNL::String >( "Real type:",     TNL::getType< typename Problem::RealType >() );
    logger.writeParameter< TNL::String >( "Index type:",    TNL::getType< typename Problem::IndexType >() );
    logger.writeParameter< TNL::String >( "Mesh type:",     TNL::getType< typename Problem::MeshType >() );
    logger.writeParameter< TNL::String >( "Sparse matrix:", TNL::getType< typename Problem::MatrixType >() );
    TNL::String massLumping;
    if( Problem::MeshDependentDataType::MassMatrix::lumping == mhfem::MassLumping::enabled )
        massLumping = "enabled";
    else
        massLumping = "disabled";
    logger.writeParameter< TNL::String >( "Mass lumping:", massLumping );
    Problem::MeshDependentDataType::writeProlog( logger );
    logger.writeSeparator();
    if( writeSystemInformation ) {
        logger.writeSystemInformation( printGPUs );
        logger.writeSeparator();
        logger.writeCurrentTime( "Started at:" );
        logger.writeSeparator();
    }
}

template< typename Problem >
void writeEpilog( TNL::Logger& logger,
                  const Problem& problem,
                  const TNL::Timer& computeTimer = TNL::Timer{},
                  const TNL::Timer& ioTimer = TNL::Timer{},
                  const TNL::Timer& totalTimer = TNL::Timer{} )
{
    logger.writeSeparator();
    logger.writeCurrentTime( "Finished at:" );
    problem.writeEpilog( logger );
    logger.writeParameter< double >( "Compute time:", computeTimer.getRealTime() );
    if( std::is_same< typename Problem::DeviceType, TNL::Devices::Cuda >::value ) {
        logger.writeParameter< const char* >( "GPU synchronization time:", "" );
        TNL::Pointers::getSmartPointersSynchronizationTimer< TNL::Devices::Cuda >().writeLog( logger, 1 );
    }
    logger.writeParameter< double >( "I/O time:", ioTimer.getRealTime() );
    logger.writeParameter< double >( "Total time:", totalTimer.getRealTime() );
    logger.writeSeparator();
}

template< typename Problem >
bool execute( const TNL::Config::ParameterContainer& controlParameters,
              const TNL::Config::ParameterContainer& solverParameters )
{
    TNL::Timer totalTimer, computeTimer, ioTimer;
    totalTimer.start();

    using SolverMonitorType = TNL::Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType >;
    SolverMonitorType solverMonitor;
    const int verbose = controlParameters.getParameter< int >( "verbose" );
    solverMonitor.setVerbose( verbose );
    solverMonitor.setTimer( totalTimer );

    // open the log file
    const TNL::String logFileName = controlParameters.getParameter< TNL::String >( "log-file" );
    std::ofstream logFile( logFileName.getString() );
    if( ! logFile ) {
        std::cerr << "Unable to open the log file " << logFileName << "." << std::endl;
        return false;
    }

    // create loggers
    const int logWidth = controlParameters.getParameter< int >( "log-width" );
    TNL::Logger consoleLogger( logWidth, std::cout );
    TNL::Logger logger( logWidth, logFile );

    Problem problem;
    auto meshPointer = std::make_shared< typename Problem::DistributedHostMeshType >();
    TNL::String stage;

    auto run = [&] ()
    {
        stage = "MHFEM initialization";
        mhfem::init( problem, meshPointer, solverParameters );

        // write a prolog
        if( verbose )
            writeProlog< Problem >( consoleLogger );
        writeProlog< Problem >( logger );

        // make sure that only the master rank has enabled monitor thread
        if( TNL::MPI::GetRank() > 0 )
            solverMonitor.stopMainLoop();

        // create solver monitor thread
        TNL::Solvers::SolverMonitorThread t( solverMonitor );

        stage = "MHFEM solver";
        mhfem::solve( problem, solverParameters, computeTimer, ioTimer, &solverMonitor );

        // stop timers
        computeTimer.stop();
        totalTimer.stop();

        // stop the solver monitor
        solverMonitor.stopMainLoop();

        // write an epilog
        if( verbose )
            writeEpilog( consoleLogger, problem, computeTimer, ioTimer, totalTimer );
        writeEpilog( logger, problem, computeTimer, ioTimer, totalTimer );
        logFile.close();
    };

    // catching exceptions ala gtest:
    // https://github.com/google/googletest/blob/59c795ce08be0c8b225bc894f8da6c7954ea5c14/googletest/src/gtest.cc#L2409-L2431
    const bool catch_exceptions = controlParameters.getParameter< bool >( "catch-exceptions" );
    if( catch_exceptions ) {
        try {
            run();
        }
        catch ( const std::exception& e ) {
            std::cerr << stage << " failed due to a C++ exception with description: " << e.what() << std::endl;
            logFile   << stage << " failed due to a C++ exception with description: " << e.what() << std::endl;
            return false;
        }
        catch (...) {
            std::cerr << stage << " failed due to an unknown C++ exception." << std::endl;
            logFile   << stage << " failed due to an unknown C++ exception." << std::endl;
            throw;
        }
    }
    else {
        run();
    }

    return true;
}

} // namespace mhfem
