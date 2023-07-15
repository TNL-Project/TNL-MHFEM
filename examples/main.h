#ifndef NDEBUG
#include <TNL/Debugging/FPE.h>
#endif

#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

#ifdef HAVE_HYPRE
#include <TNL/Hypre.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Config/parseINIConfigFile.h>
#include <TNL/MHFEM/configSetup.h>
#include <TNL/MHFEM/control.h>
#include <TNL/MHFEM/Solver.h>
using Problem = TNL::MHFEM::Solver< Model, BoundaryModel, Matrix >;

int main( int argc, char* argv[] )
{
#ifndef NDEBUG
    TNL::Debugging::trackFloatingPointExceptions();
#endif

    TNL::MPI::ScopedInitializer mpi( argc, argv );

#ifdef HAVE_HYPRE
    TNL::Hypre hypre;
#endif

    // get CLI parameters
    TNL::Config::ParameterContainer cliParams;
    TNL::Config::ConfigDescription cliConfig;

    cliConfig.addRequiredEntry< std::string >( "config", "Path to the configuration file." );
    cliConfig.addEntry< bool >( "config-help", "Print the configuration file description and exit.", false );
    cliConfig.addEntry< bool >( "print-static-configuration", "Print the static configuration (e.g. solver and model types) and exit.", false );
    cliConfig.addEntry< std::string >( "output-directory", "Path to the output directory (overrides the corresponding option in the configuration file)." );
    cliConfig.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 2 );
    cliConfig.addEntry< std::string >( "log-file", "Log file for the computation.", "log.txt" );
    cliConfig.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
    cliConfig.addEntry< bool >( "catch-exceptions",
                                "Catch C++ exceptions. Disabling it allows the program to drop into the debugger "
                                "and track the origin of the exception.",
                                true );

    // set execution parameters
    cliConfig.addDelimiter( "Execution parameters" );
    Device::configSetup( cliConfig );
    TNL::MPI::configSetup( cliConfig );

    if( ! TNL::Config::parseCommandLine( argc, argv, cliConfig, cliParams ) )
        return EXIT_FAILURE;

    // FIXME: TNL::Cuda::setup overrides the CUDA device selected by MPI::selectGPU
//    Device::setup( cliParams );
    if( std::is_same_v<Device, TNL::Devices::Host> )
        Device::setup( cliParams );
    if( ! TNL::MPI::setup( cliParams ) )
        return EXIT_FAILURE;

    // get config parameters
    TNL::Config::ParameterContainer parameters;
    TNL::Config::ConfigDescription config;

    // set MHFEM parameters
    TNL::MHFEM::configSetup( config );

    // set model parameters
    Model::configSetup( config );

    if( cliParams.getParameter< bool >( "config-help" ) ) {
        // TODO: re-format the message for the config (drop the program name and "--")
        TNL::Config::printUsage( config, argv[0] );
        return EXIT_SUCCESS;
    }
    if( cliParams.getParameter< bool >( "print-static-configuration" ) ) {
        const int logWidth = cliParams.getParameter< int >( "log-width" );
        TNL::Logger consoleLogger( logWidth, std::cout );
        TNL::MHFEM::writeProlog< Problem >( consoleLogger, false );
        return EXIT_SUCCESS;
    }

    const std::string configPath = cliParams.getParameter< std::string >( "config" );
    try {
        parameters = TNL::Config::parseINIConfigFile( configPath, config );
    }
    catch ( const std::exception& e ) {
        std::cerr << "Failed to parse the configuration file " << configPath << " due to the following error:\n" << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Failed to parse the configuration file " << configPath << " due to an unknown C++ exception." << std::endl;
        throw;
    }

    // --output-directory from the CLI overrides output-directory from the config
    if( cliParams.checkParameter( "output-directory" ) )
        parameters.setParameter< std::string >( "output-directory", cliParams.getParameter< std::string >( "output-directory" ) );
    if( ! parameters.checkParameter("output-directory")) {
        std::cerr << "The output-directory parameter was not found in the config and "
                     "--output-directory was not given on the command line." << std::endl;
        return EXIT_FAILURE;
    }

    if( ! TNL::MHFEM::execute< Problem >( cliParams, parameters ) )
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
