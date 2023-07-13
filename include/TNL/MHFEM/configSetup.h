#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/LinearSolverTypeResolver.h>

namespace TNL::MHFEM
{

inline void
configSetup( TNL::Config::ConfigDescription& config,
             std::string sectionPrefix = "MHFEM" )
{
    config.addEntry< std::string >( "output-directory", "Path to the output directory." );

    config.addDelimiter( sectionPrefix + " space discretisation" );
    config.addRequiredEntry< std::string >( "mesh", "Input mesh file path." );
    config.addEntry< std::string >( "mesh-format", "Input mesh file format.", "auto" );
        config.addEntryEnum( "auto" );
        config.addEntryEnum( "vtk" );
        config.addEntryEnum( "vtu" );
        config.addEntryEnum( "ng" );
    config.addEntry< std::string >( "boundary-conditions-file", "Path to the boundary conditions file." );

    config.addDelimiter( sectionPrefix + " time discretisation" );
    config.addEntry< std::string >( "initial-condition", "File name with the initial condition." );
    config.addRequiredEntry< double >( "final-time", "Stop time of the time dependent problem." );
    config.addEntry< double >( "initial-time", "Initial time of the time dependent problem.", 0 );
    config.addRequiredEntry< double >( "snapshot-period", "Time period for writing the problem status.");
    config.addEntry< double >( "time-step", "The time step for the time discretisation.", 1.0 );

    config.addDelimiter( sectionPrefix + " linear system solver" );
    config.addRequiredEntry< std::string >( "linear-solver", "The linear system solver:" );
#if defined( HAVE_GINKGO ) || defined( HAVE_HYPRE )
        config.addEntryEnum( "bicgstab" );
#else
    for( auto o : TNL::Solvers::getLinearSolverOptions() )
        config.addEntryEnum( std::string( o ) );
#endif
    config.addEntry< std::string >( "preconditioner", "The preconditioner for the linear system solver:", "none" );
#if defined( HAVE_GINKGO )
        config.addEntryEnum( "AMGX" );
        config.addEntryEnum( "ILU_ISAI" );
        config.addEntryEnum( "PARILU_ISAI" );
        config.addEntryEnum( "PARILUT_ISAI" );
#elif defined( HAVE_HYPRE )
        config.addEntryEnum( "BoomerAMG" );
#else
    for( auto o : TNL::Solvers::getPreconditionerOptions() )
        config.addEntryEnum( std::string( o ) );
#endif
    TNL::Solvers::IterativeSolver< double, int >::configSetup( config );
    using MatrixType = TNL::Matrices::SparseMatrix< double >;
    TNL::Solvers::Linear::CG< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::BICGStab< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::BICGStabL< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::GMRES< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::TFQMR< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::IDRs< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::SOR< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::Diagonal< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::ILU0< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::ILUT< MatrixType >::configSetup( config );
}

} // namespace TNL::MHFEM
