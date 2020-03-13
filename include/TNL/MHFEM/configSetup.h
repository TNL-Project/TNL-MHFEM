#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/LinearSolverTypeResolver.h>

namespace mhfem
{

inline void
configSetup( TNL::Config::ConfigDescription& config,
             std::string sectionPrefix = "MHFEM" )
{
    config.addEntry< TNL::String >( "output-directory", "Path to the output directory." );

    config.addDelimiter( sectionPrefix + " space discretisation" );
    config.addEntry< TNL::String >( "mesh", "A file which contains the numerical mesh. You may create it with tools like tnl-grid-setup or tnl-mesh-convert.", "mesh.tnl" );
    config.addEntry< bool >( "reorder-mesh", "Whether the mesh entities should be reordered.", true );
    config.addEntry< TNL::String >( "boundary-conditions-file", "Path to the boundary conditions file." );

    config.addDelimiter( sectionPrefix + " time discretisation" );
    config.addEntry< TNL::String >( "initial-condition", "File name with the initial condition." );
    config.addRequiredEntry< double >( "final-time", "Stop time of the time dependent problem." );
    config.addEntry< double >( "initial-time", "Initial time of the time dependent problem.", 0 );
    config.addRequiredEntry< double >( "snapshot-period", "Time period for writing the problem status.");
    config.addEntry< double >( "time-step", "The time step for the time discretisation.", 1.0 );

    config.addDelimiter( sectionPrefix + " linear system solver" );
    config.addRequiredEntry< TNL::String >( "linear-solver", "The linear system solver:" );
    for( auto o : TNL::Solvers::getLinearSolverOptions() )
        config.addEntryEnum( TNL::String( o ) );
    config.addEntry< TNL::String >( "preconditioner", "The preconditioner for the linear system solver:", "none" );
    for( auto o : TNL::Solvers::getPreconditionerOptions() )
        config.addEntryEnum( TNL::String( o ) );
    TNL::Solvers::IterativeSolver< double, int >::configSetup( config );
    using MatrixType = TNL::Matrices::Legacy::CSR< double, TNL::Devices::Host, int >;
    TNL::Solvers::Linear::CG< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::BICGStab< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::BICGStabL< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::GMRES< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::TFQMR< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::SOR< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::Diagonal< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::ILU0< MatrixType >::configSetup( config );
    TNL::Solvers::Linear::Preconditioners::ILUT< MatrixType >::configSetup( config );
}

} // namespace mhfem
