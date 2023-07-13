#pragma once

#include <TNL/MHFEM/BaseModel.h>
#include <TNL/MHFEM/UniformCoefficient.h>

#include <TNL/FileName.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/PVTUWriter.h>

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          TNL::MHFEM::MassLumping massLumping >
struct HeatEquationArrayTypes :
    public TNL::MHFEM::DefaultArrayTypes< Mesh, Real, NumberOfEquations, massLumping >
{
    using DeviceType = typename Mesh::DeviceType;

    using N_ijK_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 3 >;
    using u_ijKe_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 4 >;
    using m_iK_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 2 >;
    using D_ijK_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 3 >;
    using w_iKe_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 3 >;
    using a_ijKe_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 4 >;
    using r_ijK_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 3 >;
    using f_iK_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 2 >;

    // disable mobility upwind
    using v_iKe_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 3 >;
    using m_iE_upw_t = TNL::MHFEM::UniformCoefficient< Real, DeviceType, 2 >;
};

template< typename Mesh,
          typename Real,
          TNL::MHFEM::MassLumping massLumping = TNL::MHFEM::MassLumping::enabled >
class HeatEquationModel :
    public TNL::MHFEM::BaseModel< Mesh,
                                  Real,
                                  1,  // number of equations
                                  massLumping,
                                  TNL::MHFEM::AdvectionDiscretization::explicit_upwind,
                                  HeatEquationArrayTypes< Mesh, Real, 1, massLumping >
                                >
{
public:
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
    using GeometricVector = typename MeshType::PointType;

    static constexpr int NumberOfEquations = 1;

    // mobility is constant in this model, so upwind is useless
    static constexpr bool do_mobility_upwind = false;

    template< typename DistributedHostMeshType >
    static std::size_t estimateMemoryDemands( const DistributedHostMeshType & mesh )
    {
        const auto & localMesh = mesh.getLocalMesh();
        const std::size_t cells = localMesh.template getEntitiesCount< typename MeshType::Cell >();
        const std::size_t faces = localMesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();

        // as per BaseModel::estimateMemoryDemands, but without the uniform coefficients
        std::size_t mdd_size =
            // Z_iF
            + NumberOfEquations * faces
            // Z_iK
            + NumberOfEquations * cells
            // N_ijK
            + 1
            // u_ijKe
            + 1
            // m_iK
            + 1
            // D_ijK
            + 1
            // w_iKe
            + 1
            // a_ijKe
            + 1
            // r_ijK
            + 1
            // f_iK
            + 1
            // v_iKe
            + 1
            // m_iE_upw
            + 1
            // b_ijK_storage
            + NumberOfEquations * NumberOfEquations * cells * HeatEquationModel::MassMatrix::size
            // R_ijKe
            + NumberOfEquations * NumberOfEquations * cells * HeatEquationModel::FacesPerCell
            // R_iK
            + NumberOfEquations * cells
        ;

        // Z_ijE_upw
        mdd_size += NumberOfEquations * NumberOfEquations * faces;

        mdd_size *= sizeof(RealType);
        return mdd_size;
    }

    static void
    configSetup( TNL::Config::ConfigDescription & config )
    {
        config.addDelimiter( "Heat equation model parameters" );
        config.addEntry< double >( "diffusivity", "Positive diffusivity coefficient.", 1.0 );
    }

    bool
    init( const TNL::Config::ParameterContainer & parameters )
    {
        diffusivity = parameters.getParameter< double >( "diffusivity" );

        // set coefficients in the NumDwarf scheme
        this->N_ijK.setValue( 1 );
        this->u_ijKe.setValue( 0 );
        this->m_iK.setValue( 1 );
        this->D_ijK.setValue( diffusivity );
        this->w_iKe.setValue( 0 );
        this->a_ijKe.setValue( 0 );
        this->r_ijK.setValue( 0 );
        this->f_iK.setValue( 0 );

        // disable mobility upwind
        this->m_iE_upw.setValue( 1 );

        return true;
    }

    template< typename BoundaryConditions >
    __cuda_callable__
    RealType
    getBoundaryMobility( const MeshType & mesh,
                         const BoundaryConditions & bc,
                         const int i,
                         const IndexType E,
                         const IndexType K,  // index of the neighboring cell
                         const RealType time,
                         const RealType tau ) const
    {
        return 1;
    }

    __cuda_callable__
    void
    updateNonLinearTerms( const MeshType & mesh,
                          const IndexType K,
                          const RealType time )
    {}

    // update coefficients whose projection into the RTN_0(K) space depends on the b_ijK coefficients
    __cuda_callable__
    void
    updateVectorCoefficients( const MeshType & mesh,
                              IndexType K,
                              int i )
    {}

    template< typename DistributedMesh >
    void
    makeSnapshot( const RealType time,
                  const IndexType step,
                  const DistributedMesh & distributedMesh,
                  const std::string & outputPrefix ) const
    {
        if( distributedMesh.getCommunicator() == MPI_COMM_NULL )
            return;

        static_assert( std::is_same< typename DistributedMesh::DeviceType, TNL::Devices::Host >::value,
                       "a host mesh must be passed to makeSnapshot" );
        using LocalMesh = typename DistributedMesh::MeshType;
        const LocalMesh& localMesh = distributedMesh.getLocalMesh();

        using MeshWriter = TNL::Meshes::Writers::VTUWriter< LocalMesh >;
        using TNL::Meshes::VTK::FileFormat;
        const FileFormat format = FileFormat::zlib_compressed;

        // create a .pvtu file (only rank 0 actually writes to the file)
        TNL::FileName mainFileName;
        mainFileName.setFileNameBase( outputPrefix + "data_" );
        mainFileName.setExtension( "pvtu" );
        mainFileName.setIndex( step );
        mainFileName.setDigitsCount( 5 );
        std::ofstream file;
        if( TNL::MPI::GetSize( distributedMesh.getCommunicator() ) > 1 && TNL::MPI::GetRank( distributedMesh.getCommunicator() ) == 0 )
           file.open( mainFileName.getFileName() );
        using PVTU = TNL::Meshes::Writers::PVTUWriter< LocalMesh >;
        PVTU pvtu( file, format );
        pvtu.template writeEntities< MeshType::getMeshDimension() >( distributedMesh );
        pvtu.writeMetadata( step, time );

        // write mesh internals (all four fields are needed to ensure that PVTUReader can read the file)
        if( distributedMesh.template getGlobalIndices< 0 >().getSize() > 0 )
            pvtu.template writePPointData< typename DistributedMesh::GlobalIndexType >( "GlobalIndex" );
        if( distributedMesh.getGhostLevels() > 0 )
           pvtu.template writePPointData< std::uint8_t >( TNL::Meshes::VTK::ghostArrayName() );
        if( distributedMesh.template getGlobalIndices< DistributedMesh::getMeshDimension() >().getSize() > 0 )
            pvtu.template writePCellData< typename DistributedMesh::GlobalIndexType >( "GlobalIndex" );
        if( distributedMesh.getGhostLevels() > 0 )
           pvtu.template writePCellData< std::uint8_t >( TNL::Meshes::VTK::ghostArrayName() );

        // the PointData and CellData from the individual files should be added here
        pvtu.template writePCellData< RealType >( "Z" );

        // create a .vtu file for local data
        std::ofstream subfile;
        if( TNL::MPI::GetSize( distributedMesh.getCommunicator() ) > 1 )
            subfile.open( pvtu.addPiece( mainFileName.getFileName(), distributedMesh.getCommunicator() ) );
        else {
            TNL::FileName fileName;
            fileName.setFileNameBase( outputPrefix + "data_" );
            fileName.setExtension( "vtu" );
            fileName.setIndex( step );
            fileName.setDigitsCount( 5 );
            subfile.open( fileName.getFileName().getString() );
        }

        MeshWriter writer( subfile, format );
        writer.writeMetadata( step, time );
        writer.template writeEntities< MeshType::getMeshDimension() >( localMesh );

        // write mesh internals (all four fields are needed to ensure that PVTUReader can read the file)
        if( distributedMesh.template getGlobalIndices< 0 >().getSize() > 0 )
            writer.writePointData( distributedMesh.template getGlobalIndices< 0 >(), "GlobalIndex" );
        if( distributedMesh.getGhostLevels() > 0 )
            writer.writePointData( distributedMesh.vtkPointGhostTypes(), TNL::Meshes::VTK::ghostArrayName() );
        if( distributedMesh.template getGlobalIndices< DistributedMesh::getMeshDimension() >().getSize() > 0 )
            writer.writeCellData( distributedMesh.template getGlobalIndices< DistributedMesh::getMeshDimension() >(), "GlobalIndex" );
        if( distributedMesh.getGhostLevels() > 0 )
            writer.writeCellData( distributedMesh.vtkCellGhostTypes(), TNL::Meshes::VTK::ghostArrayName() );

        // write scalar fields
        writer.writeCellData( this->Z_iK.getStorageArray(), "Z" );
    }

protected:
    RealType diffusivity = 1;
};
