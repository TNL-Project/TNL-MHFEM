#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/Topologies/MeshEdgeTopology.h>
#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/Topologies/MeshTetrahedronTopology.h>

#include "../lib_general/FacesPerCell.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem {
namespace BasisFunctions {

template< typename CellType >
struct RTN0 {};

template< typename MeshConfig, typename CellTopology >
struct RTN0< TNL::Meshes::MeshEntity< MeshConfig, CellTopology > >
{
    static_assert( std::is_same< CellTopology, TNL::Meshes::MeshEdgeTopology >::value ||
                   std::is_same< CellTopology, TNL::Meshes::MeshTriangleTopology >::value ||
                   std::is_same< CellTopology, TNL::Meshes::MeshTetrahedronTopology >::value,
                   "The RTN0 space is not implemented for the requested entity topology yet." );

    using MeshType = TNL::Meshes::Mesh< MeshConfig >;
    using CellType = TNL::Meshes::MeshEntity< MeshConfig, CellTopology >;
    using PointType = typename TNL::Meshes::MeshTraits< MeshConfig >::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        PointType point( 0.0 );
        static constexpr int d = PointType::size;
        const auto cellSize = getEntityMeasure( mesh, entity );
        for( typename MeshConfig::LocalIndexType e = 0; e < FacesPerCell< CellType >::value; e++ ) {
            const auto& v_e = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( e ) );
            const auto x_minus_v_e = x - v_e.getPoint();
            const auto constTerm = coordinates[ e ] / ( d * cellSize );
            for( int i = 0; i < d; i++ )
                point[ i ] += constTerm * x_minus_v_e[ i ];
        }
        return point;
    }
};

template< typename Grid, typename Config >
struct RTN0< TNL::Meshes::GridEntity< Grid, 1, Config > >
{
    using MeshType = Grid;
    using CellType = TNL::Meshes::GridEntity< Grid, 1, Config >;
    using PointType = typename Grid::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    static_assert( std::is_same< typename Grid::Cell, CellType >::value, "wrong entity" );

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        PointType point( 0.0 );

        const auto h = mesh.template getSpaceStepsProducts< 1 >();
        // left vertex
        const auto v_0 = getEntityCenter( mesh, entity ).x() - 0.5 * h;

        const auto x_minus_v_0 = x.x() - v_0 - h;
        point[ 0 ] += coordinates[ 0 ] / h * x_minus_v_0;
        const auto x_minus_v_1 = x.x() - v_0;
        point[ 0 ] += coordinates[ 1 ] / h * x_minus_v_1;

        return point;
    }
};

template< typename Grid, typename Config >
struct RTN0< TNL::Meshes::GridEntity< Grid, 2, Config > >
{
    using MeshType = Grid;
    using CellType = TNL::Meshes::GridEntity< Grid, 2, Config >;
    using PointType = typename Grid::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    static_assert( std::is_same< typename Grid::Cell, CellType >::value, "wrong entity" );

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        PointType point( 0.0 );

        const auto h_x = mesh.template getSpaceStepsProducts< 1, 0 >();
        const auto h_y = mesh.template getSpaceStepsProducts< 0, 1 >();
        // left bottom vertex
        const PointType v_0 = getEntityCenter( mesh, entity ) - PointType( 0.5 * h_x, 0.5 * h_y );

        CoordinatesType constTerm = coordinates * ( 1.0 / ( h_x * h_y ) );

        const auto x_minus_v_0 = x.x() - v_0.x() - h_x;
        point[ 0 ] += constTerm[ 0 ] * x_minus_v_0;
        const auto x_minus_v_1 = x.x() - v_0.x();
        point[ 0 ] += constTerm[ 1 ] * x_minus_v_1;

        const auto y_minus_v_0 = x.y() - v_0.y() - h_y;
        point[ 1 ] += constTerm[ 2 ] * y_minus_v_0;
        const auto y_minus_v_3 = x.y() - v_0.y();
        point[ 1 ] += constTerm[ 3 ] * y_minus_v_3;

        return point;
    }
};

template< typename Grid, typename Config >
struct RTN0< TNL::Meshes::GridEntity< Grid, 3, Config > >
{
    using MeshType = Grid;
    using CellType = TNL::Meshes::GridEntity< Grid, 3, Config >;
    using PointType = typename Grid::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    static_assert( std::is_same< typename Grid::Cell, CellType >::value, "wrong entity" );

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        PointType point( 0.0 );

        const auto h_x = mesh.template getSpaceStepsProducts< 1, 0, 0 >();
        const auto h_y = mesh.template getSpaceStepsProducts< 0, 1, 0 >();
        const auto h_z = mesh.template getSpaceStepsProducts< 0, 0, 1 >();
        // left bottom vertex
        const PointType v_0 = getEntityCenter( mesh, entity ) - PointType( 0.5 * h_x, 0.5 * h_y, 0.5 * h_z );

        CoordinatesType constTerm = coordinates * ( 1.0 / ( h_x * h_y * h_z ) );

        const auto x_minus_v_0 = x.x() - v_0.x() - h_x;
        point[ 0 ] += constTerm[ 0 ] * x_minus_v_0;
        const auto x_minus_v_1 = x.x() - v_0.x();
        point[ 0 ] += constTerm[ 1 ] * x_minus_v_1;

        const auto y_minus_v_0 = x.y() - v_0.y() - h_y;
        point[ 1 ] += constTerm[ 2 ] * y_minus_v_0;
        const auto y_minus_v_3 = x.y() - v_0.y();
        point[ 1 ] += constTerm[ 3 ] * y_minus_v_3;

        const auto z_minus_v_0 = x.z() - v_0.z() - h_z;
        point[ 2 ] += constTerm[ 4 ] * z_minus_v_0;
        const auto z_minus_v_4 = x.z() - v_0.z();
        point[ 2 ] += constTerm[ 5 ] * z_minus_v_4;

        return point;
    }
};

} // namespace BasisFunctions
} // namespace mhfem
