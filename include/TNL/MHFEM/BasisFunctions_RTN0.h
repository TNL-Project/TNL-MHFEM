#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem {
namespace BasisFunctions {

template< typename CellType >
struct RTN0 {};

template< typename MeshConfig, typename Device, typename CellTopology >
struct RTN0< TNL::Meshes::MeshEntity< MeshConfig, Device, CellTopology > >
{
    static_assert( std::is_same< CellTopology, TNL::Meshes::Topologies::Edge >::value ||
                   std::is_same< CellTopology, TNL::Meshes::Topologies::Triangle >::value ||
                   std::is_same< CellTopology, TNL::Meshes::Topologies::Tetrahedron >::value,
                   "The RTN0 space is not implemented for the requested entity topology yet." );

    using MeshType = TNL::Meshes::Mesh< MeshConfig, Device >;
    using CellType = TNL::Meshes::MeshEntity< MeshConfig, Device, CellTopology >;
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
        static constexpr int d = PointType::getSize();
        const auto cellSize = getEntityMeasure( mesh, entity );
        const CoordinatesType constTerm = coordinates / ( d * cellSize );
        for( typename MeshConfig::LocalIndexType e = 0; e < FacesPerCell< CellType >::value; e++ ) {
            // In case of an edge cell, vertices and faces are the same, but by the general definition,
            // here we need to get the *opposite vertex* of the e-th face.
            const auto v = (MeshType::getMeshDimension() > 1) ? e : 1-e;
            const auto& v_e = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( v ) );
            point += constTerm[ e ] * ( x - v_e.getPoint() );
        }
        return point;
    }
};

// NOTE: this is *not* for a general quadrilateral, we assume a rectangle
template< typename MeshConfig, typename Device >
struct RTN0< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle > >
{
    using MeshType = TNL::Meshes::Mesh< MeshConfig, Device >;
    using CellType = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle >;
    using PointType = typename TNL::Meshes::MeshTraits< MeshConfig >::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        // left bottom vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top vertex
        const PointType v_2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) ).getPoint();

        PointType point( 0.0 );
        const CoordinatesType constTerm = coordinates / getEntityMeasure( mesh, entity );

        point[ 1 ] += constTerm[ 0 ] * ( x.y() - v_2.y() );  // bottom
        point[ 0 ] += constTerm[ 1 ] * ( x.x() - v_0.x() );  // right
        point[ 1 ] += constTerm[ 2 ] * ( x.y() - v_0.y() );  // top
        point[ 0 ] += constTerm[ 3 ] * ( x.x() - v_2.x() );  // left

        return point;
    }
};

// NOTE: this is *not* for a general hexahedron, we assume a voxel
template< typename MeshConfig, typename Device >
struct RTN0< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron > >
{
    using MeshType = TNL::Meshes::Mesh< MeshConfig, Device >;
    using CellType = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron >;
    using PointType = typename TNL::Meshes::MeshTraits< MeshConfig >::PointType;
    using CoordinatesType = TNL::Containers::StaticVector< FacesPerCell< CellType >::value, typename PointType::RealType >;

    __cuda_callable__
    static PointType
    evaluate( const MeshType & mesh,
              const CellType & entity,
              const PointType & x,
              const CoordinatesType & coordinates )
    {
        // left bottom front vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top back vertex
        const PointType v_6 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 6 ) ).getPoint();

        PointType point( 0.0 );
        const CoordinatesType constTerm = coordinates / getEntityMeasure( mesh, entity );

        point[ 2 ] += constTerm[ 0 ] * ( x.z() - v_6.z() );  // bottom
        point[ 1 ] += constTerm[ 1 ] * ( x.y() - v_6.y() );  // front
        point[ 0 ] += constTerm[ 2 ] * ( x.x() - v_0.x() );  // right
        point[ 1 ] += constTerm[ 3 ] * ( x.y() - v_0.y() );  // back
        point[ 0 ] += constTerm[ 4 ] * ( x.x() - v_6.x() );  // left
        point[ 2 ] += constTerm[ 5 ] * ( x.z() - v_0.z() );  // top

        return point;
    }
};

} // namespace BasisFunctions
} // namespace mhfem
