#pragma once

#include <TNL/Meshes/Mesh.h>

#include "../libs/spatial/src/idle_point_multimap.hpp"

namespace mhfem {

template< typename MeshEntity, typename MeshConfig, typename PermutationVector >
bool
getSpatialOrdering( const TNL::Meshes::Mesh< MeshConfig, TNL::Devices::Host >& mesh,
                    PermutationVector& perm,
                    PermutationVector& iperm )
{
    static_assert( std::is_same< typename PermutationVector::DeviceType, TNL::Devices::Host >::value, "" );
    using Mesh = TNL::Meshes::Mesh< MeshConfig, TNL::Devices::Host >;
    using IndexType = typename Mesh::GlobalIndexType;
    using PointType = typename Mesh::PointType;

    const IndexType numberOfEntities = mesh.template getEntitiesCount< MeshEntity >();

    // allocate permutation vectors
    if( ! perm.setSize( numberOfEntities ) ||
        ! iperm.setSize( numberOfEntities ) )
        return false;

    spatial::idle_point_multimap< PointType::size, PointType, IndexType > container;

    for( IndexType i = 0; i < numberOfEntities; i++ ) {
        const auto& entity = mesh.template getEntity< MeshEntity >( i );
        const auto center = getEntityCenter( mesh, entity );
        container.insert( std::make_pair( center, i ) );
    }

    container.rebalance();

    IndexType permIndex = 0;

    // in-order traversal of the k-d tree
    for( auto iter = container.cbegin();
         iter != container.cend();
         iter++ )
    {
        perm[ permIndex ] = iter->second;
        iperm[ iter->second ] = permIndex;
        permIndex++;
    }

    return true;
}

template< typename MeshEntity, typename MeshConfig, typename PermutationVector >
bool
getSpatialOrdering( const TNL::Meshes::Mesh< MeshConfig, TNL::Devices::Cuda >& mesh,
                    PermutationVector& perm,
                    PermutationVector& iperm )
{
    static_assert( std::is_same< typename PermutationVector::DeviceType, TNL::Devices::Cuda >::value, "" );
    using MeshHost = TNL::Meshes::Mesh< MeshConfig, TNL::Devices::Host >;
    using MeshCuda = TNL::Meshes::Mesh< MeshConfig, TNL::Devices::Host >;
    using PermutationHost = typename PermutationVector::HostType;

    const MeshHost meshHost = mesh;
    PermutationHost permHost, ipermHost;
    if( ! getSpatialOrdering( meshHost, permHost, ipermHost ) )
        return false;
    if( ! perm.setLike( permHost ) || ! iperm.setLike( ipermHost ) )
        return false;
    perm = permHost;
    iperm = ipermHost;
    return true;
}


// general implementation covering grids
template< typename Mesh >
class MeshOrdering
{
public:
    bool reorder( Mesh& mesh )
    {
        return true;
    }
};

// reordering makes sense only for unstructured meshes
template< typename MeshConfig, typename Device >
class MeshOrdering< TNL::Meshes::Mesh< MeshConfig, Device > >
{
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using IndexType = typename Mesh::GlobalIndexType;

public:
    static bool reorder( Mesh& mesh )
    {
        TNL::Containers::Vector< IndexType, Device, IndexType > perm, iperm;
        if( ! getSpatialOrdering< typename Mesh::Vertex >( mesh, perm, iperm ) ||
            ! mesh.template reorderEntities< 0 >( perm, iperm ) )
            return false;
        if( ! getSpatialOrdering< typename Mesh::Face >( mesh, perm, iperm ) ||
            ! mesh.template reorderEntities< Mesh::getMeshDimension() - 1 >( perm, iperm ) )
            return false;
        if( ! getSpatialOrdering< typename Mesh::Cell >( mesh, perm, iperm ) ||
            ! mesh.template reorderEntities< Mesh::getMeshDimension() >( perm, iperm ) )
            return false;
        return true;
    }
};

} // namespace mhfem
