#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>

using TNL::Meshes::getEntityCenter;
using TNL::Meshes::getEntityMeasure;


template< typename MeshEntity >
struct FacesPerCell
{
    static constexpr int value = MeshEntity::template SubentityTraits< MeshEntity::MeshType::getMeshDimension() - 1 >::count;
};

template< typename Grid, typename Config >
struct FacesPerCell< TNL::Meshes::GridEntity< Grid, 1, Config > >
{
    static constexpr int value = 2;
};

template< typename Grid, typename Config >
struct FacesPerCell< TNL::Meshes::GridEntity< Grid, 2, Config > >
{
    static constexpr int value = 4;
};

template< typename Grid, typename Config >
struct FacesPerCell< TNL::Meshes::GridEntity< Grid, 3, Config > >
{
    static constexpr int value = 6;
};


template< typename Mesh >
__cuda_callable__
typename Mesh::LocalIndexType
getCellsForFace( const Mesh & mesh, const typename Mesh::GlobalIndexType E, typename Mesh::GlobalIndexType* cellIndexes )
{
    using LocalIndexType = typename Mesh::LocalIndexType;
    const LocalIndexType numCells = mesh.template getSuperentitiesCount< Mesh::getMeshDimension() - 1, Mesh::getMeshDimension() >( E );
    for( LocalIndexType i = 0; i < numCells; i++ )
        cellIndexes[ i ] = mesh.template getSuperentityIndex< Mesh::getMeshDimension() - 1, Mesh::getMeshDimension() >( E, i );
    return numCells;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
int
getCellsForFace( const TNL::Meshes::Grid< 1, RealType, DeviceType, IndexType > & mesh, const IndexType E, IndexType* cellIndexes )
{
    int numCells = 0;   // number of cells adjacent to the face, will be incremented and returned (2 for inner faces, 1 for boundary faces)

    // TODO: check if any code depends on the order and swap it
    if( E < mesh.getDimensions().x() )
        cellIndexes[ numCells++ ] = E;
    if( E > 0 )
        cellIndexes[ numCells++ ] = E - 1;

    return numCells;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
int
getCellsForFace( const TNL::Meshes::Grid< 2, RealType, DeviceType, IndexType > & mesh, const IndexType E, IndexType* cellIndexes )
{
    // TODO: avoid getEntity
    const auto face = mesh.template getEntity< typename TNL::Meshes::Grid< 2, RealType, DeviceType, IndexType >::Face >( E );
    const auto coords = face.getCoordinates();
    const auto orientation = face.getOrientation();
    const auto neighbours = face.template getNeighborEntities< 2 >();

    int numCells = 0;   // number of cells adjacent to the face, will be incremented and returned (2 for inner faces, 1 for boundary faces)

    if( orientation.x() ) {
        // TODO: check if any code depends on the order and swap it
        if( coords.x() < mesh.getDimensions().x() )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 1, 0 >();
        if( coords.x() > 0 )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< -1, 0 >();
    }
    else {
        // TODO: check if any code depends on the order and swap it
        if( coords.y() < mesh.getDimensions().y() )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, 1 >();
        if( coords.y() > 0 )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, -1 >();
    }

    return numCells;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
int
getCellsForFace( const TNL::Meshes::Grid< 3, RealType, DeviceType, IndexType > & mesh, const IndexType E, IndexType* cellIndexes )
{
    // TODO: avoid getEntity
    const auto face = mesh.template getEntity< typename TNL::Meshes::Grid< 3, RealType, DeviceType, IndexType >::Face >( E );
    const auto coords = face.getCoordinates();
    const auto orientation = face.getOrientation();
    const auto neighbours = face.template getNeighborEntities< 3 >();

    int numCells = 0;   // number of cells adjacent to the face, will be incremented and returned (2 for inner faces, 1 for boundary faces)

    if( orientation.x() ) {
        // TODO: check if any code depends on the order and swap it
        if( coords.x() < mesh.getDimensions().x() )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 1, 0, 0 >();
        if( coords.x() > 0 )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< -1, 0, 0 >();
    }
    else if( orientation.y() ) {
        // TODO: check if any code depends on the order and swap it
        if( coords.y() < mesh.getDimensions().y() )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, 1, 0 >();
        if( coords.y() > 0 )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, -1, 0 >();
    }
    else {
        // TODO: check if any code depends on the order and swap it
        if( coords.z() < mesh.getDimensions().z() )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, 0, 1 >();
        if( coords.z() > 0 )
            cellIndexes[ numCells++ ] = neighbours.template getEntityIndex< 0, 0, -1 >();
    }

    return numCells;
}


// TODO: we can optimize either grids or meshes:
//  - meshes: call getSubentityIndex multiple times to avoid creating a static array
//  - grids: create a static array to avoid computing the same thing multiple times
// Maybe the interface of grid or mesh entities should be re-thought?
template< typename Mesh, typename IndexType >
__cuda_callable__
TNL::Containers::StaticArray< FacesPerCell< typename Mesh::Cell >::value, IndexType >
getFacesForCell( const Mesh & mesh, const IndexType & K )
{
    TNL::Containers::StaticArray< FacesPerCell< typename Mesh::Cell >::value, IndexType > faceIndexes;
    for( typename Mesh::LocalIndexType i = 0; i < FacesPerCell< typename Mesh::Cell >::value; i++ )
        faceIndexes[ i ] = mesh.template getSubentityIndex< Mesh::getMeshDimension(), Mesh::getMeshDimension() - 1 >( K, i );
    return faceIndexes;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
TNL::Containers::StaticArray< 2, IndexType >
getFacesForCell( const TNL::Meshes::Grid< 1, RealType, DeviceType, IndexType > & mesh, const IndexType & cell )
{
    using MeshType = TNL::Meshes::Grid< 1, RealType, DeviceType, IndexType >;

    TNL::Containers::StaticArray< 2, IndexType > faceIndexes;
    auto entity = mesh.template getEntity< typename MeshType::Cell >( cell );
    entity.refresh();
    const auto neighbours = entity.template getNeighborEntities< MeshType::getMeshDimension() - 1 >();

    // left
    faceIndexes[ 0 ] = neighbours.template getEntityIndex< -1 >();
    // right
    faceIndexes[ 1 ] = neighbours.template getEntityIndex< 1 >();

    return faceIndexes;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
TNL::Containers::StaticArray< 4, IndexType >
getFacesForCell( const TNL::Meshes::Grid< 2, RealType, DeviceType, IndexType > & mesh, const IndexType & cell )
{
    using MeshType = TNL::Meshes::Grid< 2, RealType, DeviceType, IndexType >;

    TNL::Containers::StaticArray< 4, IndexType > faceIndexes;
    auto entity = mesh.template getEntity< typename MeshType::Cell >( cell );
    entity.refresh();
    const auto neighbours = entity.template getNeighborEntities< MeshType::getMeshDimension() - 1 >();

    // left
    faceIndexes[ 0 ] = neighbours.template getEntityIndex< -1, 0 >();
    // right
    faceIndexes[ 1 ] = neighbours.template getEntityIndex<  1, 0 >();
    // bottom
    faceIndexes[ 2 ] = neighbours.template getEntityIndex< 0, -1 >();
    // top
    faceIndexes[ 3 ] = neighbours.template getEntityIndex< 0,  1 >();

    return faceIndexes;
}

template< typename RealType, typename DeviceType, typename IndexType >
__cuda_callable__
TNL::Containers::StaticArray< 6, IndexType >
getFacesForCell( const TNL::Meshes::Grid< 3, RealType, DeviceType, IndexType > & mesh, const IndexType & cell )
{
    using MeshType = TNL::Meshes::Grid< 3, RealType, DeviceType, IndexType >;

    TNL::Containers::StaticArray< 6, IndexType > faceIndexes;
    auto entity = mesh.template getEntity< typename MeshType::Cell >( cell );
    entity.refresh();
    const auto neighbours = entity.template getNeighborEntities< MeshType::getMeshDimension() - 1 >();

    // left
    faceIndexes[ 0 ] = neighbours.template getEntityIndex< -1, 0, 0 >();
    // right
    faceIndexes[ 1 ] = neighbours.template getEntityIndex<  1, 0, 0 >();
    // bottom
    faceIndexes[ 2 ] = neighbours.template getEntityIndex< 0, -1, 0 >();
    // top
    faceIndexes[ 3 ] = neighbours.template getEntityIndex< 0,  1, 0 >();
    // front
    faceIndexes[ 4 ] = neighbours.template getEntityIndex< 0, 0, -1 >();
    // back
    faceIndexes[ 5 ] = neighbours.template getEntityIndex< 0, 0,  1 >();

    return faceIndexes;
}


template< typename Mesh >
__cuda_callable__
bool
isBoundaryFace( const Mesh & mesh, const typename Mesh::GlobalIndexType E )
{
    return mesh.template isBoundaryEntity< Mesh::getMeshDimension() - 1 >( E );
}

template< int Dim, typename Real, typename Device, typename Index >
__cuda_callable__
bool
isBoundaryFace( const TNL::Meshes::Grid< Dim, Real, Device, Index > & mesh, const Index E )
{
    auto face = mesh.template getEntity< typename TNL::Meshes::Grid< Dim, Real, Device, Index >::Face>( E );
    return face.isBoundaryEntity();
}


template< typename StaticVector, typename Index >
__cuda_callable__
int
getLocalIndex( const StaticVector & vector, const Index & index )
{
    for( int i = 0; i < StaticVector::getSize(); i++ ) {
        if( vector[ i ] == index ) {
            return i;
        }
    }
    TNL_ASSERT( false,
                std::cerr << "local index not found -- this is a BUG!" << std::endl
                          << "vector = " << vector << ", index = " << index << std::endl; );
    return 0;
}


template< typename Mesh >
struct HostMesh;

template< typename Config, typename Device >
struct HostMesh< TNL::Meshes::Mesh< Config, Device > >
{
    using type = TNL::Meshes::Mesh< Config, TNL::Devices::Host >;
};

template< int Dim, typename Real, typename Device, typename Index >
struct HostMesh< TNL::Meshes::Grid< Dim, Real, Device, Index > >
{
    // Grid does not store any arrays on the GPU, so the Device template parameter does not matter for accessing internal attributes
//    using type = TNL::Meshes::Grid< Dim, Real, TNL::Devices::Host, Index >;
    using type = TNL::Meshes::Grid< Dim, Real, Device, Index >;
};



template< typename Grid, typename Config >
__cuda_callable__
typename Grid::RealType
getMinEdgeLength( const Grid & grid,
                  const TNL::Meshes::GridEntity< Grid, 0, Config > & entity )
{
    return 0.0;
}

template< typename Grid, int EntityDimension, typename Config >
__cuda_callable__
typename Grid::RealType
getMinEdgeLength( const Grid & grid,
                  const TNL::Meshes::GridEntity< Grid, EntityDimension, Config > & entity )
{
    return grid.getSmallestSpaceStep();
}

// Vertex
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Vertex > & entity )
{
    return 0;
}

// Edge
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Edge > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    return TNL::l2Norm( v1.getPoint() - v0.getPoint() );
}

// Triangle
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Triangle > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );

    const auto e0 = TNL::l2Norm( v2.getPoint() - v1.getPoint() );
    const auto e1 = TNL::l2Norm( v2.getPoint() - v0.getPoint() );
    const auto e2 = TNL::l2Norm( v1.getPoint() - v0.getPoint() );

    return TNL::min( e0, e1, e2 );
}

// Quadrangle
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );

    const auto e0 = TNL::l2Norm( v1.getPoint() - v0.getPoint() );
    const auto e1 = TNL::l2Norm( v2.getPoint() - v1.getPoint() );
    const auto e2 = TNL::l2Norm( v3.getPoint() - v2.getPoint() );
    const auto e3 = TNL::l2Norm( v0.getPoint() - v3.getPoint() );

    return TNL::min( e0, e1, e2, e3 );
}

// Tetrahedron
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Tetrahedron > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );

    const auto e10 = TNL::l2Norm( v1.getPoint() - v0.getPoint() );
    const auto e20 = TNL::l2Norm( v2.getPoint() - v0.getPoint() );
    const auto e30 = TNL::l2Norm( v3.getPoint() - v0.getPoint() );
    const auto e21 = TNL::l2Norm( v2.getPoint() - v1.getPoint() );
    const auto e32 = TNL::l2Norm( v3.getPoint() - v2.getPoint() );
    const auto e13 = TNL::l2Norm( v1.getPoint() - v3.getPoint() );

    return TNL::min( e10, e20, e30, e21, e32, e13 );
}

// Hexahedron
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getMinEdgeLength( const TNL::Meshes::Mesh< MeshConfig, Device > & mesh,
                  const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron > & entity )
{
    const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );
    const auto& v4 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 4 ) );
    const auto& v5 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 5 ) );
    const auto& v6 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 6 ) );
    const auto& v7 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 7 ) );

    const auto e0 = TNL::l2Norm( v1.getPoint() - v0.getPoint() );
    const auto e1 = TNL::l2Norm( v2.getPoint() - v1.getPoint() );
    const auto e2 = TNL::l2Norm( v3.getPoint() - v2.getPoint() );
    const auto e3 = TNL::l2Norm( v0.getPoint() - v3.getPoint() );
    const auto e4 = TNL::l2Norm( v5.getPoint() - v4.getPoint() );
    const auto e5 = TNL::l2Norm( v6.getPoint() - v5.getPoint() );
    const auto e6 = TNL::l2Norm( v7.getPoint() - v6.getPoint() );
    const auto e7 = TNL::l2Norm( v4.getPoint() - v7.getPoint() );
    const auto e8  = TNL::l2Norm( v4.getPoint() - v0.getPoint() );
    const auto e9  = TNL::l2Norm( v5.getPoint() - v1.getPoint() );
    const auto e10 = TNL::l2Norm( v6.getPoint() - v2.getPoint() );
    const auto e11 = TNL::l2Norm( v7.getPoint() - v3.getPoint() );

    return TNL::min( e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11 );
}
