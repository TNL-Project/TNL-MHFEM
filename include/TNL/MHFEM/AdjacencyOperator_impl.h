#pragma once

#include "AdjacencyOperator.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem {

template< typename Mesh,
          int NumberOfEquations >
__cuda_callable__
typename Mesh::IndexType
AdjacencyOperator< Mesh, NumberOfEquations >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    // minus the diagonal
    return ( 2 * FacesPerCell< typename MeshType::CellType >::value - 1 ) * NumberOfEquations - 1;
}

template< typename Mesh,
          int NumberOfEquations >
    template< typename Matrix >
__cuda_callable__
void
AdjacencyOperator< Mesh, NumberOfEquations >::
setMatrixElements( const Mesh & mesh,
                   const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const IndexType indexRow = i * faces + E;

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes are ordered in this way:
    //      0   1|2   3
    //      |____|____|
    //        K1   K0
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

    static constexpr int FacesPerCell = ::FacesPerCell< typename MeshType::CellType >::value;
    using LocalIndexPermutation = TNL::Containers::StaticArray< FacesPerCell, LocalIndex >;

    // For unstructured meshes the face indexes might be unsorted.
    // Therefore we build another permutation array with the correct order.
#ifndef __CUDA_ARCH__
    LocalIndexPermutation localFaceIndexesK0;
    LocalIndexPermutation localFaceIndexesK1;
#else
    // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
    // (we'll need to pass custom launch configuration to the traverser)
    __shared__ LocalIndexPermutation __permutationsK0[ 256 ];
    __shared__ LocalIndexPermutation __permutationsK1[ 256 ];
    LocalIndexPermutation& localFaceIndexesK0 = __permutationsK0[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
    LocalIndexPermutation& localFaceIndexesK1 = __permutationsK1[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
#endif
    for( LocalIndex j = 0; j < FacesPerCell; j++ )
        localFaceIndexesK0[ j ] = localFaceIndexesK1[ j ] = j;
    auto comparatorK0 = [&]( LocalIndex a, LocalIndex b ) {
        return faceIndexesK0[ a ] < faceIndexesK0[ b ];
    };
    auto comparatorK1 = [&]( LocalIndex a, LocalIndex b ) {
        return faceIndexesK1[ a ] < faceIndexesK1[ b ];
    };
    // We assume that the array size is small, so we sort it with bubble sort.
    for( LocalIndex k1 = FacesPerCell - 1; k1 > 0; k1-- )
        for( LocalIndex k2 = 0; k2 < k1; k2++ ) {
            if( ! comparatorK0( localFaceIndexesK0[ k2 ], localFaceIndexesK0[ k2+1 ] ) )
                TNL::swap( localFaceIndexesK0[ k2 ], localFaceIndexesK0[ k2+1 ] );
            if( ! comparatorK1( localFaceIndexesK1[ k2 ], localFaceIndexesK1[ k2+1 ] ) )
                TNL::swap( localFaceIndexesK1[ k2 ], localFaceIndexesK1[ k2+1 ] );
        }

    const LocalIndex e0 = getLocalIndex( faceIndexesK0, E );
    const LocalIndex e1 = getLocalIndex( faceIndexesK1, E );

    LocalIndex rowElements = 0;

    // TODO: this is divergent in principle, but might be improved
    for( int j = 0; j < NumberOfEquations; j++ )
    {
        LocalIndex g0 = 0;
        LocalIndex g1 = 0;

        while( g0 < FacesPerCell && g1 < FacesPerCell ) {
            const LocalIndex f0 = localFaceIndexesK0[ g0 ];
            const LocalIndex f1 = localFaceIndexesK1[ g1 ];
            if( faceIndexesK0[ f0 ] < faceIndexesK1[ f1 ] ) {
                matrixRow.setElement( rowElements++,
                                      j * faces + faceIndexesK0[ f0 ],
                                      true );
                g0++;
            }
            else if( faceIndexesK0[ f0 ] > faceIndexesK1[ f1 ] ) {
                matrixRow.setElement( rowElements++,
                                      j * faces + faceIndexesK1[ f1 ],
                                      true );
                g1++;
            }
            // skip diagonal
        }

        while( g0 < FacesPerCell ) {
            const LocalIndex f0 = localFaceIndexesK0[ g0 ];
            matrixRow.setElement( rowElements++,
                                  j * faces + faceIndexesK0[ f0 ],
                                  true );
            g0++;
        }

        while( g1 < FacesPerCell ) {
            const LocalIndex f1 = localFaceIndexesK1[ g1 ];
            matrixRow.setElement( rowElements++,
                                  j * faces + faceIndexesK1[ f1 ],
                                  true );
            g1++;
        }
    }

    TNL_ASSERT( rowElements == ( 2 * FacesPerCell - 1 ) * NumberOfEquations - 1,
                std::cerr << "rowElements = " << rowElements << ", expected = "
                          << ( 2 * FacesPerCell - 1 ) * NumberOfEquations
                          << std::endl; );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
__cuda_callable__
MeshIndex
AdjacencyOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, NumberOfEquations >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    // minus the diagonal
    return 3 * NumberOfEquations - 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
    template< typename Matrix >
__cuda_callable__
void
AdjacencyOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, NumberOfEquations >::
setMatrixElements( const MeshType & mesh,
                   const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes are ordered in this way:
    //      0   1|2   3
    //      |____|____|
    //        K1   K0
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

    int rowElements = 0;
    auto setElement = [&] ( IndexType columnIndex ) {
        // skip diagonal
        if( columnIndex != indexRow )
            matrixRow.setElement( rowElements++, columnIndex, true );
    };

    for( int j = 0; j < NumberOfEquations; j++ ) {
        setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
        setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
        setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
    }
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
__cuda_callable__
MeshIndex
AdjacencyOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, NumberOfEquations >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    // minus the diagonal
    return 7 * NumberOfEquations - 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
    template< typename Matrix >
__cuda_callable__
void
AdjacencyOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, NumberOfEquations >::
setMatrixElements( const MeshType & mesh,
                   const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

    int rowElements = 0;
    auto setElement = [&] ( IndexType columnIndex ) {
        // skip diagonal
        if( columnIndex != indexRow )
            matrixRow.setElement( rowElements++, columnIndex, true );
    };

    const auto & orientation = entity.getOrientation();
//    if( isVerticalFace( mesh, E ) ) {
    if( orientation.x() ) {
        //        K1   K0
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        for( int j = 0; j < NumberOfEquations; j++ ) {
            setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 3 ] );
        }
    }
    else {
        //      ______
        //      | 7  |
        //     2|   3| K0
        //      |_6__|
        //      | 5  |
        //     0|   1| K1
        //      |____|
        //        4
        for( int j = 0; j < NumberOfEquations; j++ ) {
            setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 3 ] );
        }
    }
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
__cuda_callable__
MeshIndex
AdjacencyOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, NumberOfEquations >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    // minus the diagonal
    return 11 * NumberOfEquations - 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
    template< typename Matrix >
__cuda_callable__
void
AdjacencyOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, NumberOfEquations >::
setMatrixElements( const MeshType & mesh,
                   const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

    int rowElements = 0;
    auto setElement = [&] ( IndexType columnIndex ) {
        // skip diagonal
        if( columnIndex != indexRow )
            matrixRow.setElement( rowElements++, columnIndex, true );
    };

    const auto & orientation = entity.getOrientation();
    // TODO: write something like isNxFace/isNyFace/isNzFace
//    if( E < mesh.template getNumberOfFaces< 1, 0, 0 >() ) {
    if( orientation.x() ) {
        //        K1   K0
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        for( int j = 0; j < NumberOfEquations; j++ ) {
            setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 4 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 4 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 5 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 5 ] );
        }
    }
//    else if( E < mesh.template getNumberOfFaces< 1, 1, 0 >() ) {
    else if( orientation.y() ) {
        //      ______
        //      | 7  |
        //     2|   3| K0
        //      |_6__|
        //      | 5  |
        //     0|   1| K1
        //      |____|
        //        4
        for( int j = 0; j < NumberOfEquations; j++ ) {
            setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 4 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 4 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 5 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 5 ] );
        }
    }
    else {
        // E is n_z face
        for( int j = 0; j < NumberOfEquations; j++ ) {
            setElement( j * numberOfFaces + faceIndexesK1[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 0 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 1 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 2 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 3 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 4 ] );
            setElement( j * numberOfFaces + faceIndexesK1[ 5 ] );
            setElement( j * numberOfFaces + faceIndexesK0[ 5 ] );
        }
    }
}

} // namespace mhfem
