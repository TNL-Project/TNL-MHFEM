#pragma once

#include "AdjacencyOperator.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem {

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
setMatrixElements( const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const MeshType & mesh = entity.getMesh();
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
    auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

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
setMatrixElements( const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const MeshType & mesh = entity.getMesh();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

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
setMatrixElements( const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const MeshType & mesh = entity.getMesh();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

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
