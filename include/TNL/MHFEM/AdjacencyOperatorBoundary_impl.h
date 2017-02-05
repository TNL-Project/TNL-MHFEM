#pragma once

#include "AdjacencyOperatorBoundary.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem {

template< typename Mesh, int NumberOfEquations >
__cuda_callable__
typename Mesh::GlobalIndexType
AdjacencyOperatorBoundary< Mesh, NumberOfEquations >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & E,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
//    TNL_ASSERT( mesh.isBoundaryEntity( entity ), );
    // minus the diagonal
    return FacesPerCell< typename MeshType::Cell >::value * NumberOfEquations - 1;
}

template< typename Mesh, int NumberOfEquations >
    template< typename Matrix >
__cuda_callable__
void
AdjacencyOperatorBoundary< Mesh, NumberOfEquations >::
setMatrixElements( const Mesh & mesh,
                   const typename MeshType::Face & entity,
                   const int & i,
                   Matrix & matrix ) const
{
//    TNL_ASSERT( mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // for boundary faces returns only one valid cell index
    IndexType cellIndexes[ 2 ] = {-1};
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );
    const IndexType & K = cellIndexes[ 0 ];

    TNL_ASSERT( numCells == 1,
                std::cerr << "assertion numCells == 1 failed" << std::endl
                          << "E = " << E << std::endl
                          << "K0 = " << cellIndexes[ 0 ] << std::endl
                          << "K1 = " << cellIndexes[ 1 ] << std::endl; );

    // prepare face indexes
    const auto faceIndexes = getFacesForCell( mesh, K );

    int rowElements = 0;
    auto setElement = [&] ( IndexType columnIndex ) {
        // skip diagonal
        if( columnIndex != indexRow )
            matrixRow.setElement( rowElements++, columnIndex, true );
    };

    // set non-zero elements
    for( int j = 0; j < NumberOfEquations; j++ ) {
        for( int f = 0; f < FacesPerCell< typename MeshType::Cell >::value; f++ ) {
            setElement( j * numberOfFaces + faceIndexes[ f ] );
        }
    }
}

} // namespace mhfem
