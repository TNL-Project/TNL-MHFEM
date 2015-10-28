#pragma once

#include <mesh/tnlGrid.h>
#include "DifferentialOperator.h"
#include "../mesh_helpers.h"

namespace mhfem
{

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
bindMeshDependentData( MeshDependentDataType* mdd )
{
    this->mdd = mdd;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
__cuda_callable__
typename MeshDependentData::IndexType
DifferentialOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexRow,
                          const CoordinatesType & coordinates ) const
{
    return 7 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
updateLinearSystem( const RealType & time,
                    const RealType & tau,
                    const MeshType & mesh,
                    const IndexType & indexRow,
                    const CoordinatesType & coordinates,
                    Vector & u,
                    Vector & b,
                    Matrix & matrix ) const
{
    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    const IndexType E = mdd->indexDofToFace( indexRow );
    const int i = mdd->indexDofToEqno( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    int numCells = getCellsForFace( mesh, E, cellIndexes );

    tnlAssert( numCells == 2,
               cerr << "assertion numCells == 2 failed" << endl; );

    // indexes of the faces, sorted according the following diagrams
    IndexType faceIndexes[ 8 ];

    if( isVerticalFace( mesh, E ) ) {
        // get face indexes of both cells ordered in this way:
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 4 ], faceIndexes[ 6 ] );
        getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexes[ 2 ], faceIndexes[ 3 ], faceIndexes[ 5 ], faceIndexes[ 7 ] );
        for( int j = 0; j < mdd->n; j++ ) {
            matrixRow.setElement( j * 7 + 0, mdd->getDofIndex( j, faceIndexes[ 0 ] ), getValue( i, j, E, 1, faceIndexes[ 0 ], 0, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 1, mdd->getDofIndex( j, faceIndexes[ 1 ] ), getValue( i, j, E, 1, faceIndexes[ 1 ], 1, cellIndexes[ 1 ] ) +
                                                                                      getValue( i, j, E, 0, faceIndexes[ 2 ], 0, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 2, mdd->getDofIndex( j, faceIndexes[ 3 ] ), getValue( i, j, E, 0, faceIndexes[ 3 ], 1, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 3, mdd->getDofIndex( j, faceIndexes[ 4 ] ), getValue( i, j, E, 1, faceIndexes[ 4 ], 2, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 4, mdd->getDofIndex( j, faceIndexes[ 5 ] ), getValue( i, j, E, 0, faceIndexes[ 5 ], 2, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 5, mdd->getDofIndex( j, faceIndexes[ 6 ] ), getValue( i, j, E, 1, faceIndexes[ 6 ], 3, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 6, mdd->getDofIndex( j, faceIndexes[ 7 ] ), getValue( i, j, E, 0, faceIndexes[ 7 ], 3, cellIndexes[ 0 ] ) );
        }
    }
    else {
        // get face indexes of both cells ordered in this way:
        //      ______
        //      | 7  |
        //     2|   3|
        //      |_6__|
        //      | 5  |
        //     0|   1|
        //      |____|
        //        4
        getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 4 ], faceIndexes[ 5 ] );
        getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexes[ 2 ], faceIndexes[ 3 ], faceIndexes[ 6 ], faceIndexes[ 7 ] );
        for( int j = 0; j < mdd->n; j++ ) {
            matrixRow.setElement( j * 7 + 0, mdd->getDofIndex( j, faceIndexes[ 0 ] ), getValue( i, j, E, 3, faceIndexes[ 0 ], 0, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 1, mdd->getDofIndex( j, faceIndexes[ 1 ] ), getValue( i, j, E, 3, faceIndexes[ 1 ], 1, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 2, mdd->getDofIndex( j, faceIndexes[ 2 ] ), getValue( i, j, E, 2, faceIndexes[ 2 ], 0, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 3, mdd->getDofIndex( j, faceIndexes[ 3 ] ), getValue( i, j, E, 2, faceIndexes[ 3 ], 1, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 4, mdd->getDofIndex( j, faceIndexes[ 4 ] ), getValue( i, j, E, 3, faceIndexes[ 4 ], 2, cellIndexes[ 1 ] ) );
            matrixRow.setElement( j * 7 + 5, mdd->getDofIndex( j, faceIndexes[ 5 ] ), getValue( i, j, E, 3, faceIndexes[ 5 ], 3, cellIndexes[ 1 ] ) +
                                                                                      getValue( i, j, E, 2, faceIndexes[ 6 ], 2, cellIndexes[ 0 ] ) );
            matrixRow.setElement( j * 7 + 6, mdd->getDofIndex( j, faceIndexes[ 7 ] ), getValue( i, j, E, 2, faceIndexes[ 7 ], 3, cellIndexes[ 0 ] ) );
        }
    }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
__cuda_callable__
typename MeshDependentData::RealType
DifferentialOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
getValue( const int & i,
          const int & j,
          const IndexType & E,
          const int & e,
          const IndexType & F,
          const int & f,
          const IndexType & K ) const
{
    RealType value = 0.0;
    for( int xxx = 0; xxx < mdd->n; xxx++ ) {
        value -= mdd->b_ijKe( i, xxx, K, e ) * mdd->R_ijKe( xxx, j, K, f );
        if( xxx == j && E == F )
            value += mdd->b_ijKe( i, xxx, K, e );
    }
    return value;
}

} // namespace mhfem
