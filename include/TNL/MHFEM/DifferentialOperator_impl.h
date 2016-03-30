#pragma once

#include <mesh/tnlGrid.h>
#include "DifferentialOperator.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
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
DifferentialOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexRow,
                          const CoordinatesType & coordinates ) const
{
    return 3 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
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

    // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, E, cellIndexes );

    tnlAssert( numCells == 2,
               cerr << "assertion numCells == 2 failed" << endl; );

    // indexes of the faces, sorted according the following diagrams
    FaceVectorType faceIndexesK0;
    FaceVectorType faceIndexesK1;

    // get face indexes of both cells ordered in this way:
    //      0   1|2   3
    //      |____|____|
    getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexesK1 );
    getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexesK0 );
    for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
        matrixRow.setElement( j * 3 + 0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
        matrixRow.setElement( j * 3 + 1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                    coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
        matrixRow.setElement( j * 3 + 2, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
    }
}


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
    const int numCells = getCellsForFace( mesh, E, cellIndexes );

    tnlAssert( numCells == 2,
               cerr << "assertion numCells == 2 failed" << endl; );

    // indexes of the faces, sorted according the following diagrams
    FaceVectorType faceIndexesK0;
    FaceVectorType faceIndexesK1;

    if( isVerticalFace( mesh, E ) ) {
        // get face indexes of both cells ordered in this way:
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexesK1 );
        getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexesK0 );
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 7 + 0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                        coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 2, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 3, mdd->getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 4, mdd->getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 5, mdd->getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 7 + 6, mdd->getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
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
        getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexesK1 );
        getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexesK0 );
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 7 + 0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 2, mdd->getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 3, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 4, mdd->getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 5, mdd->getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                        coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 6, mdd->getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
        }
    }
}

} // namespace mhfem
