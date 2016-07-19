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

    // face indexes for both cells
    FaceVectorType faceIndexesK0;
    FaceVectorType faceIndexesK1;
    getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexesK0 );
    getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexesK1 );

    if( isVerticalFace( mesh, E ) ) {
        //        K1   K0
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
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
        //      ______
        //      | 7  |
        //     2|   3| K0
        //      |_6__|
        //      | 5  |
        //     0|   1| K1
        //      |____|
        //        4
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


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
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
DifferentialOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexRow,
                          const CoordinatesType & coordinates ) const
{
    return 11 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
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

    // face indexes for both cells
    FaceVectorType faceIndexesK0;
    FaceVectorType faceIndexesK1;
    getFacesForCell( mesh, cellIndexes[ 0 ], faceIndexesK0 );
    getFacesForCell( mesh, cellIndexes[ 1 ], faceIndexesK1 );

    // TODO: write something like isNxFace/isNyFace/isNzFace
    if( E < mesh.template getNumberOfFaces< 1, 0, 0 >() ) {
        //        K1   K0
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                          coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  2, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  3, mdd->getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  4, mdd->getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd->getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  6, mdd->getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  7, mdd->getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  8, mdd->getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd->getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 5 ], 5 ) );
            matrixRow.setElement( j * 11 + 10, mdd->getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 5 ], 5 ) );
        }
    }
    else if( E < mesh.template getNumberOfFaces< 1, 1, 0 >() ) {
        //      ______
        //      | 7  |
        //     2|   3| K0
        //      |_6__|
        //      | 5  |
        //     0|   1| K1
        //      |____|
        //        4
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  2, mdd->getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  3, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  4, mdd->getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd->getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                          coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  6, mdd->getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  7, mdd->getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  8, mdd->getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd->getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 5 ], 5 ) );
            matrixRow.setElement( j * 11 + 10, mdd->getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 5 ], 5 ) );
        }
    }
    else {
        // E is n_z face
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd->getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd->getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  2, mdd->getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  3, mdd->getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  4, mdd->getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd->getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  6, mdd->getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  7, mdd->getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  8, mdd->getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd->getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK0[ 5 ], 5 ) +
                                                                                          coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 + 10, mdd->getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( *mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 5 ], 5 ) );
        }
    }
}

} // namespace mhfem
