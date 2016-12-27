#pragma once

#include <TNL/Meshes/Grid.h>
#include "DifferentialOperator.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
bind( const TNL::SharedPointer< MeshType > & mesh,
      TNL::SharedPointer< MeshDependentDataType > & mdd )
{
    this->mesh = mesh;
    this->mdd = mdd;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
__cuda_callable__
typename MeshDependentData::IndexType
DifferentialOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    return 3 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename DofFunctionPointer, typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >::
setMatrixElements( DofFunctionPointer & u,
                   const typename MeshType::Face & entity,
                   const RealType & time,
                   const RealType & tau,
                   const int & i,
                   Matrix & matrix,
                   Vector & b ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    // dereference the smart pointer on device
    const auto & mesh = this->mesh.template getData< DeviceType >();
    const auto & mdd = this->mdd.template getData< DeviceType >();

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;

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

    for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
        matrixRow.setElement( j * 3 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
        matrixRow.setElement( j * 3 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                   coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
        matrixRow.setElement( j * 3 + 2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
    }
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
bind( const TNL::SharedPointer< MeshType > & mesh,
      TNL::SharedPointer< MeshDependentDataType > & mdd )
{
    this->mesh = mesh;
    this->mdd = mdd;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
__cuda_callable__
typename MeshDependentData::IndexType
DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    return 7 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename DofVectorPointer, typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >::
setMatrixElements( DofVectorPointer & u,
                   const typename MeshType::Face & entity,
                   const RealType & time,
                   const RealType & tau,
                   const int & i,
                   Matrix & matrix,
                   Vector & b ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    // dereference the smart pointer on device
    const auto & mesh = this->mesh.template getData< DeviceType >();
    const auto & mdd = this->mdd.template getData< DeviceType >();

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

    const auto & orientation = entity.getOrientation();

//    if( isVerticalFace( mesh, E ) ) {
    if( orientation.x() ) {
        //        K1   K0
        //      ___________
        //      | 6  |  7 |
        //     0|   1|2   |3
        //      |____|____|
        //        4     5
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 7 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                       coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 3, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 4, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 7 + 6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
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
            matrixRow.setElement( j * 7 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 7 + 3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 7 + 4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                       coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 7 + 6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
        }
    }
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
void
DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
bind( const TNL::SharedPointer< MeshType > & mesh,
      TNL::SharedPointer< MeshDependentDataType > & mdd )
{
    this->mesh = mesh;
    this->mdd = mdd;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
__cuda_callable__
typename MeshDependentData::IndexType
DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexEntity,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );
    return 11 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
    template< typename DofVectorPointer, typename Vector, typename Matrix >
__cuda_callable__
void
DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >::
setMatrixElements( DofVectorPointer & u,
                   const typename MeshType::Face & entity,
                   const RealType & time,
                   const RealType & tau,
                   const int & i,
                   Matrix & matrix,
                   Vector & b ) const
{
    TNL_ASSERT( ! mesh.isBoundaryEntity( entity ), );

    // dereference the smart pointer on device
    const auto & mesh = this->mesh.template getData< DeviceType >();
    const auto & mdd = this->mdd.template getData< DeviceType >();

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, entity, cellIndexes );

    TNL_ASSERT( numCells == 2,
                std::cerr << "assertion numCells == 2 failed" << std::endl; );

    // face indexes for both cells
    const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
    const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

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
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                         coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 5 ], 5 ) );
            matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 5 ], 5 ) );
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
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                         coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 5 ], 5 ) );
            matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 5 ], 5 ) );
        }
    }
    else {
        // E is n_z face
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 1 ], 1 ) );
            matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 2 ], 2 ) );
            matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 3 ], 3 ) );
            matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK0[ 5 ], 5 ) +
                                                                                         coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK1[ 4 ], 4 ) );
            matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 5 ], 5 ) );
        }
    }
}

} // namespace mhfem
