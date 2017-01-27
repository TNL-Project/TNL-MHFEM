#pragma once

#include "BoundaryConditions.h"
#include "../lib_general/mesh_helpers.h"
#include "SecondaryCoefficients.h"

namespace mhfem
{

template< typename Mesh, typename MeshDependentData >
struct NeumannMatrixRowSetter
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int & i,
                        const IndexType & K,
                        const IndexType & E,
                        const int & e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;
        using LocalIndex = typename Mesh::LocalIndexType;
        using LocalIndexPermutation = TNL::Containers::StaticArray< FaceIndexes::size, LocalIndex >;

        // For unstructured meshes the face indexes might be unsorted.
        // Therefore we build another permutation array with the correct order.
#ifndef __CUDA_ARCH__
        LocalIndexPermutation localFaceIndexes;
#else
        // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
        // (we'll need to pass custom launch configuration to the traverser)
        __shared__ LocalIndexPermutation __permutations[ 256 ];
        LocalIndexPermutation& localFaceIndexes = __permutations[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
#endif
        for( LocalIndex j = 0; j < FaceIndexes::size; j++ )
            localFaceIndexes[ j ] = j;
        auto comparator = [&]( LocalIndex a, LocalIndex b ) {
            return faceIndexes[ a ] < faceIndexes[ b ];
        };
        // We assume that the array size is small, so we sort it with bubble sort.
        for( LocalIndex k1 = FaceIndexes::size - 1; k1 > 0; k1-- )
            for( LocalIndex k2 = 0; k2 < k1; k2++ )
                if( ! comparator( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] ) )
                    TNL::swap( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] );


        for( LocalIndex j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( LocalIndex g = 0; g < MeshDependentData::FacesPerCell; g++ ) {
                const LocalIndex f = localFaceIndexes[ g ];
                matrixRow.setElement( j * MeshDependentData::FacesPerCell + f,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
struct NeumannMatrixRowSetter< TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, MeshDependentData >
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int & i,
                        const IndexType & K,
                        const IndexType & E,
                        const int & e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                matrixRow.setElement( j * MeshDependentData::FacesPerCell + f,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
};


template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
void
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
bind( const TNL::SharedPointer< MeshType > & mesh,
      TNL::SharedPointer< MeshDependentDataType > & mdd )
{
    this->mesh = mesh;
    this->mdd = mdd;
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & E,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
//    TNL_ASSERT( mesh.isBoundaryEntity( entity ), );
    if( this->isDirichletBoundary( mesh, i, entity ) )
        return 1;
    return MeshDependentDataType::FacesPerCell * MeshDependentDataType::NumberOfEquations;
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
    template< typename DofVectorPointer, typename Vector, typename Matrix >
__cuda_callable__
void
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
setMatrixElements( DofVectorPointer & u,
                   const typename MeshType::Face & entity,
                   const RealType & time,
                   const RealType & tau,
                   const int & i,
                   Matrix & matrix,
                   Vector & b ) const
{
    // dereference the smart pointer on device
    const auto & mesh = this->mesh.template getData< DeviceType >();

//    TNL_ASSERT( mesh.isBoundaryEntity( entity ), );

    const IndexType E = entity.getIndex();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // Dirichlet boundary
    if( isDirichletBoundary( mesh, i, entity ) ) {
        matrixRow.setElement( 0, indexRow, 1.0 );
        b[ indexRow ] = static_cast<const ModelImplementation*>(this)->getDirichletValue( mesh, i, E, time );
    }
    // Neumann boundary
    else {
        // for boundary faces returns only one valid cell index
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, entity, cellIndexes );
        const IndexType & K = cellIndexes[ 0 ];

        TNL_ASSERT( numCells == 1,
                    std::cerr << "assertion numCells == 1 failed" << std::endl
                              << "E = " << E << std::endl
                              << "K0 = " << cellIndexes[ 0 ] << std::endl
                              << "K1 = " << cellIndexes[ 1 ] << std::endl; );

        // prepare face indexes
        const auto faceIndexes = getFacesForCell( mesh, K );
        const int e = getLocalIndex( faceIndexes, E );

        // dereference the smart pointer on device
        const auto & mdd = this->mdd.template getData< DeviceType >();

        // set right hand side value
        RealType bValue = - static_cast<const ModelImplementation*>(this)->getNeumannValue( mesh, i, E, time ) * getEntityMeasure( mesh, entity );

        bValue += mdd.w_iKe( i, K, e );
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            bValue += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
        }
        b[ indexRow ] = bValue;

        // set non-zero elements
        NeumannMatrixRowSetter< MeshType, MeshDependentDataType >::
            setRow( matrixRow,
                    mdd,
                    faceIndexes,
                    i, K, E, e );
    }
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
isNeumannBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const
{
//    if( ! face.isBoundaryEntity() )
//        return false;
    return ! isDirichletBoundary( mesh, i, face );
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
isDirichletBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const
{
//    if( ! face.isBoundaryEntity() )
//        return false;
    const IndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();
    return dirichletTags[ i * faces + face.getIndex() ];
}

} // namespace mhfem
