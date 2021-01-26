#pragma once

#include <TNL/Containers/StaticArray.h>

#include "BoundaryConditions.h"
#include "../lib_general/mesh_helpers.h"
#include "SecondaryCoefficients.h"

namespace mhfem {

template< typename MeshDependentData >
struct AdditionalTerms_AdvectiveOutflow
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;

    __cuda_callable__
    static RealType
    T_ijKef( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
    {
        return 0;
    }
};

template< typename MeshDependentData >
struct AdditionalTerms_FixedFlux
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;

    __cuda_callable__
    static RealType
    T_ijKef( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
    {
        if( e == f )
            // TODO: the effect of u_ij and a_ij in boundary conditions is still very experimental!
            return - mdd.Z_ijE_upw( i, j, E ) * ( mdd.u_ijKe( i, j, K, e ) + mdd.a_ijKe( i, j, K, e ) );
        return 0;
    }
};

template< typename Mesh, typename MeshDependentData, typename AdditionalTerms >
struct RowSetter
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int i,
                        const IndexType K,
                        const IndexType E,
                        const int e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;
        using LocalIndex = typename Mesh::LocalIndexType;
        using LocalIndexPermutation = TNL::Containers::StaticArray< FaceIndexes::getSize(), LocalIndex >;

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
        for( LocalIndex j = 0; j < FaceIndexes::getSize(); j++ )
            localFaceIndexes[ j ] = j;
        auto comparator = [&]( LocalIndex a, LocalIndex b ) {
            return faceIndexes[ a ] < faceIndexes[ b ];
        };
        // We assume that the array size is small, so we sort it with bubble sort.
        for( LocalIndex k1 = FaceIndexes::getSize() - 1; k1 > 0; k1-- )
            for( LocalIndex k2 = 0; k2 < k1; k2++ )
                if( ! comparator( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] ) )
                    TNL::swap( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] );

        for( LocalIndex j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( LocalIndex g = 0; g < MeshDependentData::FacesPerCell; g++ ) {
                const LocalIndex f = localFaceIndexes[ g ];
                // NOTE: the local element index depends on the DOF vector ordering
                matrixRow.setElement( j + MeshDependentData::NumberOfEquations * g,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) + AdditionalTerms::T_ijKef( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename AdditionalTerms >
struct RowSetter< TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, MeshDependentData, AdditionalTerms >
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int i,
                        const IndexType K,
                        const IndexType E,
                        const int e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                // NOTE: the local element index depends on the DOF vector ordering
                matrixRow.setElement( j + MeshDependentData::NumberOfEquations * f,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) + AdditionalTerms::T_ijKef( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
};


template< typename MeshDependentData,
          typename BoundaryModel >
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
init( const BoundaryConditionsStorage< RealType > & storage )
{
    tags = storage.tags;
    values = storage.values;
    dirichletValues = storage.dirichletValues;
}

template< typename MeshDependentData,
          typename BoundaryModel >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< MeshDependentData, BoundaryModel >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType E,
                          const int i ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];
    if( type == BoundaryConditionsType::FixedValue )
        return 1;
    return MeshDependentDataType::FacesPerCell * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshDependentData,
          typename BoundaryModel >
    template< typename Matrix, typename Vector >
__cuda_callable__
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
setMatrixElements( const MeshType & mesh,
                   const MeshDependentDataType & mdd,
                   const IndexType rowIndex,
                   const IndexType E,
                   const int i,
                   const RealType time,
                   const RealType tau,
                   Matrix & matrix,
                   Vector & b ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

    auto matrixRow = matrix.getRow( rowIndex );

    TNL_ASSERT_GE( matrixRow.getSize(), getLinearSystemRowLength( mesh, E, i ), "matrix row is too small" );

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];

    switch( type ) {
        // fixed-value (Dirichlet) boundary condition
        case BoundaryConditionsType::FixedValue:
            matrixRow.setElement( 0, mdd.getDofIndex( i, E ), 1.0 );
            b[ rowIndex ] = getDirichletValue( mesh, i, E, time, tau );
            break;

        // fixed-flux (Neumann) boundary condition
        case BoundaryConditionsType::FixedFlux:
        {
            // for boundary faces returns only one valid cell index
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( mesh, E, cellIndexes );
            const IndexType & K = cellIndexes[ 0 ];

            TNL_ASSERT( numCells == 1,
                        std::cerr << "assertion numCells == 1 failed" << std::endl
                                  << "E = " << E << std::endl
                                  << "K0 = " << cellIndexes[ 0 ] << std::endl
                                  << "K1 = " << cellIndexes[ 1 ] << std::endl; );
            (void) numCells;  // silence unused-variable warning for Release build

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set right hand side value
            const auto& entity = mesh.template getEntity< typename MeshType::Face >( E );
            RealType bValue = - getNeumannValue( mesh, i, E, time, tau ) * getEntityMeasure( mesh, entity );

            bValue += mdd.w_iKe( i, K, e );
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                bValue += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
            }
            b[ rowIndex ] = bValue;

            // set non-zero elements
            RowSetter< MeshType, MeshDependentDataType, AdditionalTerms_FixedFlux< MeshDependentDataType > >::
                setRow( matrixRow,
                        mdd,
                        faceIndexes,
                        i, K, E, e );
            break;
        }

        // advective outflow boundary condition
        case BoundaryConditionsType::AdvectiveOutflow:
        {
            // for boundary faces returns only one valid cell index
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( mesh, E, cellIndexes );
            const IndexType & K = cellIndexes[ 0 ];

            TNL_ASSERT( numCells == 1,
                        std::cerr << "assertion numCells == 1 failed" << std::endl
                                  << "E = " << E << std::endl
                                  << "K0 = " << cellIndexes[ 0 ] << std::endl
                                  << "K1 = " << cellIndexes[ 1 ] << std::endl; );
            (void) numCells;  // silence unused-variable warning for Release build

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set right hand side value
            RealType bValue = 0;

            bValue += mdd.w_iKe( i, K, e );
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                bValue += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
            }
            b[ rowIndex ] = bValue;

            // set non-zero elements
            RowSetter< MeshType, MeshDependentDataType, AdditionalTerms_AdvectiveOutflow< MeshDependentDataType > >::
                setRow( matrixRow,
                        mdd,
                        faceIndexes,
                        i, K, E, e );
            break;
        }

        default:
            TNL_ASSERT_TRUE( false, "unknown boundary condition type was encountered" );
            break;
    }

#ifndef NDEBUG
    // the diagonal element should be positive
    if( matrix.getElement( rowIndex, mdd.getDofIndex( i, E ) ) <= 0 ) {
#ifndef __CUDA_ARCH__
        const auto center = getEntityCenter( mesh, mesh.template getEntity< typename MeshType::Face >( E ) );
        std::cerr << "error BC (type = " << (int) type << "): E = " << E << ", rowIndex = " << rowIndex << ", dofIndex = " << mdd.getDofIndex( i, E )
                  << "\nrow:  " << matrixRow
                  << "\nface center = " << center
                  << std::endl;
#endif
        TNL_ASSERT_TRUE( false, "the diagonal matrix element is not positive" );
    }
#endif
}

} // namespace mhfem
