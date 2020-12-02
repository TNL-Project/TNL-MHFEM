#pragma once

#include "MassMatrix.h"

namespace mhfem
{

template< typename MeshDependentData, bool = MeshDependentData::MassMatrix::is_diagonal >
struct MassLumpingDependentCoefficients;

template< typename MeshDependentData >
struct MassLumpingDependentCoefficients< MeshDependentData, true >
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    R_ijKe( const MeshDependentData & mdd,
            const FaceVectorType & faceIndexes,
            const int i,
            const int j,
            const IndexType K,
            const int e,
            const RealType tau )
    {
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell, "" );

        const IndexType & E = faceIndexes[ e ];
        return mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, j, K, e ) * tau;
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    v_iKE( const MeshDependentData & mdd,
           const FaceVectorType & faceIndexes,
           const int i,
           const IndexType K,
           const IndexType E,
           const int e )
    {
        // split into 2 sums to limit catastrophic truncation
        RealType sum_K = 0.0;
        RealType sum_E = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            const auto b = MassMatrix::b_ijKe( mdd, i, j, K, e );
            sum_K += b * mdd.Z_iK( j, K );
            sum_E += b * mdd.Z_iF( j, E );
        }
        return sum_K - sum_E + mdd.w_iKe( i, K, e );
    }
};

template< typename MeshDependentData >
struct MassLumpingDependentCoefficients< MeshDependentData, false >
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    R_ijKe( const MeshDependentData & mdd,
            const FaceVectorType & faceIndexes,
            const int i,
            const int j,
            const IndexType K,
            const int e,
            const RealType tau )
    {
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell, "" );

        RealType R = 0.0;
        for( int f = 0; f < mdd.FacesPerCell; f++ ) {
            const IndexType & F = faceIndexes[ f ];
            R += mdd.m_iE_upw( i, F ) * MassMatrix::b_ijKef( mdd, i, j, K, f, e ) * tau;
        }
        return R;
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    v_iKE( const MeshDependentData & mdd,
           const FaceVectorType & faceIndexes,
           const int i,
           const IndexType K,
           const IndexType E,
           const int e )
    {
        // split into 2 sums to limit catastrophic truncation
        RealType sum_K = 0.0;
        RealType sum_E = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            sum_K += MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.Z_iK( j, K );
            for( int f = 0; f < mdd.FacesPerCell; f++ ) {
                const IndexType & F = faceIndexes[ f ];
                sum_E += MassMatrix::b_ijKef( mdd, i, j, K, e, f ) * mdd.Z_iF( j, F );
            }
        }
        return sum_K - sum_E + mdd.w_iKe( i, K, e );
    }
};

} // namespace mhfem
