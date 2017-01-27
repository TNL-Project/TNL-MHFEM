#pragma once

#include "MassMatrix.h"

namespace mhfem
{

template< typename MeshDependentData, MassLumping = MeshDependentData::MassMatrix::lumping >
class MassMatrixDependentCode
{};

template< typename MeshDependentData >
class MassMatrixDependentCode< MeshDependentData, MassLumping::enabled >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    // TODO: refactoring, A_ijKEF does not depend on mass-lumping
    __cuda_callable__
    static inline RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const IndexType & K,
             const IndexType & E,
             const int & e,
             const IndexType & F,
             const int & f )
    {
//        RealType value = MassMatrix::b_ijKef( mdd, i, j, K, e, f );
//        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
//            value -= MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
//        }
//        return value;
        // more careful version with only one subtraction to avoid catastrophic truncation
        RealType sum = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            sum += MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
        }
        return MassMatrix::b_ijKef( mdd, i, j, K, e, f ) - sum;
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    R_ijKe( const MeshDependentData & mdd,
            const FaceVectorType & faceIndexes,
            const int & i,
            const int & j,
            const IndexType & K,
            const int & e )
    {
        static_assert( FaceVectorType::size == MeshDependentData::FacesPerCell, "" );

        const IndexType & E = faceIndexes[ e ];
        return mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.current_tau;
    }

    template< typename DofVectorType, typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    v_iKE( const MeshDependentData & mdd,
           const DofVectorType & Z_iF,
           const FaceVectorType & faceIndexes,
           const int & i,
           const IndexType & K,
           const IndexType & E,
           const int & e )
    {
        // split into 2 sums to limit catastrophic truncation
        RealType sum_K = 0.0;
        RealType sum_E = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            const auto b = MassMatrix::b_ijKe( mdd, i, j, K, e );
            sum_K += b * mdd.Z_iK( j, K );
            sum_E += b * Z_iF[ mdd.getDofIndex( j, E ) ];
        }
        return sum_K - sum_E + mdd.w_iKe( i, K, e );
    }
};

template< typename MeshDependentData >
class MassMatrixDependentCode< MeshDependentData, MassLumping::disabled >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    // TODO: refactoring, A_ijKEF does not depend on mass-lumping
    __cuda_callable__
    static inline RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const IndexType & K,
             const IndexType & E,
             const int & e,
             const IndexType & F,
             const int & f )
    {
        RealType value = MassMatrix::b_ijKef( mdd, i, j, K, e, f );
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            value -= MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
        }
        return value;
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    R_ijKe( const MeshDependentData & mdd,
            const FaceVectorType & faceIndexes,
            const int & i,
            const int & j,
            const IndexType & K,
            const int & e )
    {
        static_assert( FaceVectorType::size == MeshDependentData::FacesPerCell, "" );

        RealType R = 0.0;
        for( int f = 0; f < mdd.FacesPerCell; f++ ) {
            const IndexType & F = faceIndexes[ f ];
            R += mdd.m_upw[ mdd.getDofIndex( i, F ) ] * MassMatrix::b_ijKef( mdd, i, j, K, f, e ) * mdd.current_tau;
        }
        return R;
    }

    template< typename DofVectorType, typename FaceVectorType >
    __cuda_callable__
    static inline RealType
    v_iKE( const MeshDependentData & mdd,
           const DofVectorType & Z_iF,
           const FaceVectorType & faceIndexes,
           const int & i,
           const IndexType & K,
           const IndexType & E,
           const int & e )
    {
        // split into 2 sums to limit catastrophic truncation
        RealType sum_K = 0.0;
        RealType sum_E = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            sum_K += MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.Z_iK( j, K );
            for( int f = 0; f < mdd.FacesPerCell; f++ ) {
                const IndexType & F = faceIndexes[ f ];
                sum_E = MassMatrix::b_ijKef( mdd, i, j, K, e, f ) * Z_iF[ mdd.getDofIndex( j, F ) ];
            }
        }
        return sum_K - sum_E + mdd.w_iKe( i, K, e );
    }
};

} // namespace mhfem
