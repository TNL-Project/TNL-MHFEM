#pragma once

#include "MassMatrix.h"

namespace mhfem {

template< typename MeshDependentData >
struct SecondaryCoefficients
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;
    using MeshType = typename MeshDependentData::MeshType;

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
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell );

        // split into 2 sums to limit catastrophic truncation
        RealType sum_K = 0;
        RealType sum_E = 0;

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            const auto b = MassMatrix::b_ijKe( mdd, i, j, K, e );
            sum_K += b * mdd.Z_iK( j, K );
            // the sum of b_ijKef * Z_iF over f gets trivial when the mass matrix is diagonal
            if constexpr( MeshDependentData::MassMatrix::is_diagonal ) {
                sum_E += b * mdd.Z_iF( j, E );
            }
            else {
                for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                    const IndexType & F = faceIndexes[ f ];
                    sum_E += MassMatrix::b_ijKef( mdd, i, j, K, e, f ) * mdd.Z_iF( j, F );
                }
            }
        }

        return sum_K - sum_E + mdd.w_iKe( i, K, e );
    }

    __cuda_callable__
    static RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
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
            const int i,
            const int j,
            const IndexType K,
            const int e,
            const RealType tau )
    {
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell );

        if constexpr( MeshDependentData::MassMatrix::is_diagonal ) {
            const IndexType & E = faceIndexes[ e ];
            return mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, j, K, e ) * tau;
        }
        else {
            RealType R = 0;
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                const IndexType & F = faceIndexes[ f ];
                R += mdd.m_iE_upw( i, F ) * MassMatrix::b_ijKef( mdd, i, j, K, f, e );
            }
            return R * tau;
        }
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static RealType
    R_iK( const MeshDependentData & mdd,
          const MeshType & mesh,
          const typename MeshType::Cell & entity,
          const FaceVectorType & faceIndexes,
          const int i,
          const IndexType K,
          const RealType tau )
    {
        RealType R = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            R += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
        }
        R += mdd.f_iK( i, K ) * tau;
        R *= getEntityMeasure( mesh, entity );
        for( int e = 0; e < mdd.FacesPerCell; e++ ) {
            const IndexType & E = faceIndexes[ e ];
            R -= mdd.m_iE_upw( i, E ) * mdd.w_iKe( i, K, e ) * tau;
        }

        // sum into separate variable to do only one subtraction (avoids catastrophic truncation)
        RealType aux = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ )
            for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                const IndexType & E = faceIndexes[ e ];
                aux += ( mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e ) )
                       * mdd.Z_ijE_upw( i, j, E ) * tau;
            }
        R -= aux;

        return R;
    }

    template< typename FaceVectorType >
    __cuda_callable__
    static RealType
    Q_ijK( const MeshDependentData & mdd,
           const MeshType & mesh,
           const typename MeshType::Cell & entity,
           const FaceVectorType & faceIndexes,
           const int i,
           const int j,
           const IndexType K,
           const RealType tau )
    {
        RealType Q = 0.0;
        for( int e = 0; e < mdd.FacesPerCell; e++ ) {
            const IndexType & E = faceIndexes[ e ];
            Q += mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, j, K, e ) - mdd.u_ijKe( i, j, K, e );
        }
        Q *= tau;
        Q += getEntityMeasure( mesh, entity ) * ( mdd.N_ijK( i, j, K ) + mdd.r_ijK( i, j, K ) * tau );
        return Q;
    }

    // expression for Z_iK due to the elimination/hybridization of the complete linear system
    // (to be computed after the solution of the hybridized linear system for Z_iF)
    template< typename FaceVectorType >
    __cuda_callable__
    static RealType
    Z_iK( const MeshDependentData & mdd,
          const FaceVectorType & faceIndexes,
          const int i,
          const IndexType K )
    {
        RealType result = 0.0;

        for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
                result += mdd.R_ijKe( i, j, K, f ) * mdd.Z_iF( j, F );
            }
        }

        result += mdd.R_iK( i, K );

        return result;
    }
};

} // namespace mhfem
