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
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    // NOTE: MeshDependentData cannot be const, because tnlSharedVector expects RealType*, does not work with const RealType*
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
        RealType value = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            value -= MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
            // TODO: maybe the condition is useless, if we happen to add 0.0, there is no additional overhead involving global memory read
            if( xxx == j && E == F )
                value += MassMatrix::b_ijKe( mdd, i, xxx, K, e );
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

        const IndexType & E = faceIndexes[ e ];
        return mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.current_tau; // TODO: - u_ijKe
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
        RealType v = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            v += MassMatrix::b_ijKe( mdd, i, j, K, e ) * ( mdd.Z_iK( j, K ) - Z_iF[ mdd.getDofIndex( j, E ) ] );
        }
        v += mdd.w_iKe( i, K, e );
        return v;
    }
};

template< typename MeshDependentData >
class MassMatrixDependentCode< MeshDependentData, MassLumping::disabled >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    // NOTE: MeshDependentData cannot be const, because tnlSharedVector expects RealType*, does not work with const RealType*
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
        RealType value = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            value += MassMatrix::b_ijKef( mdd, i, xxx, K, e, f );
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
            R += mdd.m_upw[ mdd.getDofIndex( i, F ) ] * MassMatrix::b_ijKef( i, j, K, f, e ) * mdd.current_tau; // TODO: - u_ijKe
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
        RealType v = 0.0;
        for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
            v += MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.Z_iK( j, K );
            for( int f = 0; f < mdd.FacesPerCell; f++ ) {
                const IndexType & F = faceIndexes[ f ];
                v -= MassMatrix::b_ijKef( mdd, i, j, K, e, f ) * Z_iF[ mdd.getDofIndex( j, F ) ];
            }
        }
        v += mdd.w_iKe( i, K, e );
        return v;
    }
};

} // namespace mhfem
