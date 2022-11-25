#pragma once

#include "MassMatrix.h"
#include "BaseModel.h"

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

        RealType R = 0;
        if constexpr( MeshDependentData::MassMatrix::is_diagonal ) {
            const IndexType & E = faceIndexes[ e ];
            R = mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, j, K, e );
        }
        else {
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                const IndexType & F = faceIndexes[ f ];
                R += mdd.m_iE_upw( i, F ) * MassMatrix::b_ijKef( mdd, i, j, K, f, e );
            }
        }

        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_upwind ) {
            const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
            if( vel < 0 )
                R -= 2 * vel;
        }
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_trace ) {
            const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
            R -= vel;
        }

        return R;
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
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell );

        const RealType measure_K = getEntityMeasure( mesh, entity );

        RealType R = 0;
        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            R += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
        }
        R *= measure_K / tau;
        R += measure_K * mdd.f_iK( i, K );
        for( int e = 0; e < MeshDependentData::FacesPerCell; e++ ) {
            const IndexType & E = faceIndexes[ e ];
            R -= mdd.m_iE_upw( i, E ) * mdd.w_iKe( i, K, e );
        }

        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind ) {
            // sum into separate variable to do only one subtraction (avoids catastrophic truncation)
            RealType aux = 0;
            for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ )
                for( int e = 0; e < MeshDependentData::FacesPerCell; e++ ) {
                    const IndexType & E = faceIndexes[ e ];
                    aux += ( mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e ) )
                           * mdd.Z_ijE_upw( i, j, E );
                }
            R -= aux;
        }

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
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell );

        const RealType measure_K = getEntityMeasure( mesh, entity );

        RealType Q = 0;
        for( int e = 0; e < MeshDependentData::FacesPerCell; e++ ) {
            const IndexType & E = faceIndexes[ e ];
            Q += mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, j, K, e ) - mdd.u_ijKe( i, j, K, e );
            if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_upwind ) {
                const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
                if( vel >= 0 )
                    Q += vel;
                else
                    Q -= vel;
            }
        }
        Q += measure_K / tau * mdd.N_ijK( i, j, K ) + measure_K * mdd.r_ijK( i, j, K );
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
        static_assert( FaceVectorType::getSize() == MeshDependentData::FacesPerCell );

        RealType Z = 0;

        for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
                Z += mdd.R_ijKe( i, j, K, f ) * mdd.Z_iF( j, F );
            }
        }

        Z += mdd.R_iK( i, K );

        return Z;
    }

    // for use in BCs without the advective flux, and for balancing on interior faces
    // when the explicit upwind for advection is used
    __cuda_callable__
    static RealType
    A_ijKEF_no_advection( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
    {
        // SQinvR = S * Q^{-1} * R, where Q^{-1} * R is already computed in mdd.R_ijKe
        RealType SQinvR = 0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            RealType S = MassMatrix::b_ijKe( mdd, i, xxx, K, e );
            SQinvR += S * mdd.R_ijKe( xxx, j, K, f );
        }

        RealType T = MassMatrix::b_ijKef( mdd, i, j, K, e, f );

        // A = T - S * Q^{-1} * R
        return T - SQinvR;
    }

    // for use in BCs including the advective flux, and for balancing on interior faces
    // when the implicit upwind for advection is used
    __cuda_callable__
    static RealType
    A_ijKEF_advection( const MeshDependentData & mdd,
                       const int i,
                       const int j,
                       const IndexType K,
                       const IndexType E,
                       const int e,
                       const IndexType F,
                       const int f )
    {
        // SQinvR = S * Q^{-1} * R, where Q^{-1} * R is already computed in mdd.R_ijKe
        RealType SQinvR = 0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            RealType S = mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKe( mdd, i, xxx, K, e );
            if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_upwind ) {
                const RealType vel = mdd.a_ijKe( i, xxx, K, e ) + mdd.u_ijKe( i, xxx, K, e );
                if( vel >= 0 )
                    S += vel;
                else
                    S -= vel;
            }
            SQinvR += S * mdd.R_ijKe( xxx, j, K, f );
        }

        RealType T = mdd.m_iE_upw( i, E ) * MassMatrix::b_ijKef( mdd, i, j, K, e, f );
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_upwind ) {
            if( E == F ) {
                const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
                if( vel < 0 )
                    T -= 2 * vel;
            }
        }
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_trace ) {
            if( E == F ) {
                const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
                T -= vel;
            }
        }

        // A = T - S * Q^{-1} * R
        return T - SQinvR;
    }

    // used for the balancing on the interior faces
    // (checks if the advection term is implicit or explicit)
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
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind )
            return A_ijKEF_no_advection( mdd, i, j, K, E, e, F, f );
        // implicit discretizations: upwind, trace
        return A_ijKEF_advection( mdd, i, j, K, E, e, F, f );
    }

    // for use in BCs without the advective flux, and for balancing on interior faces
    // when the explicit upwind for advection is used
    __cuda_callable__
    static RealType
    RHS_iKE_no_advection( const MeshDependentData & mdd,
                          const int i,
                          const IndexType K,
                          const IndexType E,
                          const int e )
    {
        // start with the matrix H = w
        RealType value = mdd.w_iKe( i, K, e );
        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            // add S * Q^{-1} * G, where Q^{-1} * G is already computed in mdd.R_iK
            value += MeshDependentData::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
        }
        return value;
    }

    // for use in BCs including the advective flux, and for balancing on interior faces
    // when the implicit upwind for advection is used
    __cuda_callable__
    static RealType
    RHS_iKE_advection( const MeshDependentData & mdd,
                       const int i,
                       const IndexType K,
                       const IndexType E,
                       const int e )
    {
        // start with the matrix H = m * w
        RealType value = mdd.m_iE_upw( i, E ) * mdd.w_iKe( i, K, e );

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            // add S * Q^{-1} * G, where Q^{-1} * G is already computed in mdd.R_iK
            RealType S = mdd.m_iE_upw( i, E ) * MeshDependentData::MassMatrix::b_ijKe( mdd, i, j, K, e );
            const RealType vel = mdd.a_ijKe( i, j, K, e ) + mdd.u_ijKe( i, j, K, e );
            if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::implicit_upwind ) {
                if( vel >= 0 )
                    S += vel;
                else
                    S -= vel;
            }
            value += S * mdd.R_iK( j, K );

            // add the explicit advection
            if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind ) {
                value += mdd.Z_ijE_upw( i, j, E ) * vel;
            }
        }

        return value;
    }

    // used for the balancing on the interior faces
    // (checks if the advection term is implicit or explicit)
    __cuda_callable__
    static RealType
    RHS_iKE( const MeshDependentData & mdd,
             const int i,
             const IndexType K,
             const IndexType E,
             const int e )
    {
        if constexpr( MeshDependentData::AdvectionDiscretization == AdvectionDiscretization::explicit_upwind )
            return RHS_iKE_no_advection( mdd, i, K, E, e );
        // implicit discretizations: upwind, trace
        return RHS_iKE_advection( mdd, i, K, E, e );
    }
};

} // namespace mhfem
