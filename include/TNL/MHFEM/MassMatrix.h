#pragma once

#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/MeshEntity.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/StaticMatrix.h"
#include "../lib_general/LU.h"

//#include <armadillo>

// TODO: implement SlicedNDArray, write accessor classes for slices and use them in the update methods below

namespace mhfem {

enum class MassLumping {
    enabled,
    disabled
};

template< typename MeshEntity, MassLumping >
class MassMatrix
{};

// NOTE: everything is only for D isotropic (represented by scalar value)

template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 1, Config >, MassLumping::enabled >
{
public:
    static_assert( Grid::getMeshDimension() == 1, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 1, Config >;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1 >();  // h_x^-1
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        if( e == f )
            return mdd.b_ijK_storage( i, j, K, 0 );
        return 0.0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        return mdd.b_ijK_storage( i, j, K, 0 );
    }
};

template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 1, Config >, MassLumping::disabled >
{
public:
    static_assert( Grid::getMeshDimension() == 1, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 1, Config >;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1 >();  // h_x^-1
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        if( e == f )
            return 2 * mdd.b_ijK_storage( i, j, K, 0 );
        return mdd.b_ijK_storage( i, j, K, 0 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        return 3 * mdd.b_ijK_storage( i, j, K, 0 );
    }
};


template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 2, Config >, MassLumping::enabled >
{
public:
    static_assert( Grid::getMeshDimension() == 2, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 2, Config >;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        // value for vertical faces (e=0, e=1)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1 >();  // h_y / h_x
        // value for horizontal faces (e=2, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1 >();  // h_x / h_y
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        // non-diagonal entries
        if( e != f )
            return 0.0;
        // vertical face (e=0, e=1)
        if( e < 2 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // horizontal face (e=2, e=3)
        return mdd.b_ijK_storage( i, j, K, 1 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        // vertical face (e=0, e=1)
        if( e < 2 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // horizontal face (e=2, e=3)
        return mdd.b_ijK_storage( i, j, K, 1 );
    }
};

template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 2, Config >, MassLumping::disabled >
{
public:
    static_assert( Grid::getMeshDimension() == 2, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 2, Config >;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        // value for vertical faces (e=0, e=1)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1 >();  // h_y / h_x
        // value for horizontal faces (e=2, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1 >();  // h_x / h_y
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        // vertical faces (e,f = 0 or 1)
        if( e < 2 && f < 2 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 0 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 0 );
        }
        // horizontal faces (e,f = 2 or 3)
        if( e >= 2 && f >= 2 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 1 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 1 );
        }
        // non-diagonal blocks
        return 0.0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        // vertical face (e=0, e=1)
        if( e < 2 )
            return 3 * mdd.b_ijK_storage( i, j, K, 0 );
        // horizontal face (e=2, e=3)
        return 3 * mdd.b_ijK_storage( i, j, K, 1 );
    }
};


template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 3, Config >, MassLumping::enabled >
{
public:
    static_assert( Grid::getMeshDimension() == 3, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 3, Config >;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 3;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        // value for n_x faces (e=0, e=1)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1, 1 >();  // h_y * h_z / h_x
        // value for n_y faces (e=2, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1, 1 >();  // h_x * h_z / h_y
        // value for n_z faces (e=4, e=5)
        mdd.b_ijK_storage( i, j, K, 2 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, 1, -1 >();  // h_x * h_y / h_z
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        // non-diagonal entries
        if( e != f )
            return 0.0;
        // n_x face (e=0, e=1)
        if( e < 2 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // n_y face (e=2, e=3)
        if( e < 4 )
            return mdd.b_ijK_storage( i, j, K, 1 );
        // n_z face (e=4, e=5)
        return mdd.b_ijK_storage( i, j, K, 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        // n_x face (e=0, e=1)
        if( e < 2 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // n_y face (e=2, e=3)
        if( e < 4 )
            return mdd.b_ijK_storage( i, j, K, 1 );
        // n_z face (e=4, e=5)
        return mdd.b_ijK_storage( i, j, K, 2 );
    }
};

template< typename Grid, typename Config >
class MassMatrix< TNL::Meshes::GridEntity< Grid, 3, Config >, MassLumping::disabled >
{
public:
    static_assert( Grid::getMeshDimension() == 3, "The MassMatrix is defined only on cell entities." );
    using MeshEntity = TNL::Meshes::GridEntity< Grid, 3, Config >;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 3;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Grid & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const int & i,
            const int & j )
    {
        const auto K = entity.getIndex();
        // value for n_x faces (e=0, e=1)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1, 1 >();  // h_y * h_z / h_x
        // value for n_y faces (e=2, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1, 1 >();  // h_x * h_z / h_y
        // value for n_z faces (e=4, e=5)
        mdd.b_ijK_storage( i, j, K, 2 ) = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, 1, -1 >();  // h_x * h_y / h_z
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const typename MeshDependentData::IndexType & K,
             const int & e,
             const int & f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        // n_x faces (e,f = 0 or 1)
        if( e < 2 && f < 2 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 0 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 0 );
        }
        // n_y faces (e,f = 2 or 3)
        if( e >= 2 && f >= 2 && e < 4 && f < 4 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 1 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 1 );
        }
        // n_z faces (e,f = 4 or 5)
        if( e >= 4 && f >= 4 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 2 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 2 );
        }
        // non-diagonal blocks
        return 0.0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        // n_x face (e=0, e=1)
        if( e < 2 )
            return 3 * mdd.b_ijK_storage( i, j, K, 0 );
        // n_y face (e=2, e=3)
        if( e < 4 )
            return 3 * mdd.b_ijK_storage( i, j, K, 1 );
        // n_z face (e=4, e=5)
        return 3 * mdd.b_ijK_storage( i, j, K, 2 );
    }
};


template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshEdgeTopology >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshEdgeTopology >;
    using LocalIndex = typename MeshEntity::LocalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Mesh & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto K = entity.getIndex();
        const auto h_x = getEntityMeasure( mesh, entity );
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) / h_x;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const typename MeshDependentData::IndexType & K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        if( e == f )
            return 2 * mdd.b_ijK_storage( i, j, K, 0 );
        return mdd.b_ijK_storage( i, j, K, 0 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const typename MeshDependentData::IndexType & K,
            const LocalIndex e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        return 3 * mdd.b_ijK_storage( i, j, K, 0 );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshTriangleTopology >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshTriangleTopology >;
    using LocalIndex = typename MeshEntity::LocalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 9;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Mesh & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );

        const auto P0 = v0.getPoint() - v2.getPoint();
        const auto P1 = v1.getPoint() - v2.getPoint();
        // P2 = 0
        
        const auto P00 = P0 * P0;
        const auto P01 = P0 * P1;
        const auto P11 = P1 * P1;

        const auto K = entity.getIndex();
        const auto denominator = 24 * getEntityMeasure( mesh, entity );


        // most stable version using armadillo/LAPACK
//        arma::mat A( 3, 3 );
//
//        A( 0, 0 ) = ( 3 * P00 + P11 - 3 * P01 ) / denominator;
//        A( 1, 1 ) = ( P00 + 3 * P11 - 3 * P01 ) / denominator;
//        A( 2, 2 ) = ( P00 + P11 + P01 ) / denominator;
//        A( 0, 1 ) = ( 3 * P01 - P00 - P11 ) / denominator;
//        A( 0, 2 ) = ( P11 - P00 - P01 ) / denominator;
//        A( 1, 2 ) = ( P00 - P11 - P01 ) / denominator;
//
//        A( 1, 0 ) = A( 0, 1 );
//        A( 2, 0 ) = A( 0, 2 );
//        A( 2, 1 ) = A( 1, 2 );
//
////        arma::mat a = arma::inv( A );
//        arma::mat a = arma::inv_sympd( A );
//
////        std::cerr << "K = " << K << ": cond(A) = " << arma::cond(A) << ", cond(a) = " << arma::cond(a) << ", cond(Aa) = " << cond(A*a) << std::endl;
//
//        // store the inverse in the packed format (upper triangle, column by column)
//        // see: http://www.netlib.org/lapack/lug/node123.html
//        mdd.b_ijK_storage( i, j, K, 0 ) = a( 0, 0 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 1 ) = a( 0, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 2 ) = a( 1, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 3 ) = a( 0, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 4 ) = a( 1, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 5 ) = a( 2, 2 ) * mdd.D_ijK( i, j, K );
//
//        // the last 3 values are the sums for the b_ijK coefficients
//        mdd.b_ijK_storage( i, j, K, 6 ) = ( a( 0, 0 ) + a( 0, 1 ) + a( 0, 2 ) ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 7 ) = ( a( 0, 1 ) + a( 1, 1 ) + a( 1, 2 ) ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 8 ) = ( a( 0, 2 ) + a( 1, 2 ) + a( 2, 2 ) ) * mdd.D_ijK( i, j, K );



//        // FIXME: our implementation of GE is not numerically stable - maybe we need pivoting
//
//        StaticMatrix< 3, 6, typename Mesh::RealType > matrix;
//
//        matrix.setElementFast( 0, 0,  ( 3 * P00 + P11 - 3 * P01 ) / denominator );
//        matrix.setElementFast( 1, 1,  ( P00 + 3 * P11 - 3 * P01 ) / denominator );
//        matrix.setElementFast( 2, 2,  ( P00 + P11 + P01 ) / denominator );
//        matrix.setElementFast( 0, 1,  ( 3 * P01 - P00 - P11 ) / denominator );
//        matrix.setElementFast( 0, 2,  ( P11 - P00 - P01 ) / denominator );
//        matrix.setElementFast( 1, 2,  ( P00 - P11 - P01 ) / denominator );
//
//        matrix.setElementFast( 1, 0,  matrix.getElementFast( 0, 1 ) );
//        matrix.setElementFast( 2, 0,  matrix.getElementFast( 0, 2 ) );
//        matrix.setElementFast( 2, 1,  matrix.getElementFast( 1, 2 ) );
//
//        // set identity in the right half
//        for( LocalIndex i = 0; i < 3; i++ )
//            for( LocalIndex j = 0; j < 3; j++ )
//                if( i == j )
//                    matrix.setElementFast( i, j + 3, 1.0 );
//                else
//                    matrix.setElementFast( i, j + 3, 0.0 );
//
////        if( K == 300 )
////            std::cout << "matrix before inversion:\n" << matrix;
//        // invert
//        GE( matrix );
////        if( K == 300 )
////            std::cout << "matrix after inversion:\n" << matrix;
//
//        // store the inverse in the packed format (upper triangle, column by column)
//        // see: http://www.netlib.org/lapack/lug/node123.html
//        mdd.b_ijK_storage( i, j, K, 0 ) = matrix.getElementFast( 0, 0 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 1 ) = matrix.getElementFast( 0, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 2 ) = matrix.getElementFast( 1, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 3 ) = matrix.getElementFast( 0, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 4 ) = matrix.getElementFast( 1, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 5 ) = matrix.getElementFast( 2, 2 ) * mdd.D_ijK( i, j, K );
//
//        // the last 3 values are the sums for the b_ijK coefficients
//        mdd.b_ijK_storage( i, j, K, 6 ) = ( matrix.getElementFast( 0, 0 ) + matrix.getElementFast( 0, 1 ) + matrix.getElementFast( 0, 2 ) ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 7 ) = ( matrix.getElementFast( 0, 1 ) + matrix.getElementFast( 1, 1 ) + matrix.getElementFast( 1, 2 ) ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 8 ) = ( matrix.getElementFast( 0, 2 ) + matrix.getElementFast( 1, 2 ) + matrix.getElementFast( 2, 2 ) ) * mdd.D_ijK( i, j, K );



        // LU decomposition is stable
        // TODO: use Cholesky instead

        StaticMatrix< 3, 3, typename Mesh::RealType > matrix;
        TNL::Containers::StaticVector< 3, typename Mesh::RealType > v;

        matrix.setElementFast( 0, 0,  ( 3 * P00 + P11 - 3 * P01 ) / denominator );
        matrix.setElementFast( 1, 1,  ( P00 + 3 * P11 - 3 * P01 ) / denominator );
        matrix.setElementFast( 2, 2,  ( P00 + P11 + P01 ) / denominator );
        matrix.setElementFast( 0, 1,  ( 3 * P01 - P00 - P11 ) / denominator );
        matrix.setElementFast( 0, 2,  ( P11 - P00 - P01 ) / denominator );
        matrix.setElementFast( 1, 2,  ( P00 - P11 - P01 ) / denominator );

        matrix.setElementFast( 1, 0,  matrix.getElementFast( 0, 1 ) );
        matrix.setElementFast( 2, 0,  matrix.getElementFast( 0, 2 ) );
        matrix.setElementFast( 2, 1,  matrix.getElementFast( 1, 2 ) );

        LU_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * mdd.D_ijK( i, j, K );

        // the last 3 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 6 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 );
        mdd.b_ijK_storage( i, j, K, 7 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 );
        mdd.b_ijK_storage( i, j, K, 8 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const typename MeshDependentData::IndexType & K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        return mdd.b_ijK_storage( i, j, K, 6 + e );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshTetrahedronTopology >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::MeshTetrahedronTopology >;
    using LocalIndex = typename MeshEntity::LocalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 14;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Mesh & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
        const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );

        const auto P0 = v0.getPoint() - v3.getPoint();
        const auto P1 = v1.getPoint() - v3.getPoint();
        const auto P2 = v2.getPoint() - v3.getPoint();
        // P3 = 0
        
        const auto P00 = P0 * P0;
        const auto P11 = P1 * P1;
        const auto P22 = P2 * P2;
        const auto P01 = P0 * P1;
        const auto P02 = P0 * P2;
        const auto P12 = P1 * P2;

        const auto K = entity.getIndex();
        const auto denominator = 180 * getEntityMeasure( mesh, entity );

//        // most stable version using armadillo/LAPACK
//        arma::mat A( 4, 4 );
//
//        A( 0, 0 ) = ( 12 * P00 +  2 * P11 +  2 * P22 - 8 * P01 - 8 * P02 + 2 * P12 ) / denominator;
//        A( 1, 1 ) = (  2 * P00 + 12 * P11 +  2 * P22 - 8 * P01 + 2 * P02 - 8 * P12 ) / denominator;
//        A( 2, 2 ) = (  2 * P00 +  2 * P11 + 12 * P22 + 2 * P01 - 8 * P02 - 8 * P12 ) / denominator;
//        A( 3, 3 ) = 2 * ( P00 + P11 + P22 + P01 + P02 + P12 ) / denominator;
//        A( 0, 1 ) = ( - 3 * P00 - 3 * P11 + 2 * P22 + 12 * P01 -  3 * P02 -  3 * P12 ) / denominator;
//        A( 0, 2 ) = ( - 3 * P00 + 2 * P11 - 3 * P22 -  3 * P01 + 12 * P02 -  3 * P12 ) / denominator;
//        A( 1, 2 ) = (   2 * P00 - 3 * P11 - 3 * P22 -  3 * P01 -  3 * P02 + 12 * P12 ) / denominator;
//        A( 0, 3 ) = ( - 3 * P00 + 2 * P11 + 2 * P22 - 3 * P01 - 3 * P02 + 2 * P12 ) / denominator;
//        A( 1, 3 ) = (   2 * P00 - 3 * P11 + 2 * P22 - 3 * P01 + 2 * P02 - 3 * P12 ) / denominator;
//        A( 2, 3 ) = (   2 * P00 + 2 * P11 - 3 * P22 + 2 * P01 - 3 * P02 - 3 * P12 ) / denominator;
//
//        A( 1, 0 ) = A( 0, 1 );
//        A( 2, 0 ) = A( 0, 2 );
//        A( 2, 1 ) = A( 1, 2 );
//        A( 3, 0 ) = A( 0, 3 );
//        A( 3, 1 ) = A( 1, 3 );
//        A( 3, 2 ) = A( 2, 3 );
//
////        arma::mat a = arma::inv( A );
//        arma::mat a = arma::inv_sympd( A );
//
//        // store the inverse in the packed format (upper triangle, column by column)
//        // see: http://www.netlib.org/lapack/lug/node123.html
//        mdd.b_ijK_storage( i, j, K, 0 ) = a( 0, 0 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 1 ) = a( 0, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 2 ) = a( 1, 1 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 3 ) = a( 0, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 4 ) = a( 1, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 5 ) = a( 2, 2 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 6 ) = a( 0, 3 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 7 ) = a( 1, 3 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 8 ) = a( 2, 3 ) * mdd.D_ijK( i, j, K );
//        mdd.b_ijK_storage( i, j, K, 9 ) = a( 3, 3 ) * mdd.D_ijK( i, j, K );

        // LU decomposition is stable
        // TODO: use Cholesky instead

        StaticMatrix< 4, 4, typename Mesh::RealType > matrix;
        TNL::Containers::StaticVector< 4, typename Mesh::RealType > v;

        matrix.setElementFast( 0, 0,  ( 12 * P00 +  2 * P11 +  2 * P22 - 8 * P01 - 8 * P02 + 2 * P12 ) / denominator );
        matrix.setElementFast( 1, 1,  (  2 * P00 + 12 * P11 +  2 * P22 - 8 * P01 + 2 * P02 - 8 * P12 ) / denominator );
        matrix.setElementFast( 2, 2,  (  2 * P00 +  2 * P11 + 12 * P22 + 2 * P01 - 8 * P02 - 8 * P12 ) / denominator );
        matrix.setElementFast( 3, 3,  2 * ( P00 + P11 + P22 + P01 + P02 + P12 ) / denominator );
        matrix.setElementFast( 0, 1,  ( - 3 * P00 - 3 * P11 + 2 * P22 + 12 * P01 -  3 * P02 -  3 * P12 ) / denominator );
        matrix.setElementFast( 0, 2,  ( - 3 * P00 + 2 * P11 - 3 * P22 -  3 * P01 + 12 * P02 -  3 * P12 ) / denominator );
        matrix.setElementFast( 1, 2,  (   2 * P00 - 3 * P11 - 3 * P22 -  3 * P01 -  3 * P02 + 12 * P12 ) / denominator );
        matrix.setElementFast( 0, 3,  ( - 3 * P00 + 2 * P11 + 2 * P22 - 3 * P01 - 3 * P02 + 2 * P12 ) / denominator );
        matrix.setElementFast( 1, 3,  (   2 * P00 - 3 * P11 + 2 * P22 - 3 * P01 + 2 * P02 - 3 * P12 ) / denominator );
        matrix.setElementFast( 2, 3,  (   2 * P00 + 2 * P11 - 3 * P22 + 2 * P01 - 3 * P02 - 3 * P12 ) / denominator );

        matrix.setElementFast( 1, 0,  matrix.getElementFast( 0, 1 ) );
        matrix.setElementFast( 2, 0,  matrix.getElementFast( 0, 2 ) );
        matrix.setElementFast( 2, 1,  matrix.getElementFast( 1, 2 ) );
        matrix.setElementFast( 3, 0,  matrix.getElementFast( 0, 3 ) );
        matrix.setElementFast( 3, 1,  matrix.getElementFast( 1, 3 ) );
        matrix.setElementFast( 3, 2,  matrix.getElementFast( 2, 3 ) );

        LU_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 3 ] = 1.0;
        LU_solve( matrix, v, v );
        mdd.b_ijK_storage( i, j, K, 6 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 7 ) = v[ 1 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 8 ) = v[ 2 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 9 ) = v[ 3 ] * mdd.D_ijK( i, j, K );

        // the last 4 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 10 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 6 );
        mdd.b_ijK_storage( i, j, K, 11 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 7 );
        mdd.b_ijK_storage( i, j, K, 12 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 ) + mdd.b_ijK_storage( i, j, K, 8 );
        mdd.b_ijK_storage( i, j, K, 13 ) = mdd.b_ijK_storage( i, j, K, 6 ) + mdd.b_ijK_storage( i, j, K, 7 ) + mdd.b_ijK_storage( i, j, K, 8 ) + mdd.b_ijK_storage( i, j, K, 9 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const typename MeshDependentData::IndexType & K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value && f < FacesPerCell< MeshEntity >::value, );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static inline typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K,
            const int & e )
    {
        TNL_ASSERT( e < FacesPerCell< MeshEntity >::value, );

        return mdd.b_ijK_storage( i, j, K, 10 + e );
    }
};

} // namespace mhfem
