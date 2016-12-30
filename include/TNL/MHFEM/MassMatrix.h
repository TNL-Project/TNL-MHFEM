#pragma once

#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/MeshEntity.h>

#include "../lib_general/mesh_helpers.h"
#include "../lib_general/StaticMatrix.h"
#include "../lib_general/GE.h"

namespace mhfem
{

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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1 >();  // h_x^-1
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        if( e == f )
            return storage[ 0 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return storage[ 0 ];
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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1 >();  // h_x^-1
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        if( e == f )
            return 2 * storage[ 0 ];
        return storage[ 0 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return 3 * storage[ 0 ];
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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for vertical faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1 >();  // h_y / h_x
        // value for horizontal faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1 >();  // h_x / h_y
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // non-diagonal entries
        if( e != f )
            return 0.0;
        // vertical face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        return storage[ 1 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        return storage[ 1 ];
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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for vertical faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1 >();  // h_y / h_x
        // value for horizontal faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1 >();  // h_x / h_y
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical faces (e,f = 0 or 1)
        if( e < 2 && f < 2 ) {
            if( e == f )
                // diagonal
                return 2 * storage[ 0 ];
            // non-diagonal
            return storage[ 0 ];
        }
        // horizontal faces (e,f = 2 or 3)
        if( e >= 2 && f >= 2 ) {
            if( e == f )
                // diagonal
                return 2 * storage[ 1 ];
            // non-diagonal
            return storage[ 1 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical face (e=0, e=1)
        if( e < 2 )
            return 3 * storage[ 0 ];
        // horizontal face (e=2, e=3)
        return 3 * storage[ 1 ];
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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for n_x faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1, 1 >();  // h_y * h_z / h_x
        // value for n_y faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1, 1 >();  // h_x * h_z / h_y
        // value for n_z faces (e=4, e=5)
        storage[ 2 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, 1, -1 >();  // h_x * h_y / h_z
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // non-diagonal entries
        if( e != f )
            return 0.0;
        // n_x face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // n_y face (e=2, e=3)
        if( e < 4 )
            return storage[ 1 ];
        // n_z face (e=4, e=5)
        return storage[ 2 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // n_x face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // n_y face (e=2, e=3)
        if( e < 4 )
            return storage[ 1 ];
        // n_z face (e=4, e=5)
        return storage[ 2 ];
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
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for n_x faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< -1, 1, 1 >();  // h_y * h_z / h_x
        // value for n_y faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, -1, 1 >();  // h_x * h_z / h_y
        // value for n_z faces (e=4, e=5)
        storage[ 2 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.template getSpaceStepsProducts< 1, 1, -1 >();  // h_x * h_y / h_z
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // n_x faces (e,f = 0 or 1)
        if( e < 2 && f < 2 ) {
            if( e == f )
                // diagonal
                return 2 * storage[ 0 ];
            // non-diagonal
            return storage[ 0 ];
        }
        // n_y faces (e,f = 2 or 3)
        if( e >= 2 && f >= 2 && e < 4 && f < 4 ) {
            if( e == f )
                // diagonal
                return 2 * storage[ 1 ];
            // non-diagonal
            return storage[ 1 ];
        }
        // n_z faces (e,f = 4 or 5)
        if( e >= 4 && f >= 4 ) {
            if( e == f )
                // diagonal
                return 2 * storage[ 2 ];
            // non-diagonal
            return storage[ 2 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // n_x face (e=0, e=1)
        if( e < 2 )
            return 3 * storage[ 0 ];
        // n_y face (e=2, e=3)
        if( e < 4 )
            return 3 * storage[ 1 ];
        // n_z face (e=4, e=5)
        return 3 * storage[ 2 ];
    }
};


template< typename MeshConfig >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, TNL::Meshes::MeshEdgeTopology >, MassLumping::disabled >
{
public:
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, TNL::Meshes::MeshEdgeTopology >;
    using LocalIndex = typename MeshEntity::LocalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename Mesh, typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Mesh & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j )
    {
        static_assert( std::is_same< typename Mesh::Config, MeshConfig >::value, "wrong mesh" );

        const auto K = entity.getIndex();
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        const auto h_x = getEntityMeasure( mesh, entity );
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) / h_x;
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        if( e == f )
            return 2 * storage[ 0 ];
        return storage[ 0 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return 3 * storage[ 0 ];
    }
};

template< typename MeshConfig >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, TNL::Meshes::MeshTriangleTopology >, MassLumping::disabled >
{
public:
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, TNL::Meshes::MeshTriangleTopology >;
    using LocalIndex = typename MeshEntity::LocalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;

    // number of independent values defining the matrix
    static constexpr int size = 9;

    template< typename Mesh, typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const Mesh & mesh,
            const MeshEntity & entity,
            MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j )
    {
        static_assert( std::is_same< typename Mesh::Config, MeshConfig >::value, "wrong mesh" );

        StaticMatrix< 3, 6, typename Mesh::RealType > matrix;

        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );

        const auto P0 = v0.getPoint() - v2.getPoint();
        const auto P1 = v1.getPoint() - v2.getPoint();
        // P2 = 0
        
        const auto P00 = P0 * P0;
        const auto P01 = P0 * P1;
        const auto P11 = P1 * P1;

        matrix.setElementFast( 0, 0,  3 * P00 + P11 - 3 * P01 );
        matrix.setElementFast( 1, 1,  P00 + 3 * P11 - 3 * P01 );
        matrix.setElementFast( 2, 2,  P00 + P11 + P01 );
        matrix.setElementFast( 0, 1,  3 * P01 - P00 - P11 );
        matrix.setElementFast( 0, 2,  P11 - P00 - P01 );
        matrix.setElementFast( 1, 2,  P00 - P11 - P01 );

        matrix.setElementFast( 1, 0,  matrix.getElementFast( 0, 1 ) );
        matrix.setElementFast( 2, 0,  matrix.getElementFast( 0, 2 ) );
        matrix.setElementFast( 2, 1,  matrix.getElementFast( 1, 2 ) );

        // set identity in the right half
        for( LocalIndex i = 0; i < 3; i++ )
            for( LocalIndex j = 0; j < 3; j++ )
                if( i == j )
                    matrix.setElementFast( i, j + 3, 1.0 );
                else
                    matrix.setElementFast( i, j + 3, 0.0 );

        // invert
        GE( matrix );

        const auto K = entity.getIndex();
        const auto constantFactor = 24 * mdd.D_ijK( i, j, K ) * getEntityMeasure( mesh, entity );

        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html
        storage[ 0 ] = matrix.getElementFast( 0, 0 ) * constantFactor;
        storage[ 1 ] = matrix.getElementFast( 0, 1 ) * constantFactor;
        storage[ 2 ] = matrix.getElementFast( 1, 1 ) * constantFactor;
        storage[ 3 ] = matrix.getElementFast( 0, 2 ) * constantFactor;
        storage[ 4 ] = matrix.getElementFast( 1, 2 ) * constantFactor;
        storage[ 5 ] = matrix.getElementFast( 2, 2 ) * constantFactor;

        // the last 3 values are the sums for the b_ijK coefficients
        storage[ 6 ] = ( matrix.getElementFast( 0, 0 ) + matrix.getElementFast( 0, 1 ) + matrix.getElementFast( 0, 2 ) ) * constantFactor;
        storage[ 7 ] = ( matrix.getElementFast( 0, 1 ) + matrix.getElementFast( 1, 1 ) + matrix.getElementFast( 1, 2 ) ) * constantFactor;
        storage[ 8 ] = ( matrix.getElementFast( 0, 2 ) + matrix.getElementFast( 1, 2 ) + matrix.getElementFast( 2, 2 ) ) * constantFactor;
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return storage[ e + ( f * (f+1) ) / 2 ];
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

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return storage[ 6 + e ];
    }
};

} // namespace mhfem
