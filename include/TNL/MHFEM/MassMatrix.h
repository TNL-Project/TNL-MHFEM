#pragma once

#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/MeshEntity.h>

#include "../lib_general/mesh_helpers.h"

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

} // namespace mhfem
