#pragma once

#include <mesh/tnlGrid.h>
#include <core/tnlAssert.h>

namespace mhfem
{

enum class MassLumping {
    enabled,
    disabled
};

template< typename MeshType, MassLumping >
class MassMatrix
{};

// NOTE: everything is only for D isotropic (represented by scalar value)

template< typename MeshReal, typename Device, typename MeshIndex >
class MassMatrix< tnlGrid< 1, MeshReal, Device, MeshIndex >, MassLumping::enabled >
{
public:
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const MeshType & mesh,
            MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K )
    {
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHxInverse();
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
        tnlAssert( e < 2 && f < 2, );

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
        tnlAssert( e < 2, );

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return storage[ 0 ];
    }
};

template< typename MeshReal, typename Device, typename MeshIndex >
class MassMatrix< tnlGrid< 1, MeshReal, Device, MeshIndex >, MassLumping::disabled >
{
public:
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const MeshType & mesh,
            MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K )
    {
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHxInverse();
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
        tnlAssert( e < 2 && f < 2, );

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
        tnlAssert( e < 2, );

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        return 3 * storage[ 0 ];
    }
};


template< typename MeshReal, typename Device, typename MeshIndex >
class MassMatrix< tnlGrid< 2, MeshReal, Device, MeshIndex >, MassLumping::enabled >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const MeshType & mesh,
            MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K )
    {
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for vertical faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHy() * mesh.getHxInverse();
        // value for horizontal faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHx() * mesh.getHyInverse();
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
        tnlAssert( e < 4 && f < 4, );

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical face (e=0, e=1)
        if( e == f && e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        if( e == f )
            return storage[ 1 ];
        // non-diagonal entries
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
        tnlAssert( e < 4, );

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        return storage[ 1 ];
    }
};

template< typename MeshReal, typename Device, typename MeshIndex >
class MassMatrix< tnlGrid< 2, MeshReal, Device, MeshIndex >, MassLumping::disabled >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    static constexpr MassLumping lumping = MassLumping::enabled;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static inline void
    update( const MeshType & mesh,
            MeshDependentData & mdd,
            const int & i,
            const int & j,
            const typename MeshDependentData::IndexType & K )
    {
        typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // value for vertical faces (e=0, e=1)
        storage[ 0 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHy() * mesh.getHxInverse();
        // value for horizontal faces (e=2, e=3)
        storage[ 1 ] = 2 * mdd.D_ijK( i, j, K ) * mesh.getHx() * mesh.getHyInverse();
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
        tnlAssert( e < 4 && f < 4, );

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
        tnlAssert( e < 4, );

        const typename MeshDependentData::RealType* storage = mdd.b_ijK( i, j, K );
        // vertical face (e=0, e=1)
        if( e < 2 )
            return 3 * storage[ 0 ];
        // horizontal face (e=2, e=3)
        return 3 * storage[ 1 ];
    }
};

} // namespace mhfem
