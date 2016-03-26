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

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename Vector >
    __cuda_callable__
    static void update( const MeshType & mesh,
                        const typename Vector::RealType & diffusionCoefficient,
                        Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );
        storage[ 0 ] = 2 * diffusionCoefficient * mesh.getHxInverse();
    }

    template< typename Vector >
    __cuda_callable__
    static typename Vector::RealType get( const int & e,
                                          const int & f,
                                          Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );
        tnlAssert( e < 2 && f < 2, );

        if( e == f )
            return storage[ 0 ];
        return 0.0;
    }

    // optimized version returning diagonal entries
    template< typename Vector >
    __cuda_callable__
    static typename Vector::RealType get( const int & e,
                                          Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );
        tnlAssert( e < 2, );

        return storage[ 0 ];
    }
};

template< typename MeshReal, typename Device, typename MeshIndex >
class MassMatrix< tnlGrid< 2, MeshReal, Device, MeshIndex >, MassLumping::enabled >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename Vector >
    __cuda_callable__
    static void update( const MeshType & mesh,
                        const typename Vector::RealType & diffusionCoefficient,
                        Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );

        // TODO: check this
        // value for vertical faces (e=0, e=1)
        storage[ 0 ] = 2 * diffusionCoefficient * mesh.getHy() * mesh.getHxInverse();
        // value for horizontal faces (e=2, e=3)
        storage[ 1 ] = 2 * diffusionCoefficient * mesh.getHx() * mesh.getHyInverse();
    }

    template< typename Vector >
    __cuda_callable__
    static typename Vector::RealType get( const int & e,
                                          const int & f,
                                          Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );
        tnlAssert( e < 4 && f < 4, );

        // vertical face (e=0, e=1)
        if( e == f && e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        if( e == f )
            return storage[ 1 ];
        // non-diagonal entries
        return 0.0;
    }

    // optimized version returning diagonal entries
    template< typename Vector >
    __cuda_callable__
    static typename Vector::RealType get( const int & e,
                                          Vector & storage )
    {
        tnlAssert( storage.getSize() == size, );
        tnlAssert( e < 4, );

        // vertical face (e=0, e=1)
        if( e < 2 )
            return storage[ 0 ];
        // horizontal face (e=2, e=3)
        return storage[ 1 ];
    }
};

} // namespace mhfem
