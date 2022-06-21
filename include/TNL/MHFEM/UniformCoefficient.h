#pragma once

#include <TNL/Containers/Array.h>

template< typename Value, typename Device, int Dimension >
class UniformCoefficient
{
public:
    using StorageArray = TNL::Containers::Array< Value, Device, int >;

    template< typename... Indices >
    void setSizes( Indices... indices )
    {
        static_assert( sizeof...(indices) == Dimension, "invalid number of indices passed to UniformCoefficient" );
        storage.setSize( 1 );
    }

    void setValue( Value value )
    {
        storage.setValue( value );
    }

    template< typename... Indices >
    __cuda_callable__
    const Value& operator()( Indices... indices ) const
    {
        static_assert( sizeof...(indices) == Dimension, "invalid number of indices passed to UniformCoefficient" );
        return storage[ 0 ];
    }

    template< typename... Indices >
    __cuda_callable__
    Value& operator()( Indices... indices )
    {
        static_assert( sizeof...(indices) == Dimension, "invalid number of indices passed to UniformCoefficient" );
        return storage[ 0 ];
    }

    const StorageArray& getStorageArray() const
    {
        return storage;
    }

    StorageArray& getStorageArray()
    {
        return storage;
    }

private:
    StorageArray storage;
};
