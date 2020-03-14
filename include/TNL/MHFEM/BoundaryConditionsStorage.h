#pragma once

#include <vector>

#include <TNL/Object.h>
#include <TNL/File.h>
#include <TNL/Containers/detail/ArrayIO.h>

#include "BoundaryConditionsType.h"

namespace mhfem {

template< typename Real >
struct BoundaryConditionsStorage
    : public TNL::Object
{
    std::size_t dofSize = 0;
    std::vector< BoundaryConditionsType > tags;
    std::vector< Real > values;

    void save( const TNL::String & fileName ) const
    {
        TNL::Object::save( fileName );
    }

    void load( const TNL::String & fileName )
    {
        TNL::Object::load( fileName );
    }

    void save( TNL::File & file ) const
    {
        // check dofSize
        if( tags.size() != dofSize ||
            values.size() != dofSize )
        {
            std::cerr << "Invalid dofSize in BoundaryConditionsStorage: dofSize = " << dofSize << "," << std::endl
                 << "tags.getSize() = " << tags.size() << "," << std::endl
                 << "values.getSize() = " << values.size() << "." << std::endl;
            throw false;
        }

        // save serialization type
        TNL::Object::save( file );

        // save dofSize
        file.save( &dofSize );

        // save tags
        using TagsIO = TNL::Containers::detail::ArrayIO< BoundaryConditionsType, std::size_t, TNL::Allocators::Host<BoundaryConditionsType> >;
        saveObjectType( file, TNL::getType< decltype(tags) >() );
        TagsIO::save( file, tags.data(), tags.size() );

        // save values
        using ValuesIO = TNL::Containers::detail::ArrayIO< Real, std::size_t, TNL::Allocators::Host<Real> >;
        saveObjectType( file, TNL::getType< decltype(values) >() );
        ValuesIO::save( file, values.data(), values.size() );
    }

    void load( TNL::File & file )
    {
        // check serialization type
        TNL::Object::load( file );

        // load dofSize
        file.load( &dofSize );

        // read tags
        using TagsIO = TNL::Containers::detail::ArrayIO< BoundaryConditionsType, std::size_t, TNL::Allocators::Host<BoundaryConditionsType> >;
        const TNL::String type = getObjectType( file );
        if( type != TNL::getType< decltype(tags) >() )
            throw TNL::Exceptions::FileDeserializationError( file.getFileName(), "object type does not match (expected " + TNL::getType< decltype(tags) >() + ", found " + type + ")." );
        tags.resize( dofSize );
        TagsIO::load( file, tags.data(), tags.size() );

        // read values
        using ValuesIO = TNL::Containers::detail::ArrayIO< Real, std::size_t, TNL::Allocators::Host<Real> >;
        const TNL::String valuesType = getObjectType( file );
        if( valuesType != TNL::getType< decltype(values) >() )
            throw TNL::Exceptions::FileDeserializationError( file.getFileName(), "object type does not match (expected " + TNL::getType< decltype(values) >() + ", found " + valuesType + ")." );
        values.resize( dofSize );
        ValuesIO::load( file, values.data(), values.size() );
    }
};

} // namespace mhfem
