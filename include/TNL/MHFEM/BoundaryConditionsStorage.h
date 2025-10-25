#pragma once

#include <TNL/Containers/Array.h>

#include "BoundaryConditionsType.h"

namespace TNL::MHFEM {

template< typename Real >
struct BoundaryConditionsStorage
{
    std::int64_t dofSize = 0;
    TNL::Containers::Array< MHFEM::BoundaryConditionsType, TNL::Devices::Host, std::int64_t > tags;
    TNL::Containers::Array< Real, TNL::Devices::Host, std::int64_t > values, dirichletValues;

    static std::string
    getSerializationType()
    {
        return "TNL::MHFEM::BoundaryConditionsStorage< " + TNL::getSerializationType< Real >() + " >";
    }

    void save( TNL::File & file ) const
    {
        // save serialization type
        saveObjectType( file, getSerializationType() );

        // save dofSize
        file.save( &dofSize );

        // save vectors
        file << tags << values << dirichletValues;
    }

    void load( TNL::File & file )
    {
        // check serialization type
        const std::string type = getObjectType( file );
        if( type != getSerializationType() )
            throw Exceptions::FileDeserializationError(
                file.getFileName(), "object type does not match (expected " + getSerializationType() + ", found " + type + ")." );

        // load dofSize
        file.load( &dofSize );

        // read vectors
        file >> tags >> values >> dirichletValues;

        // check dofSize
        if( tags.getSize() != dofSize ||
            values.getSize() != dofSize ||
            dirichletValues.getSize() != dofSize )
        {
            std::cerr << "Invalid dofSize in BoundaryConditionsStorage: dofSize = " << dofSize << "," << std::endl
                 << "tags.getSize() = " << tags.getSize() << "," << std::endl
                 << "values.getSize() = " << values.getSize() << "," << std::endl
                 << "dirichletValues.getSize() = " << dirichletValues.getSize() << "." << std::endl;
            throw false;
        }
    }

    void save( const std::string& filename ) const
    {
        File file( filename, std::ios_base::out );
        save( file );
    }

    void load( const std::string& filename )
    {
        File file( filename, std::ios_base::in );
        save( file );
    }
};

} // namespace TNL::MHFEM
