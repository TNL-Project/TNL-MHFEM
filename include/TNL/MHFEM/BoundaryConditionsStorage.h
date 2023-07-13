#pragma once

#include <TNL/Containers/Array.h>

#include "BoundaryConditionsType.h"

namespace TNL::MHFEM {

template< typename Real >
struct BoundaryConditionsStorage
    : public TNL::Object
{
    std::int64_t dofSize = 0;
    TNL::Containers::Array< MHFEM::BoundaryConditionsType, TNL::Devices::Host, std::int64_t > tags;
    TNL::Containers::Array< Real, TNL::Devices::Host, std::int64_t > values, dirichletValues;

    using TNL::Object::save;

    using TNL::Object::load;

    void save( TNL::File & file ) const override
    {
        // save serialization type
        TNL::Object::save( file );

        // save dofSize
        file.save( &dofSize );

        // save vectors
        file << tags << values << dirichletValues;
    }

    void load( TNL::File & file ) override
    {
        // check serialization type
        TNL::Object::load( file );

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
};

} // namespace TNL::MHFEM
