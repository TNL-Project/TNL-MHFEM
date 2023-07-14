#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>
#include <pytnl/tnl_str_conversion.h>

#include "BoundaryConditionsStorage.h"

PYBIND11_MODULE(tnl_mhfem, m)
{
    register_exceptions(m);

    export_BoundaryConditionsStorage< RealType, IndexType >(m);
}
