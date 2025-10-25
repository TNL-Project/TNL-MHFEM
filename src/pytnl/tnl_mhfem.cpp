#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include "BoundaryConditionsStorage.h"

NB_MODULE(tnl_mhfem, m)
{
    register_exceptions(m);

    export_BoundaryConditionsStorage< RealType, IndexType >(m);
}
