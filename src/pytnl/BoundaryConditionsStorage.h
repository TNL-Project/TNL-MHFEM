#pragma once

#include <pytnl/pytnl.h>
#include <pytnl/containers/Array.h>

#include <TNL/MHFEM/BoundaryConditionsStorage.h>

// operator needed for the bindings for Array<BoundaryConditionsType>
namespace TNL::MHFEM {
    inline std::ostream& operator<<( std::ostream& str, BoundaryConditionsType type )
    {
        return str << (int) type;
    }
}

template< typename RealType, typename IndexType >
void export_BoundaryConditionsStorage( nb::module_ & m )
{
    nb::enum_< TNL::MHFEM::BoundaryConditionsType >(m, "BoundaryConditionsType")
        .value("FixedValue", TNL::MHFEM::BoundaryConditionsType::FixedValue)
        .value("FixedFlux", TNL::MHFEM::BoundaryConditionsType::FixedFlux)
        .value("FixedFluxNoAdvection", TNL::MHFEM::BoundaryConditionsType::FixedFluxNoAdvection)
        .value("AdvectiveOutflow", TNL::MHFEM::BoundaryConditionsType::AdvectiveOutflow)
    ;

    using BCS = TNL::MHFEM::BoundaryConditionsStorage< RealType >;

    void (BCS::* _save1)( const std::string &) const = &BCS::save;
    void (BCS::* _load1)( const std::string &) = &BCS::load;
    void (BCS::* _save2)( TNL::File &) const = &BCS::save;
    void (BCS::* _load2)( TNL::File &) = &BCS::load;

    nb::class_< BCS >(m, "BoundaryConditionsStorage")
        .def(nb::init<>())
        .def_rw("dofSize", &BCS::dofSize)
        .def_rw("tags", &BCS::tags)
        .def_rw("values", &BCS::values)
        .def_rw("dirichletValues", &BCS::dirichletValues)
        .def("save", _save1)
        .def("load", _load1)
        .def("save", _save2)
        .def("load", _load2)
    ;

    export_Array< decltype(std::declval<BCS>().tags) >( m, "Array_BoundaryConditionsType" );
    export_Array< decltype(std::declval<BCS>().values) >( m, "Array_values" );
}
