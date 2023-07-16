#pragma once

#include <pytnl/tnl/Array.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/MHFEM/BoundaryConditionsStorage.h>

// operator needed for the bindings for Array<BoundaryConditionsType>
namespace TNL::MHFEM {
    std::ostream& operator<<( std::ostream& str, BoundaryConditionsType type )
    {
        return str << (int) type;
    }
}

template< typename RealType, typename IndexType >
void export_BoundaryConditionsStorage( py::module & m )
{
    py::enum_< TNL::MHFEM::BoundaryConditionsType >(m, "BoundaryConditionsType")
        .value("FixedValue", TNL::MHFEM::BoundaryConditionsType::FixedValue)
        .value("FixedFlux", TNL::MHFEM::BoundaryConditionsType::FixedFlux)
        .value("FixedFluxNoAdvection", TNL::MHFEM::BoundaryConditionsType::FixedFluxNoAdvection)
        .value("AdvectiveOutflow", TNL::MHFEM::BoundaryConditionsType::AdvectiveOutflow)
    ;

    using BCS = TNL::MHFEM::BoundaryConditionsStorage< RealType >;

    void (BCS::* _save1)( const TNL::String &) const = &BCS::save;
    void (BCS::* _load1)( const TNL::String &) = &BCS::load;
    void (BCS::* _save2)( TNL::File &) const = &BCS::save;
    void (BCS::* _load2)( TNL::File &) = &BCS::load;

    py::class_< BCS >(m, "BoundaryConditionsStorage")
        .def(py::init<>())
        .def_readwrite("dofSize", &BCS::dofSize)
        .def_readwrite("tags", &BCS::tags)
        .def_readwrite("values", &BCS::values)
        .def_readwrite("dirichletValues", &BCS::dirichletValues)
        .def("save", _save1)
        .def("load", _load1)
        .def("save", _save2)
        .def("load", _load2)
    ;

    export_Array< decltype(std::declval<BCS>().tags) >( m, "Array_BoundaryConditionsType" );
    export_Array< decltype(std::declval<BCS>().values) >( m, "Array_values" );
}
