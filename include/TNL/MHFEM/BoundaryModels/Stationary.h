#pragma once

#include <TNL/Cuda/CudaCallable.h>

namespace TNL::MHFEM::BoundaryModels
{

template< typename Mesh >
struct Stationary
{
    template< typename BoundaryConditions >
    __cuda_callable__
    static typename BoundaryConditions::RealType
    getNeumannValue( const BoundaryConditions & bc,
                     const Mesh & mesh,
                     const int i,
                     const typename Mesh::GlobalIndexType E,
                     const typename BoundaryConditions::RealType time,
                     const typename BoundaryConditions::RealType tau )
    {
        const typename Mesh::GlobalIndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();
        return bc.values[ i * faces + E ];
    }

    template< typename BoundaryConditions >
    __cuda_callable__
    static typename BoundaryConditions::RealType
    getDirichletValue( const BoundaryConditions & bc,
                       const Mesh & mesh,
                       const int i,
                       const typename Mesh::GlobalIndexType E,
                       const typename BoundaryConditions::RealType time,
                       const typename BoundaryConditions::RealType tau )
    {
        const typename Mesh::GlobalIndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();
        return bc.dirichletValues[ i * faces + E ];
    }
};

} // namespace TNL::MHFEM::BoundaryModels
