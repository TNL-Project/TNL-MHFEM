#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Operators/Operator.h>

#include "../lib_general/FacesPerCell.h"

namespace mhfem {

template< typename Mesh, int NumberOfEquations = 1 >
class AdjacencyOperator
    : public TNL::Operators::Operator< Mesh,
                                       TNL::Functions::MeshInteriorDomain,
                                       Mesh::getMeshDimension() - 1,
                                       Mesh::getMeshDimension() - 1,
                                       bool,
                                       typename Mesh::GlobalIndexType,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using DeviceType = typename Mesh::DeviceType;
    using IndexType = typename Mesh::GlobalIndexType;
    using LocalIndex = typename Mesh::LocalIndexType;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const Mesh & mesh,
                            const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
class AdjacencyOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, NumberOfEquations >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       bool,
                                       MeshIndex,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
    using DeviceType = Device;
    using IndexType = MeshIndex;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
class AdjacencyOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, NumberOfEquations >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       bool,
                                       MeshIndex,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
    using DeviceType = Device;
    using IndexType = MeshIndex;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          int NumberOfEquations >
class AdjacencyOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, NumberOfEquations >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       bool,
                                       MeshIndex,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
    using DeviceType = Device;
    using IndexType = MeshIndex;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

} // namespace mhfem

#include "AdjacencyOperator_impl.h"
