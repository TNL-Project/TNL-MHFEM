#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Operators/Operator.h>

#include "../lib_general/FacesPerCell.h"

namespace mhfem {

template< typename Mesh, int NumberOfEquations = 1 >
class AdjacencyOperator
{
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
    using CoordinatesType = typename MeshType::CoordinatesType;
    using DeviceType = Device;
    using IndexType = MeshIndex;
    using FaceVectorType = TNL::Containers::StaticVector< FacesPerCell< typename MeshType::Cell >::value, IndexType >;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix, typename RealType >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const
    {
        setMatrixElements( entity, i, matrix );
    }

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const typename MeshType::Face & entity,
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
    using CoordinatesType = typename MeshType::CoordinatesType;
    using DeviceType = Device;
    using IndexType = MeshIndex;
    using FaceVectorType = TNL::Containers::StaticVector< FacesPerCell< typename MeshType::Cell >::value, IndexType >;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix, typename RealType >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const
    {
        setMatrixElements( entity, i, matrix );
    }

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const typename MeshType::Face & entity,
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
    using CoordinatesType = typename MeshType::CoordinatesType;
    using DeviceType = Device;
    using IndexType = MeshIndex;
    using FaceVectorType = TNL::Containers::StaticVector< FacesPerCell< typename MeshType::Cell >::value, IndexType >;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix, typename RealType >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const
    {
        setMatrixElements( entity, i, matrix );
    }

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

} // namespace mhfem

#include "AdjacencyOperator_impl.h"
