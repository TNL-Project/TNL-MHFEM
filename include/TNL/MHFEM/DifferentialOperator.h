#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Operators/Operator.h>

#include "MassMatrixDependentCode.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class DifferentialOperator
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    void bindMeshDependentData( TNL::SharedPointer< MeshDependentDataType > & mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    TNL::SharedPointer< MeshDependentDataType > mdd;
    using coeff = MassMatrixDependentCode< MeshDependentDataType >;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    void bindMeshDependentData( TNL::SharedPointer< MeshDependentDataType > & mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    TNL::SharedPointer< MeshDependentDataType > mdd;
    using coeff = MassMatrixDependentCode< MeshDependentDataType >;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimension() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    void bindMeshDependentData( TNL::SharedPointer< MeshDependentDataType > & mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename DofFunctionPointer, typename Vector, typename Matrix >
    __cuda_callable__
    void setMatrixElements( DofFunctionPointer & u,
                            const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    TNL::SharedPointer< MeshDependentDataType > mdd;
    using coeff = MassMatrixDependentCode< MeshDependentDataType >;
};

} // namespace mhfem

#include "DifferentialOperator_impl.h"
