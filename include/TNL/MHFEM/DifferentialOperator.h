#pragma once

#include <TNL/Meshes/Grid.h>

#include "SecondaryCoefficients.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class DifferentialOperator
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename Mesh::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using LocalIndex = typename Mesh::LocalIndexType;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType E,
                                        const int i ) const;

    template< typename Matrix, typename Vector >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const MeshDependentDataType & mdd,
                            const IndexType E,
                            const int i,
                            const RealType time,
                            const RealType tau,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    using coeff = SecondaryCoefficients< MeshDependentDataType >;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    using MeshType = TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType E,
                                        const int i ) const;

    template< typename Matrix, typename Vector >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const MeshDependentDataType & mdd,
                            const IndexType E,
                            const int i,
                            const RealType time,
                            const RealType tau,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    using coeff = SecondaryCoefficients< MeshDependentDataType >;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    using MeshType = TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType E,
                                        const int i ) const;

    template< typename Matrix, typename Vector >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const MeshDependentDataType & mdd,
                            const IndexType E,
                            const int i,
                            const RealType time,
                            const RealType tau,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    using coeff = SecondaryCoefficients< MeshDependentDataType >;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    using MeshType = TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType E,
                                        const int i ) const;

    template< typename Matrix, typename Vector >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const MeshDependentDataType & mdd,
                            const IndexType E,
                            const int i,
                            const RealType time,
                            const RealType tau,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    using coeff = SecondaryCoefficients< MeshDependentDataType >;
};

} // namespace mhfem

#include "DifferentialOperator_impl.h"
