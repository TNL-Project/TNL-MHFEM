#pragma once

#include <TNL/Meshes/Grid.h>

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
{
public:
    typedef TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    void bindMeshDependentData( MeshDependentDataType* mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexRow,
                                        const CoordinatesType & coordinates ) const;

    template< typename Vector, typename Matrix >
    __cuda_callable__
    void updateLinearSystem( const RealType & time,
                             const RealType & tau,
                             const MeshType & mesh,
                             const IndexType & indexRow,
                             const CoordinatesType & coordinates,
                             Vector & u,
                             Vector & b,
                             Matrix & matrix ) const;

protected:
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    void bindMeshDependentData( MeshDependentDataType* mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexRow,
                                        const CoordinatesType & coordinates ) const;

    template< typename Vector, typename Matrix >
    __cuda_callable__
    void updateLinearSystem( const RealType & time,
                             const RealType & tau,
                             const MeshType & mesh,
                             const IndexType & indexRow,
                             const CoordinatesType & coordinates,
                             Vector & u,
                             Vector & b,
                             Matrix & matrix ) const;

protected:
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    void bindMeshDependentData( MeshDependentDataType* mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexRow,
                                        const CoordinatesType & coordinates ) const;

    template< typename Vector, typename Matrix >
    __cuda_callable__
    void updateLinearSystem( const RealType & time,
                             const RealType & tau,
                             const MeshType & mesh,
                             const IndexType & indexRow,
                             const CoordinatesType & coordinates,
                             Vector & u,
                             Vector & b,
                             Matrix & matrix ) const;

protected:
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

} // namespace mhfem

#include "DifferentialOperator_impl.h"
