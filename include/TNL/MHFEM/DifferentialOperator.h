#pragma once

#include <mesh/tnlGrid.h>

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
class DifferentialOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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
class DifferentialOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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
class DifferentialOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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
