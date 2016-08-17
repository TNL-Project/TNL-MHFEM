#pragma once

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
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
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
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
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
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class DifferentialOperator< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public TNL::Operators::Operator< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                                       TNL::Functions::MeshInteriorDomain,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >::getMeshDimensions() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
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
    MeshDependentDataType* mdd;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;
};

} // namespace mhfem

#include "DifferentialOperator_impl.h"
