#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Operators/Operator.h>

#include "MassMatrixDependentCode.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
class BoundaryConditions
    : public TNL::Operators::Operator< Mesh,
                                       TNL::Functions::MeshInteriorDomain,
                                       Mesh::getMeshDimensions() - 1,
                                       Mesh::getMeshDimensions() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using CoordinatesType = typename MeshType::CoordinatesType;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using TagArrayType = TNL::Containers::Array< bool, DeviceType, IndexType >;
    using FaceVectorType = TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType >;

    // NOTE: children of BoundaryConditions (i.e. ModelImplementation) must implement these methods
//    bool
//    init( const tnlParameterContainer & parameters,
//          const MeshType & mesh,
//          const MeshDependentDataType & mdd );
//
//    __cuda_callable__
//    typename MeshDependentData::RealType
//    getNeumannValue( const MeshType & mesh,
//                     const int & i,
//                     const IndexType & E,
//                     const RealType & time ) const;
//
//    __cuda_callable__
//    typename MeshDependentData::RealType
//    getDirichletValue( const MeshType & mesh,
//                       const int & i,
//                       const IndexType & E,
//                       const RealType & time ) const;

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

    __cuda_callable__
    bool isNeumannBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const;

    __cuda_callable__
    bool isDirichletBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const;

protected:
    TNL::SharedPointer< MeshDependentDataType > mdd;

    // vector holding tags to differentiate the boundary condition based on the face index
    // (true indicates Dirichlet boundary)
    TagArrayType dirichletTags;

    using coeff = MassMatrixDependentCode< MeshDependentDataType >;
};

} // namespace mhfem

#include "BoundaryConditions_impl.h"
