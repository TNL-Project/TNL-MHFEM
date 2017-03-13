#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Operators/Operator.h>

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
class BoundaryConditions
    : public TNL::Operators::Operator< Mesh,
                                       TNL::Functions::MeshInteriorDomain,
                                       Mesh::getMeshDimension() - 1,
                                       Mesh::getMeshDimension() - 1,
                                       typename MeshDependentData::RealType,
                                       typename MeshDependentData::IndexType,
                                       MeshDependentData::NumberOfEquations,
                                       MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using TagArrayType = TNL::Containers::Array< bool, DeviceType, IndexType >;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

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

    bool init( const TNL::Config::ParameterContainer & parameters,
               const MeshType & mesh );

    template< typename MeshOrdering >
    bool reorderBoundaryConditions( const MeshOrdering & meshOrdering );

    void bind( const TNL::SharedPointer< MeshType > & mesh,
               TNL::SharedPointer< MeshDependentDataType > & mdd );

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
    TNL::SharedPointer< MeshType > mesh;
    TNL::SharedPointer< MeshDependentDataType > mdd;

    // vector holding tags to differentiate the boundary condition based on the face index
    // (true indicates Dirichlet boundary)
    TagArrayType dirichletTags;

    // vectors holding the Dirichlet and Neumann values
    DofVectorType dirichletValues, neumannValues;
};

} // namespace mhfem

#include "BoundaryConditions_impl.h"
