#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/Array.h>
#include <TNL/Operators/Operator.h>

#include "BoundaryConditionsType.h"

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
    using TagArrayType = TNL::Containers::Array< BoundaryConditionsType, DeviceType, IndexType >;
    using ValueArrayType = TNL::Containers::Array< RealType, DeviceType, IndexType >;

    // NOTE: children of BoundaryConditions (i.e. ModelImplementation) must implement these methods
//    bool
//    init( const tnlParameterContainer & parameters,
//          const MeshType & mesh,
//          const MeshDependentDataType & mdd );
//
//    __cuda_callable__
//    typename MeshDependentData::RealType
//    getNeumannValue( const MeshType & mesh,
//                     const int i,
//                     const IndexType E,
//                     const RealType time,
//                     const RealType tau ) const;
//
//    __cuda_callable__
//    typename MeshDependentData::RealType
//    getDirichletValue( const MeshType & mesh,
//                       const int i,
//                       const IndexType E,
//                       const RealType time,
//                       const RealType tau ) const;

    bool init( const TNL::Config::ParameterContainer & parameters,
               const MeshType & mesh );

    template< typename MeshOrdering >
    void reorderBoundaryConditions( const MeshOrdering & meshOrdering );

    void bind( const TNL::Pointers::SharedPointer< MeshType > & mesh,
               TNL::Pointers::SharedPointer< MeshDependentDataType > & mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Vector, typename Matrix >
    __cuda_callable__
    void setMatrixElements( const typename MeshType::Face & entity,
                            const RealType & time,
                            const RealType & tau,
                            const int & i,
                            Matrix & matrix,
                            Vector & b ) const;

protected:
    TNL::Pointers::SharedPointer< MeshType > mesh;
    TNL::Pointers::SharedPointer< MeshDependentDataType > mdd;

    // array holding tags to differentiate the boundary condition based on the face index
    TagArrayType tags;

    // array holding the values to be interpreted by the boundary condition specified on each face
    ValueArrayType values;
};

} // namespace mhfem

#include "BoundaryConditions_impl.h"
