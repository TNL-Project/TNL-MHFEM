#pragma once

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlStaticVector.h>

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
class BoundaryConditions
{
public:
    typedef Mesh MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< bool, DeviceType, IndexType > TagVectorType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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

    void bindMeshDependentData( MeshDependentDataType* mdd );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexRow,
                                        const CoordinatesType & coordinates ) const;

    template< typename Vector, typename Matrix >
    __cuda_callable__
    void updateLinearSystem( const RealType & time,
                             const MeshType & mesh,
                             const IndexType & indexRow,
                             const CoordinatesType & coordinates,
                             Vector & u,
                             Vector & b,
                             Matrix & matrix ) const;

    __cuda_callable__
    bool isNeumannBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const;

    __cuda_callable__
    bool isDirichletBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const;

protected:
    MeshDependentDataType* mdd;

    // vector holding tags to differentiate the boundary condition based on the face index
    // (true indicates Dirichlet boundary)
    TagVectorType dirichletTags;

    __cuda_callable__
    RealType getValue( const int & i,
                       const int & j,
                       const IndexType & E,
                       const int & e,
                       const IndexType & F,
                       const int & f,
                       const IndexType & K ) const;
};

} // namespace mhfem

#include "BoundaryConditions_impl.h"
