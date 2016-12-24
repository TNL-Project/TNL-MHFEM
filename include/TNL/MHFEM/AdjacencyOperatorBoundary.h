#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Operators/Operator.h>

#include "../lib_general/FacesPerCell.h"

namespace mhfem {

template< typename Mesh, int NumberOfEquations = 1 >
class AdjacencyOperatorBoundary
    : public TNL::Operators::Operator< Mesh,
                                       TNL::Functions::MeshInteriorDomain,
                                       Mesh::getMeshDimensions() - 1,
                                       Mesh::getMeshDimensions() - 1,
                                       bool,
                                       typename Mesh::IndexType,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using CoordinatesType = typename MeshType::CoordinatesType;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::IndexType;
    using TagArrayType = TNL::Containers::Array< bool, DeviceType, IndexType >;
    using FaceVectorType = TNL::Containers::StaticVector< FacesPerCell< MeshType >::value, IndexType >;

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

#include "AdjacencyOperatorBoundary_impl.h"
