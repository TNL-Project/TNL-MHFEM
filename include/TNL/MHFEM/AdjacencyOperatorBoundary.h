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
                                       Mesh::getMeshDimension() - 1,
                                       Mesh::getMeshDimension() - 1,
                                       bool,
                                       typename Mesh::GlobalIndexType,
                                       NumberOfEquations,
                                       NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;
    using TagArrayType = TNL::Containers::Array< bool, DeviceType, IndexType >;

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType & indexEntity,
                                        const typename MeshType::Face & entity,
                                        const int & i ) const;

    template< typename Matrix >
    __cuda_callable__
    void setMatrixElements( const Mesh & mesh,
                            const typename MeshType::Face & entity,
                            const int & i,
                            Matrix & matrix ) const;
};

} // namespace mhfem

#include "AdjacencyOperatorBoundary_impl.h"
