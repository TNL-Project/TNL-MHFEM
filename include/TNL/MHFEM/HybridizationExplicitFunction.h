#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>
#include <TNL/SharedPointer.h>
#include <TNL/Containers/StaticVector.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class HybridizationExplicitFunction
    : public TNL::Functions::Domain< Mesh::getMeshDimension(), TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType>;
    using FaceVectorType = TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType >;

    static constexpr int getEntitiesDimensions() { return Mesh::getMeshDimension(); }
 
    void bind( TNL::SharedPointer< MeshDependentDataType > & mdd,
               DofVectorType & dofVector )
    {
        this->mdd = mdd;
        this->dofVector.bind( dofVector );
    }

    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getEntityDimension() == getEntitiesDimensions(),
                       "This function is defined on cells." );

        const auto & mesh = entity.getMesh();
        const IndexType K = entity.getIndex();

        RealType result = 0.0;
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );

        // dereference the smart pointers on device
        const auto & mdd = this->mdd.template getData< DeviceType >();

        for( int f = 0; f < MeshDependentDataType::FacesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                result += mdd.R_ijKe( i, j, K, f ) * dofVector[ mdd.getDofIndex( j, F ) ];
            }
        }

        result += mdd.R_iK( i, K );

        return result;
    }

protected:
    TNL::SharedPointer< MeshDependentDataType > mdd;
    DofVectorType dofVector;
};

} // namespace mhfem
