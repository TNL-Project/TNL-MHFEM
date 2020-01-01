#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>
#include <TNL/Pointers/SharedPointer.h>
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

    static constexpr int getEntitiesDimensions() { return Mesh::getMeshDimension(); }
 
    void bind( const TNL::Pointers::SharedPointer< MeshType > & mesh,
               TNL::Pointers::SharedPointer< MeshDependentDataType > & mdd )
    {
        this->mesh = mesh;
        this->mdd = mdd;
    }

    // FIXME: template needed due to limitation of FunctionAdapter, otherwise we would use MeshType::Cell
    // (for grids it is different from MeshType::template EntityType< d >, because it has non-default Config parameter)
    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getEntityDimension() == getEntitiesDimensions(),
                       "This function is defined on cells." );

        // dereference the smart pointer on device
        const auto & mdd = this->mdd.template getData< DeviceType >();
        const auto & mesh = this->mesh.template getData< DeviceType >();

        const IndexType K = entity.getIndex();

        RealType result = 0.0;
        const auto faceIndexes = getFacesForCell( mesh, K );

        for( int f = 0; f < MeshDependentDataType::FacesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                result += mdd.R_ijKe( i, j, K, f ) * mdd.Z_iF( j, F );
            }
        }

        result += mdd.R_iK( i, K );

        return result;
    }

protected:
    TNL::Pointers::SharedPointer< MeshType > mesh;
    TNL::Pointers::SharedPointer< MeshDependentDataType > mdd;
};

} // namespace mhfem
