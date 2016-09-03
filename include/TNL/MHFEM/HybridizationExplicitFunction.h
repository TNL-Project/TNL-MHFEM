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
    : public TNL::Functions::Domain< Mesh::meshDimensions, TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    typedef Mesh MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    static constexpr int getEntitiesDimensions() { return Mesh::meshDimensions; }
 
    void bind( TNL::SharedPointer< MeshDependentDataType > & mdd,
               TNL::SharedPointer< DofVectorType > & dofVector )
    {
        this->mdd = mdd;
        this->dofVector = dofVector;
    }

    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getDimensions() == getEntitiesDimensions(),
                       "This function is defined on cells." );

        const auto & mesh = entity.getMesh();
        const IndexType K = entity.getIndex();

        RealType result = 0.0;
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );

        // dereference the smart pointers on device
        const auto & mdd = this->mdd.template getData< DeviceType >();
        const auto & dofVector = this->dofVector.template getData< DeviceType >();

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
    TNL::SharedPointer< DofVectorType > dofVector;
};

} // namespace mhfem
