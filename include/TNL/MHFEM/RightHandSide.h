#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class RightHandSide
    : public TNL::Functions::Domain< Mesh::getMeshDimension(), TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    static constexpr int getEntitiesDimensions() { return Mesh::getMeshDimension() - 1; }
 
    void bind( const TNL::SharedPointer< MeshType > & mesh,
               TNL::SharedPointer< MeshDependentDataType > & mdd )
    {
        this->mesh = mesh;
        this->mdd = mdd;
    }

    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getEntityDimension() == getEntitiesDimensions(),
                       "This function is defined on faces." );

        // dereference the smart pointer on device
        const auto & mesh = this->mesh.template getData< DeviceType >();
        const auto & mdd = this->mdd.template getData< DeviceType >();

        const IndexType E = entity.getIndex();

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, entity, cellIndexes );

        TNL_ASSERT( numCells == 2,
                    std::cerr << "assertion numCells == 2 failed" << std::endl; );

        // prepare right hand side value
        RealType result = 0.0;
        for( int xxx = 0; xxx < numCells; xxx++ ) {
            const IndexType & K = cellIndexes[ xxx ];

            // find local index of face E
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            result += mdd.w_iKe( i, K, e );
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                result += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
            }
        }
        return result;
    }

protected:
    TNL::SharedPointer< MeshType > mesh;
    TNL::SharedPointer< MeshDependentDataType > mdd;
};

} // namespace mhfem
