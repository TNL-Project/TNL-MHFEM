#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class RightHandSide
    : public TNL::Functions::Domain< Mesh::meshDimensions, TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    typedef Mesh MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    static constexpr int getEntitiesDimensions() { return Mesh::meshDimensions - 1; }
 
    void bindMeshDependentData( MeshDependentDataType* mdd )
    {
        this->mdd = mdd;
    }

    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getDimensions() == getEntitiesDimensions(),
                       "This function is defined on faces." );

        const MeshType & mesh = entity.getMesh();
        const IndexType E = entity.getIndex();

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        tnlAssert( numCells == 2,
                   std::cerr << "assertion numCells == 2 failed" << std::endl; );

        // prepare right hand side value
        RealType result = 0.0;
        for( int xxx = 0; xxx < numCells; xxx++ ) {
            const IndexType & K = cellIndexes[ xxx ];

            // find local index of face E
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );
            const int e = getLocalIndex( faceIndexes, E );

            result += mdd->w_iKe( i, K, e );
            for( int j = 0; j < mdd->NumberOfEquations; j++ ) {
                result += MeshDependentDataType::MassMatrix::b_ijKe( *mdd, i, j, K, e ) * mdd->R_iK( j, K );
            }
        }
        return result;
    }

protected:
    MeshDependentDataType* mdd;
};

} // namespace mhfem
