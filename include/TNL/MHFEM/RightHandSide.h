#pragma once

#include <functions/tnlDomain.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class RightHandSide
    : public tnlDomain< Mesh::Dimensions - 1, MeshDomain >
{
public:
    typedef Mesh MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

    void bindMeshDependentData( MeshDependentDataType* mdd )
    {
        this->mdd = mdd;
    }

    __cuda_callable__
    RealType getValue( const MeshType & mesh,
                       const IndexType & indexRow,
                       const RealType & time ) const
    {
        const IndexType E = mdd->indexDofToFace( indexRow );
        const int i = mdd->indexDofToEqno( indexRow );

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        tnlAssert( numCells == 2,
                   cerr << "assertion numCells == 2 failed" << endl; );

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
