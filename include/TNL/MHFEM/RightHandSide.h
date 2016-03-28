#pragma once

#include <functors/tnlFunction.h>

#include "../mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class RightHandSide
    : public tnlFunction< tnlGeneralFunction >
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
        int numCells = getCellsForFace( mesh, E, cellIndexes );

        tnlAssert( numCells == 2,
                   cerr << "assertion numCells == 2 failed" << endl; );

        // prepare right hand side value
        RealType result = 0.0;
        for( int xxx = 0; xxx < numCells; xxx++ ) {
            const IndexType & K = cellIndexes[ xxx ];

            // find local index of face E
            // TODO: simplify
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );
            int e = 0;
            for( int xxx = 0; xxx < mdd->FacesPerCell; xxx++ ) {
                if( faceIndexes[ xxx ] == E ) {
                    e = xxx;
                    break;
                }
            }

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
