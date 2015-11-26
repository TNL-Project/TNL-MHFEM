#pragma once

#include <mesh/tnlGrid.h>
#include <functors/tnlFunction.h>

#include "../mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class RightHandSide
{
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class RightHandSide< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
    : public tnlFunction< tnlGeneralFunction >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;

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
            IndexType faceIndexes[ 4 ];
            getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );
            int e = 0;
            for( int xxx = 0; xxx < mdd->facesPerCell; xxx++ ) {
                if( faceIndexes[ xxx ] == E ) {
                    e = xxx;
                    break;
                }
            }

            result += mdd->w_iKe( i, K, e );
            for( int j = 0; j < mdd->n; j++ ) {
                result += mdd->b_ijKe( i, j, K, e ) * mdd->R_iK( j, K );
            }
        }
        return result;
    }

protected:
    MeshDependentDataType* mdd;
};

} // namespace mhfem
