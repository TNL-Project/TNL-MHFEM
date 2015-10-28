#pragma once

#include <mesh/tnlGrid.h>
#include <core/vectors/tnlSharedVector.h>

#include "../mesh_helpers.h"

template< typename Mesh,
          typename MeshDependentData >
class Upwind
{
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class Upwind< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;

    void bind( MeshDependentDataType* mdd,
               DofVectorType & Z_iF )
    {
        this->mdd = mdd;
        this->Z_iF.bind( Z_iF.getData(), Z_iF.getSize() );
    }

    // FIXME: velocities should be pre-calculated before determining upwind values and updating the pressure values
    __cuda_callable__
    RealType getVelocity( const MeshType & mesh,
                          const int & i,
                          const IndexType & K,
                          const IndexType & E ) const
    {
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

        RealType result = 0.0;
        for( int j = 0; j < mdd->n; j++ ) {
            result += mdd->b_ijKe( i, j, K, e ) * ( mdd->Z_iK( j, K ) - Z_iF[ mdd->getDofIndex( j, E ) ] );
        }
        result += mdd->w_iKe( i, K, e );
        return result;
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
        
        // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
        const IndexType & K1 = cellIndexes[ 0 ];

        if( getVelocity( mesh, i, K1, E ) >= 0 ) {
            return mdd->m_iK( i, K1 );
        }
        else if( numCells == 2 ) {
            const IndexType & K2 = cellIndexes[ 1 ];
            return mdd->m_iK( i, K2 );
        }
        else {
            // FIXME: boundary condition has to be respected (we need to know the density on \Gamma_c ... part of the boundary where the fluid flows in)
            return mdd->m_iK( i, K1 );
        }
    }

protected:
    MeshDependentDataType* mdd;
    // auxiliary vector shared with other objects
    SharedVectorType Z_iF;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class tnlFunctionType< Upwind< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData > >
{
public:
//    enum { Type = tnlDiscreteFunction };
    // specify type so that tnlFunctionAdapter passes the right parameters to getValue method
    enum { Type = tnlGeneralFunction };
};
