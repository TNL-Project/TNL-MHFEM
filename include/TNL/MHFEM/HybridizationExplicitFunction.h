#pragma once

#include <mesh/tnlGrid.h>
#include <core/vectors/tnlSharedVector.h>

#include "../mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class HybridizationExplicitFunction
{
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class HybridizationExplicitFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;

    void bind( MeshDependentDataType* mdd,
               DofVectorType & dofVector )
    {
        this->mdd = mdd;
        this->dofVector.bind( dofVector.getData(), dofVector.getSize() );
    }

    __cuda_callable__
    RealType getValue( const MeshType & mesh,
                       const IndexType & index,
                       const RealType & time ) const
    {
        const IndexType cells = mesh.getNumberOfCells();
        const IndexType K = index % cells;
        const int i = index / cells;

        RealType result = 0.0;
        IndexType faceIndexes[ 4 ];
        getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

        for( int f = 0; f < mdd->facesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < mdd->n; j++ ) {
                result += mdd->R_ijKe( i, j, K, f ) * dofVector[ mdd->getDofIndex( j, F ) ];
            }
        }

        result += mdd->R_iK( i, K );

        return result;
    }

protected:
    MeshDependentDataType* mdd;
    SharedVectorType dofVector;
};

} // namespace mhfem

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class tnlFunctionType< mhfem::HybridizationExplicitFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData > >
{
public:
//    enum { Type = tnlDiscreteFunction };
    // specify type so that tnlFunctionAdapter passes the right parameters to getValue method
    enum { Type = tnlGeneralFunction };
};
