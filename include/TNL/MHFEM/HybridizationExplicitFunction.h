#pragma once

#include <functors/tnlFunction.h>
#include <core/vectors/tnlSharedVector.h>

#include "../mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class HybridizationExplicitFunction
    : public tnlFunction< tnlGeneralFunction >
{
public:
    typedef Mesh MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );

        for( int f = 0; f < mdd->FacesPerCell; f++ ) {
            const IndexType F = faceIndexes[ f ];
            for( int j = 0; j < mdd->NumberOfEquations; j++ ) {
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
