#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Containers/StaticVector.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class HybridizationExplicitFunction
    : public TNL::Functions::Domain< Mesh::meshDimensions, TNL::Functions::MeshDomain >
{
public:
    typedef Mesh MeshType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
    typedef TNL::Containers::SharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;

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
