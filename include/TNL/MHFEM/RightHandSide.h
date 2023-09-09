#pragma once

#include "mesh_helpers.h"
#include "SecondaryCoefficients.h"

namespace TNL::MHFEM
{

template< typename MeshDependentData >
struct RightHandSide
{
    using MeshType = typename MeshDependentData::MeshType;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;

    __cuda_callable__
    static RealType
    getValue( const MeshType & mesh,
              const MeshDependentDataType & mdd,
              const IndexType E,
              const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "this is a bug" );

        // prepare right hand side value
        RealType result = 0.0;
        for( int xxx = 0; xxx < numCells; xxx++ ) {
            const IndexType & K = cellIndexes[ xxx ];

            // find local index of face E
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            using coeff = SecondaryCoefficients< MeshDependentData >;
            result += coeff::RHS_iKE( mdd, i, K, E, e );
        }
        return result;
    }
};

} // namespace TNL::MHFEM
