#pragma once

#include "mesh_helpers.h"

namespace mhfem
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
};

} // namespace mhfem
