#pragma once

#include <TNL/Containers/StaticArray.h>

#include "BoundaryConditions.h"
#include "mesh_helpers.h"
#include "SecondaryCoefficients.h"

namespace mhfem {

template< typename MeshDependentData >
struct BoundaryCoefficients_AdvectiveOutflow
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;

    __cuda_callable__
    static RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
    {
        const RealType vel = mdd.a_ijKe( i, i, K, e ) + mdd.u_ijKe( i, i, K, e );
        if( vel >= 0 ) {
            using coeff = SecondaryCoefficients< MeshDependentData >;
            return coeff::A_ijKEF_no_advection( mdd, i, j, K, E, e, F, f );
        }

        // inflow: prescribe current Z_iK using Dirichlet condition
        if( i == j && E == F )
            return 1;
        return 0;
    }

    __cuda_callable__
    static RealType
    RHS_iKE( const MeshDependentData & mdd,
             const int i,
             const IndexType K,
             const IndexType E,
             const int e )
    {
        const RealType vel = mdd.a_ijKe( i, i, K, e ) + mdd.u_ijKe( i, i, K, e );
        if( vel >= 0 ) {
            using coeff = SecondaryCoefficients< MeshDependentData >;
            return coeff::RHS_iKE_no_advection( mdd, i, K, E, e );
        }

        // inflow: prescribe current Z_iK using Dirichlet condition
        return mdd.Z_iK( i, K );
    }
};

template< typename MeshDependentData >
struct BoundaryCoefficients_FixedFlux
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;

    __cuda_callable__
    static RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int i,
             const int j,
             const IndexType K,
             const IndexType E,
             const int e,
             const IndexType F,
             const int f )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;
        return coeff::A_ijKEF_advection( mdd, i, j, K, E, e, F, f );
    }

    __cuda_callable__
    static RealType
    RHS_iKE( const MeshDependentData & mdd,
             const int i,
             const IndexType K,
             const IndexType E,
             const int e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;
        return coeff::RHS_iKE_advection( mdd, i, K, E, e );
    }
};

template< typename Mesh, typename MeshDependentData, typename BoundaryCoefficients >
struct RowSetter
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static typename MeshDependentData::RealType
    setRow(
#ifdef HAVE_HYPRE
            MatrixRow & diag_row,
            MatrixRow & offd_row,
            const Mesh & mesh,
#else
            MatrixRow & matrixRow,
#endif
            const MeshDependentData & mdd,
            const FaceIndexes & faceIndexes,
            const int i,
            const IndexType K,
            const IndexType E,
            const int e )
    {
        using RealType = typename MeshDependentData::RealType;
        using LocalIndex = typename Mesh::LocalIndexType;
        using LocalIndexPermutation = TNL::Containers::StaticArray< FaceIndexes::getSize(), LocalIndex >;

        // For unstructured meshes the face indexes might be unsorted.
        // Therefore we build another permutation array with the correct order.
#ifndef __CUDA_ARCH__
        LocalIndexPermutation localFaceIndexes;
#else
        // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
        // (we'll need to pass custom launch configuration to the traverser)
        __shared__ LocalIndexPermutation __permutations[ 256 ];
        LocalIndexPermutation& localFaceIndexes = __permutations[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
#endif
        for( LocalIndex j = 0; j < FaceIndexes::getSize(); j++ )
            localFaceIndexes[ j ] = j;
        auto comparator = [&]( LocalIndex a, LocalIndex b ) {
            return faceIndexes[ a ] < faceIndexes[ b ];
        };
        // We assume that the array size is small, so we sort it with bubble sort.
        for( LocalIndex k1 = FaceIndexes::getSize() - 1; k1 > 0; k1-- )
            for( LocalIndex k2 = 0; k2 < k1; k2++ )
                if( ! comparator( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] ) )
                    TNL::swap( localFaceIndexes[ k2 ], localFaceIndexes[ k2+1 ] );

        // we will scale the row such that the diagonal element equals one
        const RealType diagonalValue = BoundaryCoefficients::A_ijKEF( mdd, i, i, K, E, e, E, e );
        // the diagonal element should be positive
        TNL_ASSERT_GT( diagonalValue, 0, "the diagonal matrix element is not positive" );

#ifdef HAVE_HYPRE
        const IndexType localDofs = MeshDependentData::NumberOfEquations * mesh.template getGhostEntitiesOffset< Mesh::getMeshDimension() - 1 >();
        LocalIndex diagElements = 0;
        LocalIndex offdElements = 0;
#else
        LocalIndex rowElements = 0;
#endif

#ifdef HAVE_HYPRE
        // the diagonal element must be set first
        diag_row.setElement( diagElements++, mdd.getDofIndex( i, E ), 1 );
#endif

        for( LocalIndex g = 0; g < MeshDependentData::FacesPerCell; g++ ) {
            const LocalIndex f = localFaceIndexes[ g ];
            for( LocalIndex j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
                // set the diagonal element
                if( j == i && faceIndexes[ f ] == E ) {
#ifndef HAVE_HYPRE
                    matrixRow.setElement( rowElements++, mdd.getDofIndex( j, faceIndexes[ f ] ), 1 );
#endif
                }
                else {
                    const RealType value = BoundaryCoefficients::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f );
                    const IndexType dof = mdd.getDofIndex( j, faceIndexes[ f ] );
#ifdef HAVE_HYPRE
                    if( dof < localDofs )
                        diag_row.setElement( diagElements++, dof, value / diagonalValue );
                    else
                        offd_row.setElement( offdElements++, dof - localDofs, value / diagonalValue );
#else
                    matrixRow.setElement( rowElements++, dof, value / diagonalValue );
#endif
                }
            }
        }

        return diagonalValue;
    }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename BoundaryCoefficients >
struct RowSetter< TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, MeshDependentData, BoundaryCoefficients >
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static typename MeshDependentData::RealType
    setRow( MatrixRow & matrixRow,
            const MeshDependentData & mdd,
            const FaceIndexes & faceIndexes,
            const int i,
            const IndexType K,
            const IndexType E,
            const int e )
    {
        // we will scale the row such that the diagonal element equals one
        const auto diagonalValue = BoundaryCoefficients::A_ijKEF( mdd, i, i, K, E, e, E, e );
        // the diagonal element should be positive
        TNL_ASSERT_GT( diagonalValue, 0, "the diagonal matrix element is not positive" );

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                // set the diagonal element
                if( j == i && faceIndexes[ f ] == E ) {
                    // NOTE: the local element index depends on the DOF vector ordering
                    matrixRow.setElement( j + MeshDependentData::NumberOfEquations * f,
                                          mdd.getDofIndex( j, faceIndexes[ f ] ),
                                          1 );
                }
                else {
                    const auto value = BoundaryCoefficients::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f );
                    // NOTE: the local element index depends on the DOF vector ordering
                    matrixRow.setElement( j + MeshDependentData::NumberOfEquations * f,
                                          mdd.getDofIndex( j, faceIndexes[ f ] ),
                                          value / diagonalValue );
                }
            }
        }

        return diagonalValue;
    }
};


template< typename MeshDependentData,
          typename BoundaryModel >
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
init( const BoundaryConditionsStorage< RealType > & storage )
{
    tags = storage.tags;
    values = storage.values;
    dirichletValues = storage.dirichletValues;
}

template< typename MeshDependentData,
          typename BoundaryModel >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< MeshDependentData, BoundaryModel >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType E,
                          const int i ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];
    if( type == BoundaryConditionsType::FixedValue )
        return 1;
    return MeshDependentDataType::FacesPerCell * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshDependentData,
          typename BoundaryModel >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< MeshDependentData, BoundaryModel >::
getLinearSystemRowLengthDiag( const MeshType & mesh,
                              const IndexType E,
                              const int i ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];
    if( type == BoundaryConditionsType::FixedValue )
        return 1;

    // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
    IndexType cellIndexes[ 2 ];
    const int numCells = getCellsForFace( mesh, E, cellIndexes );

    TNL_ASSERT_EQ( numCells, 1, "assertion numCells == 1 failed" );
    (void) numCells;  // silence unused-variable warning for Release build

    const auto faceIndexes = getFacesForCell( mesh, cellIndexes[ 0 ] );

    const IndexType localFaces = mesh.template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >();
    IndexType count = 0;
    for( int f = 0; f < MeshDependentDataType::FacesPerCell; f++ )
        if( faceIndexes[ f ] < localFaces )
            count++;

    return MeshDependentDataType::NumberOfEquations * count;
}

template< typename MeshDependentData,
          typename BoundaryModel >
    template< typename Matrix, typename Vector >
__cuda_callable__
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
setMatrixElements( const MeshType & mesh,
                   const MeshDependentDataType & mdd,
                   const IndexType rowIndex,
                   const IndexType E,
                   const int i,
                   const RealType time,
                   const RealType tau,
#ifdef HAVE_HYPRE
                   Matrix & diag,
                   Matrix & offd,
#else
                   Matrix & matrix,
#endif
                   Vector & b ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

#ifdef HAVE_HYPRE
    auto diag_row = diag.getRow( rowIndex );
    auto offd_row = offd.getRow( rowIndex );
    TNL_ASSERT_GE( diag_row.getSize(), getLinearSystemRowLengthDiag( mesh, E, i ), "diag matrix row is too small" );
    TNL_ASSERT_GE( offd_row.getSize(), getLinearSystemRowLength( mesh, E, i ) - diag_row.getSize(), "offd matrix row is too small" );
#else
    auto matrixRow = matrix.getRow( rowIndex );
    TNL_ASSERT_GE( matrixRow.getSize(), getLinearSystemRowLength( mesh, E, i ), "matrix row is too small" );
#endif

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];

    switch( type ) {
        // fixed-value (Dirichlet) boundary condition
        case BoundaryConditionsType::FixedValue:
#ifdef HAVE_HYPRE
            diag_row.setElement( 0, mdd.getDofIndex( i, E ), 1.0 );
#else
            matrixRow.setElement( 0, mdd.getDofIndex( i, E ), 1.0 );
#endif
            b[ rowIndex ] = getDirichletValue( mesh, i, E, time, tau );
            break;

        // fixed-flux (Neumann) boundary condition
        case BoundaryConditionsType::FixedFlux:
        {
            // for boundary faces returns only one valid cell index
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( mesh, E, cellIndexes );
            const IndexType & K = cellIndexes[ 0 ];

            TNL_ASSERT( numCells == 1,
                        std::cerr << "assertion numCells == 1 failed" << std::endl
                                  << "E = " << E << std::endl
                                  << "K0 = " << cellIndexes[ 0 ] << std::endl
                                  << "K1 = " << cellIndexes[ 1 ] << std::endl; );
            (void) numCells;  // silence unused-variable warning for Release build

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set non-zero elements and get the diagonal entry value
            const RealType diagonalValue =
                RowSetter< MeshType, MeshDependentDataType, BoundaryCoefficients_FixedFlux< MeshDependentDataType > >::
                    setRow(
#ifdef HAVE_HYPRE
                            diag_row,
                            offd_row,
                            mesh,
#else
                            matrixRow,
#endif
                            mdd,
                            faceIndexes,
                            i, K, E, e );

            // set right hand side value
            const auto& entity = mesh.template getEntity< typename MeshType::Face >( E );
            RealType bValue = - getNeumannValue( mesh, i, E, time, tau ) * getEntityMeasure( mesh, entity );
            // add terms from the MHFEM scheme
            bValue += BoundaryCoefficients_FixedFlux< MeshDependentDataType >::RHS_iKE( mdd, i, K, E, e );
            // scale the right hand side value by the matrix diagonal
            b[ rowIndex ] = bValue / diagonalValue;

            break;
        }

        // advective outflow boundary condition
        case BoundaryConditionsType::AdvectiveOutflow:
        {
            // for boundary faces returns only one valid cell index
            IndexType cellIndexes[ 2 ];
            const int numCells = getCellsForFace( mesh, E, cellIndexes );
            const IndexType & K = cellIndexes[ 0 ];

            TNL_ASSERT( numCells == 1,
                        std::cerr << "assertion numCells == 1 failed" << std::endl
                                  << "E = " << E << std::endl
                                  << "K0 = " << cellIndexes[ 0 ] << std::endl
                                  << "K1 = " << cellIndexes[ 1 ] << std::endl; );
            (void) numCells;  // silence unused-variable warning for Release build

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set non-zero elements
            const RealType diagonalValue =
                RowSetter< MeshType, MeshDependentDataType, BoundaryCoefficients_AdvectiveOutflow< MeshDependentDataType > >::
                    setRow(
#ifdef HAVE_HYPRE
                            diag_row,
                            offd_row,
                            mesh,
#else
                            matrixRow,
#endif
                            mdd,
                            faceIndexes,
                            i, K, E, e );

            // set right hand side value - zero diffusive flux
            RealType bValue = 0;
            // add terms from the MHFEM scheme
            bValue += BoundaryCoefficients_AdvectiveOutflow< MeshDependentDataType >::RHS_iKE( mdd, i, K, E, e );
            // scale the right hand side value by the matrix diagonal
            b[ rowIndex ] = bValue / diagonalValue;

            break;
        }

        default:
            TNL_ASSERT_TRUE( false, "unknown boundary condition type was encountered" );
            break;
    }
}

} // namespace mhfem
