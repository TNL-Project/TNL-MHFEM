#pragma once

#include <TNL/Meshes/Grid.h>
#include "RightHandSide.h"
#include "SecondaryCoefficients.h"
#include "mesh_helpers.h"

namespace TNL::MHFEM
{

template< typename Mesh,
          typename MeshDependentData >
class LinearSystem
{
protected:
    using coeff = SecondaryCoefficients< MeshDependentData >;
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename Mesh::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using RHS = RightHandSide< MeshDependentData >;

    __cuda_callable__
    static IndexType
    getRowLength( const MeshType & mesh,
                  const IndexType E,
                  const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );
        return ( 2 * MeshDependentDataType::FacesPerCell - 1 ) * MeshDependentDataType::NumberOfEquations;
    }

    __cuda_callable__
    static IndexType
    getRowLengthDiag( const MeshType & mesh,
                      const IndexType E,
                      const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

        // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "assertion numCells == 2 failed" );
        (void) numCells;  // silence unused-variable warning for Release build

        const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
        const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

        const IndexType localFaces = mesh.template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >();
        IndexType count = 0;
        for( LocalIndex f = 0; f < MeshDependentDataType::FacesPerCell; f++ )
            if( faceIndexesK0[ f ] < localFaces )
                count++;
        for( LocalIndex f = 0; f < MeshDependentDataType::FacesPerCell; f++ )
            if( faceIndexesK1[ f ] < localFaces && faceIndexesK1[ f ] != E )
                count++;

        return MeshDependentDataType::NumberOfEquations * count;
    }

    template< typename Matrix, typename Vector >
    __cuda_callable__
    static RealType
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
                       Vector & b )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

#ifdef HAVE_HYPRE
        auto diag_row = diag.getRow( rowIndex );
        auto offd_row = offd.getRow( rowIndex );
        TNL_ASSERT_GE( diag_row.getSize(), getRowLengthDiag( mesh, E, i ), "diag matrix row is too small" );
        TNL_ASSERT_GE( offd_row.getSize(), getRowLength( mesh, E, i ) - diag_row.getSize(), "offd matrix row is too small" );
#else
        auto matrixRow = matrix.getRow( rowIndex );
        TNL_ASSERT_GE( matrixRow.getSize(), getRowLength( mesh, E, i ), "matrix row is too small" );
#endif

        // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "assertion numCells == 2 failed" );
        (void) numCells;  // silence unused-variable warning for Release build

        const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
        const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

        using LocalIndexPermutation = TNL::Containers::StaticArray< MeshDependentDataType::FacesPerCell, LocalIndex >;

        // For unstructured meshes the face indexes might be unsorted.
        // Therefore we build another permutation array with the correct order.
#ifndef __CUDA_ARCH__
        LocalIndexPermutation localFaceIndexesK0;
        LocalIndexPermutation localFaceIndexesK1;
#else
        // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
        // (we'll need to pass custom launch configuration to the traverser)
        __shared__ LocalIndexPermutation __permutationsK0[ 256 ];
        __shared__ LocalIndexPermutation __permutationsK1[ 256 ];
        LocalIndexPermutation& localFaceIndexesK0 = __permutationsK0[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
        LocalIndexPermutation& localFaceIndexesK1 = __permutationsK1[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
#endif
        for( LocalIndex j = 0; j < MeshDependentDataType::FacesPerCell; j++ )
            localFaceIndexesK0[ j ] = localFaceIndexesK1[ j ] = j;
        auto comparatorK0 = [&]( LocalIndex a, LocalIndex b ) {
            return faceIndexesK0[ a ] < faceIndexesK0[ b ];
        };
        auto comparatorK1 = [&]( LocalIndex a, LocalIndex b ) {
            return faceIndexesK1[ a ] < faceIndexesK1[ b ];
        };
        // We assume that the array size is small, so we sort it with bubble sort.
        for( LocalIndex k1 = MeshDependentDataType::FacesPerCell - 1; k1 > 0; k1-- )
            for( LocalIndex k2 = 0; k2 < k1; k2++ ) {
                if( ! comparatorK0( localFaceIndexesK0[ k2 ], localFaceIndexesK0[ k2+1 ] ) )
                    TNL::swap( localFaceIndexesK0[ k2 ], localFaceIndexesK0[ k2+1 ] );
                if( ! comparatorK1( localFaceIndexesK1[ k2 ], localFaceIndexesK1[ k2+1 ] ) )
                    TNL::swap( localFaceIndexesK1[ k2 ], localFaceIndexesK1[ k2+1 ] );
            }

        const LocalIndex e0 = getLocalIndex( faceIndexesK0, E );
        const LocalIndex e1 = getLocalIndex( faceIndexesK1, E );

#ifdef HAVE_HYPRE
        const IndexType localDofs = MeshDependentDataType::NumberOfEquations * mesh.template getGhostEntitiesOffset< MeshType::getMeshDimension() - 1 >();
        LocalIndex diagElements = 0;
        LocalIndex offdElements = 0;
#else
        LocalIndex rowElements = 0;
#endif

        // we will scale the row such that the diagonal element equals one
        const RealType diagonalValue = coeff::A_ijKEF( mdd, i, i, cellIndexes[ 0 ], E, e0, E, e0 ) +
                                       coeff::A_ijKEF( mdd, i, i, cellIndexes[ 1 ], E, e1, E, e1 );
        // the diagonal element should be positive
        TNL_ASSERT_GT( diagonalValue, 0, "the diagonal matrix element is not positive" );

        auto setElement = [&] ( RealType value, IndexType dof )
        {
#ifdef HAVE_HYPRE
            if( dof < localDofs )
                diag_row.setElement( diagElements++, dof, value / diagonalValue );
            else
                offd_row.setElement( offdElements++, dof - localDofs, value / diagonalValue );
#else
            matrixRow.setElement( rowElements++, dof, value / diagonalValue );
#endif
        };
#ifdef HAVE_HYPRE
        // the diagonal element must be set first
        diag_row.setElement( diagElements++, mdd.getDofIndex( i, E ), 1 );
#endif

        // NOTE: the placement of the j-loop depends on the DOF vector ordering
        //for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
        {
            LocalIndex g0 = 0;
            LocalIndex g1 = 0;

#ifndef NDEBUG
            bool setDiag = false;
#endif

            while( g0 < MeshDependentDataType::FacesPerCell && g1 < MeshDependentDataType::FacesPerCell ) {
                const LocalIndex f0 = localFaceIndexesK0[ g0 ];
                const LocalIndex f1 = localFaceIndexesK1[ g1 ];
                if( faceIndexesK0[ f0 ] < faceIndexesK1[ f1 ] ) {
                    for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                        const RealType value = coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, e0, faceIndexesK0[ f0 ], f0 );
                        const IndexType dof = mdd.getDofIndex( j, faceIndexesK0[ f0 ] );
                        setElement( value, dof );
                    }
                    g0++;
                }
                else if( faceIndexesK0[ f0 ] == faceIndexesK1[ f1 ] ) {
                    TNL_ASSERT_FALSE( setDiag, "" );
                    for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                        // set the diagonal element
                        if( j == i ) {
#ifndef HAVE_HYPRE
                            matrixRow.setElement( rowElements++,
                                                  mdd.getDofIndex( j, faceIndexesK0[ f0 ] ),
                                                  1 );
#endif
                        }
                        else {
                            const RealType value = coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, e0, faceIndexesK0[ f0 ], f0 ) +
                                                   coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, e1, faceIndexesK1[ f1 ], f1 );
                            const IndexType dof = mdd.getDofIndex( j, faceIndexesK0[ f0 ] );
                            setElement( value, dof );
                        }
                    }
                    g0++;
                    g1++;
#ifndef NDEBUG
                    setDiag = true;
#endif
                }
                else {
                    for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                        const RealType value = coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, e1, faceIndexesK1[ f1 ], f1 );
                        const IndexType dof = mdd.getDofIndex( j, faceIndexesK1[ f1 ] );
                        setElement( value, dof );
                    }
                    g1++;
                }
            }
            TNL_ASSERT_TRUE( setDiag == true, "" );

            while( g0 < MeshDependentDataType::FacesPerCell ) {
                const LocalIndex f0 = localFaceIndexesK0[ g0 ];
                for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                    const RealType value = coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, e0, faceIndexesK0[ f0 ], f0 );
                    const IndexType dof = mdd.getDofIndex( j, faceIndexesK0[ f0 ] );
                    setElement( value, dof );
                }
                g0++;
            }

            while( g1 < MeshDependentDataType::FacesPerCell ) {
                const LocalIndex f1 = localFaceIndexesK1[ g1 ];
                for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                    const RealType value = coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, e1, faceIndexesK1[ f1 ], f1 );
                    const IndexType dof = mdd.getDofIndex( j, faceIndexesK1[ f1 ] );
                    setElement( value, dof );
                }
                g1++;
            }
        }

#ifdef HAVE_HYPRE
        TNL_ASSERT_EQ( diagElements, getRowLengthDiag( mesh, E, i ), "bug: diagElements reached wrong value" );
        TNL_ASSERT_EQ( offdElements, getRowLength( mesh, E, i ) - getRowLengthDiag( mesh, E, i ), "bug: offdElements reached wrong value" );
#else
        TNL_ASSERT_EQ( rowElements, getRowLength( mesh, E, i ), "bug: rowElements reached wrong value" );
#endif

        return diagonalValue;
    }
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class LinearSystem< TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >
{
protected:
    using coeff = SecondaryCoefficients< MeshDependentData >;
public:
    using MeshType = TNL::Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using RHS = RightHandSide< MeshDependentData >;

    __cuda_callable__
    static IndexType
    getRowLength( const MeshType & mesh,
                  const IndexType E,
                  const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );
        return 3 * MeshDependentDataType::NumberOfEquations;
    }

    template< typename Matrix, typename Vector >
    __cuda_callable__
    static void
    setMatrixElements( const MeshType & mesh,
                       const MeshDependentDataType & mdd,
                       const IndexType rowIndex,
                       const IndexType E,
                       const int i,
                       const RealType time,
                       const RealType tau,
                       Matrix & matrix,
                       Vector & b )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

        auto matrixRow = matrix.getRow( rowIndex );

        // indexes of the right (cellIndexes[0]) and left (cellIndexes[1]) cells
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "assertion numCells == 2 failed" );
        (void) numCells;  // silence unused-variable warning for Release build

        // face indexes are ordered in this way:
        //      0   1|2   3
        //      |____|____|
        //        K1   K0
        const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
        const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            // TODO: scale the row such that the diagonal element equals one
            matrixRow.setElement( j * 3 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
            matrixRow.setElement( j * 3 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                       coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
            matrixRow.setElement( j * 3 + 2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
        }
    }
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class LinearSystem< TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
protected:
    using coeff = SecondaryCoefficients< MeshDependentData >;
public:
    using MeshType = TNL::Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using RHS = RightHandSide< MeshDependentData >;

    __cuda_callable__
    static IndexType
    getRowLength( const MeshType & mesh,
                  const IndexType E,
                  const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );
        return 7 * MeshDependentDataType::NumberOfEquations;
    }

    template< typename Matrix, typename Vector >
    __cuda_callable__
    static void
    setMatrixElements( const MeshType & mesh,
                       const MeshDependentDataType & mdd,
                       const IndexType rowIndex,
                       const IndexType E,
                       const int i,
                       const RealType time,
                       const RealType tau,
                       Matrix & matrix,
                       Vector & b )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

        auto matrixRow = matrix.getRow( rowIndex );

        // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "assertion numCells == 2 failed" );
        (void) numCells;  // silence unused-variable warning for Release build

        // face indexes for both cells
        const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
        const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

        if( E < mesh.getNumberOfNxFaces() ) {
            //        K1   K0
            //      ___________
            //      | 6  |  7 |
            //     0|   1|2   |3
            //      |____|____|
            //        4     5
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                // TODO: scale the row such that the diagonal element equals one
                matrixRow.setElement( j * 7 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
                matrixRow.setElement( j * 7 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                           coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
                matrixRow.setElement( j * 7 + 2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
                matrixRow.setElement( j * 7 + 3, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
                matrixRow.setElement( j * 7 + 4, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
                matrixRow.setElement( j * 7 + 5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
                matrixRow.setElement( j * 7 + 6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
            }
        }
        else {
            //      ______
            //      | 7  |
            //     2|   3| K0
            //      |_6__|
            //      | 5  |
            //     0|   1| K1
            //      |____|
            //        4
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                // TODO: scale the row such that the diagonal element equals one
                matrixRow.setElement( j * 7 + 0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
                matrixRow.setElement( j * 7 + 1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
                matrixRow.setElement( j * 7 + 2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
                matrixRow.setElement( j * 7 + 3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
                matrixRow.setElement( j * 7 + 4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
                matrixRow.setElement( j * 7 + 5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                           coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
                matrixRow.setElement( j * 7 + 6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
            }
        }
    }
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class LinearSystem< TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >, MeshDependentData >
{
protected:
    using coeff = SecondaryCoefficients< MeshDependentData >;
public:
    using MeshType = TNL::Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = Device;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using RHS = RightHandSide< MeshDependentData >;

    __cuda_callable__
    static IndexType
    getRowLength( const MeshType & mesh,
                  const IndexType E,
                  const int i )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );
        return 11 * MeshDependentDataType::NumberOfEquations;
    }

    template< typename Matrix, typename Vector >
    __cuda_callable__
    static void
    setMatrixElements( const MeshType & mesh,
                       const MeshDependentDataType & mdd,
                       const IndexType rowIndex,
                       const IndexType E,
                       const int i,
                       const RealType time,
                       const RealType tau,
                       Matrix & matrix,
                       Vector & b )
    {
        TNL_ASSERT_TRUE( ! isBoundaryFace( mesh, E ), "" );

        auto matrixRow = matrix.getRow( rowIndex );

        // indexes of the right/top (cellIndexes[0]) and left/bottom (cellIndexes[1]) cells
        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, E, cellIndexes );

        TNL_ASSERT_EQ( numCells, 2, "assertion numCells == 2 failed" );
        (void) numCells;  // silence unused-variable warning for Release build

        // face indexes for both cells
        const auto faceIndexesK0 = getFacesForCell( mesh, cellIndexes[ 0 ] );
        const auto faceIndexesK1 = getFacesForCell( mesh, cellIndexes[ 1 ] );

        if( E < mesh.getNumberOfNxFaces() ) {
            //        K1   K0
            //      ___________
            //      | 6  |  7 |
            //     0|   1|2   |3
            //      |____|____|
            //        4     5
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                // TODO: scale the row such that the diagonal element equals one
                matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 1 ], 1 ) +
                                                                                             coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 1 ], 1 ) );
                matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 3 ], 3 ) );
                matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 3 ], 3 ) );
                matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 1, faceIndexesK1[ 5 ], 5 ) );
                matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 0, faceIndexesK0[ 5 ], 5 ) );
            }
        }
        else if( E < mesh.getNumberOfNxAndNyFaces() ) {
            //      ______
            //      | 7  |
            //     2|   3| K0
            //      |_6__|
            //      | 5  |
            //     0|   1| K1
            //      |____|
            //        4
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                // TODO: scale the row such that the diagonal element equals one
                matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 1 ], 1 ) );
                matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 1 ], 1 ) );
                matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 3 ], 3 ) +
                                                                                             coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 3 ], 3 ) );
                matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK0[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 3, faceIndexesK1[ 5 ], 5 ) );
                matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 2, faceIndexesK0[ 5 ], 5 ) );
            }
        }
        else {
            // E is n_z face
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                // TODO: scale the row such that the diagonal element equals one
                matrixRow.setElement( j * 11 +  0, mdd.getDofIndex( j, faceIndexesK1[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  1, mdd.getDofIndex( j, faceIndexesK1[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 1 ], 1 ) );
                matrixRow.setElement( j * 11 +  2, mdd.getDofIndex( j, faceIndexesK0[ 0 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 0 ], 0 ) );
                matrixRow.setElement( j * 11 +  3, mdd.getDofIndex( j, faceIndexesK0[ 1 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 1 ], 1 ) );
                matrixRow.setElement( j * 11 +  4, mdd.getDofIndex( j, faceIndexesK1[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  5, mdd.getDofIndex( j, faceIndexesK1[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 3 ], 3 ) );
                matrixRow.setElement( j * 11 +  6, mdd.getDofIndex( j, faceIndexesK0[ 2 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 2 ], 2 ) );
                matrixRow.setElement( j * 11 +  7, mdd.getDofIndex( j, faceIndexesK0[ 3 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 3 ], 3 ) );
                matrixRow.setElement( j * 11 +  8, mdd.getDofIndex( j, faceIndexesK1[ 4 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK1[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 +  9, mdd.getDofIndex( j, faceIndexesK1[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 1 ], E, 5, faceIndexesK0[ 5 ], 5 ) +
                                                                                             coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK1[ 4 ], 4 ) );
                matrixRow.setElement( j * 11 + 10, mdd.getDofIndex( j, faceIndexesK0[ 5 ] ), coeff::A_ijKEF( mdd, i, j, cellIndexes[ 0 ], E, 4, faceIndexesK0[ 5 ], 5 ) );
            }
        }
    }
};

} // namespace TNL::MHFEM
