#pragma once

#include <TNL/Containers/StaticArray.h>

#include "BoundaryConditions.h"
#include "../lib_general/mesh_helpers.h"
#include "SecondaryCoefficients.h"
#include "BoundaryConditionsStorage.h"

namespace mhfem
{

template< typename Mesh, typename MeshDependentData >
struct AdvectiveRowSetter
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int i,
                        const IndexType K,
                        const IndexType E,
                        const int e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;
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


        for( LocalIndex j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( LocalIndex g = 0; g < MeshDependentData::FacesPerCell; g++ ) {
                const LocalIndex f = localFaceIndexes[ g ];
                matrixRow.setElement( j * MeshDependentData::FacesPerCell + g,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }

#ifndef NDEBUG
    int errors = 0;
    for( int c = 1; c < MeshDependentData::FacesPerCell * MeshDependentData::NumberOfEquations; c++ )
        if( matrixRow.getElementColumn( c - 1 ) >= matrixRow.getElementColumn( c ) ) {
#ifndef __CUDA_ARCH__
            std::cerr << "error: E = " << E << ", c = " << c << ", row = " << matrixRow << std::endl;
#endif
            errors += 1;
        }
    TNL_ASSERT( errors == 0,
                std::cerr << "count of wrong rows: " << errors << std::endl; );
#endif
    }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
struct AdvectiveRowSetter< TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, MeshDependentData >
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int i,
                        const IndexType K,
                        const IndexType E,
                        const int e )
    {
        using coeff = SecondaryCoefficients< MeshDependentData >;

        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            for( int f = 0; f < MeshDependentData::FacesPerCell; f++ ) {
                matrixRow.setElement( j * MeshDependentData::FacesPerCell + f,
                                      mdd.getDofIndex( j, faceIndexes[ f ] ),
                                      coeff::A_ijKEF( mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
};


template< typename Mesh, typename MeshDependentData >
struct FluxRowSetter
{
    template< typename MatrixRow, typename FaceIndexes, typename IndexType >
    __cuda_callable__
    static void setRow( MatrixRow & matrixRow,
                        const MeshDependentData & mdd,
                        const FaceIndexes & faceIndexes,
                        const int i,
                        const IndexType K,
                        const IndexType E,
                        const int e )
    {
        AdvectiveRowSetter< Mesh, MeshDependentData >::setRow( matrixRow, mdd, faceIndexes, i, K, E, e );

        // modify the diagonal elements in each j-block
        // TODO: the effect of u_ij and a_ij in boundary conditions is still very experimental!
        for( int j = 0; j < MeshDependentData::NumberOfEquations; j++ ) {
            const auto value = matrixRow.getElementValue( j * MeshDependentData::FacesPerCell + e );
            matrixRow.setElement( j * MeshDependentData::FacesPerCell + e,
                                  mdd.getDofIndex( j, E ),
                                  value - mdd.u_ijKe( i, j, K, e ) - mdd.a_ijKe( i, j, K, e ) );
        }
    }
};


template< typename MeshDependentData,
          typename BoundaryModel >
bool
BoundaryConditions< MeshDependentData, BoundaryModel >::
init( const TNL::Config::ParameterContainer & parameters,
      const MeshType & mesh )
{
    const IndexType numberOfFaces = mesh.template getEntitiesCount< typename MeshType::Face >();

    const TNL::String fname = parameters.getParameter< TNL::String >( "boundary-conditions-file" );

    BoundaryConditionsStorage< RealType > storage;
    storage.load( fname );

    if( MeshDependentDataType::NumberOfEquations * numberOfFaces != storage.dofSize ) {
        std::cerr << "Wrong dofSize in BoundaryConditionsStorage loaded from file " << fname << ". Expected " << numberOfFaces
             << " elements, got " << storage.dofSize << "." << std::endl;
        return false;
    }

    tags = storage.tags;
    values = storage.values;
    dirichletValues = storage.dirichletValues;

    return true;
}

template< typename MeshDependentData,
          typename BoundaryModel >
    template< typename MeshOrdering >
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
reorderBoundaryConditions( const MeshOrdering & meshOrdering )
{
    typename TagArrayType::ViewType aux_tags;
    typename ValueArrayType::ViewType aux_values, aux_dirValues;
    const IndexType faces = tags.getSize() / MeshDependentData::NumberOfEquations;
    for( int i = 0; i < MeshDependentData::NumberOfEquations; i++ ) {
        // TODO: this depends on the specific layout of dofs, general reordering of NDArray is needed
        aux_tags.bind( tags.getData() + i * faces, faces );
        aux_values.bind( values.getData() + i * faces, faces );
        aux_dirValues.bind( dirichletValues.getData() + i * faces, faces );
        meshOrdering.template reorderVector< MeshType::getMeshDimension() - 1 >( aux_tags );
        meshOrdering.template reorderVector< MeshType::getMeshDimension() - 1 >( aux_values );
        meshOrdering.template reorderVector< MeshType::getMeshDimension() - 1 >( aux_dirValues );
    }
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
    template< typename Matrix, typename Vector >
__cuda_callable__
void
BoundaryConditions< MeshDependentData, BoundaryModel >::
setMatrixElements( const MeshType & mesh,
                   const MeshDependentDataType & mdd,
                   const IndexType E,
                   const int i,
                   const RealType time,
                   const RealType tau,
                   Matrix & matrix,
                   Vector & b ) const
{
    TNL_ASSERT_TRUE( isBoundaryFace( mesh, E ), "" );

    const IndexType indexRow = mdd.getDofIndex( i, E );

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    const IndexType faces = mesh.template getEntitiesCount< typename MeshType::Face >();
    const BoundaryConditionsType type = tags[ i * faces + E ];

    switch( type ) {
        // fixed-value (Dirichlet) boundary condition
        case BoundaryConditionsType::FixedValue:
            matrixRow.setElement( 0, indexRow, 1.0 );
            b[ indexRow ] = getDirichletValue( mesh, i, E, time, tau );
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

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set right hand side value
            const auto& entity = mesh.template getEntity< typename MeshType::Face >( E );
            RealType bValue = - getNeumannValue( mesh, i, E, time, tau ) * getEntityMeasure( mesh, entity );

            bValue += mdd.w_iKe( i, K, e );
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                bValue += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
            }
            b[ indexRow ] = bValue;

            // set non-zero elements
            FluxRowSetter< MeshType, MeshDependentDataType >::
                setRow( matrixRow,
                        mdd,
                        faceIndexes,
                        i, K, E, e );
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

            // prepare face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );
            const int e = getLocalIndex( faceIndexes, E );

            // set right hand side value
            RealType bValue = 0;

            bValue += mdd.w_iKe( i, K, e );
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
                bValue += MeshDependentDataType::MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.R_iK( j, K );
            }
            b[ indexRow ] = bValue;

            // set non-zero elements
            AdvectiveRowSetter< MeshType, MeshDependentDataType >::
                setRow( matrixRow,
                        mdd,
                        faceIndexes,
                        i, K, E, e );
            break;
        }

        default:
            TNL_ASSERT_TRUE( false, "unknown boundary condition type was encountered" );
            break;
    }
}

} // namespace mhfem
