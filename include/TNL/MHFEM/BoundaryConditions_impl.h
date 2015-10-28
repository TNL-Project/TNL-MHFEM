#pragma once

#include "BoundaryConditions.h"
#include "../mesh_helpers.h"

namespace mhfem
{

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
void
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
bindMeshDependentData( MeshDependentDataType* mdd )
{
    this->mdd = mdd;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexDof,
                          const CoordinatesType & coordinates ) const
{
    const IndexType faces = mesh.template getNumberOfFaces< 1, 1 >();

    // TODO: completely depends on the indexation of vectors in MeshDependentData, probably should be generalized
    const IndexType E = indexDof % faces;
    const int i = indexDof / faces;

    if( this->isDirichletBoundary( mesh, i, E ) )
        return 1;
    return 4 * MeshDependentDataType::NumberOfEquations;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
    template< typename Vector, typename Matrix >
__cuda_callable__
void
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
updateLinearSystem( const RealType & time,
                    const MeshType & mesh,
                    const IndexType & indexRow,
                    const CoordinatesType & coordinates,
                    Vector & u,
                    Vector & b,
                    Matrix & matrix ) const
{
    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    const IndexType E = mdd->indexDofToFace( indexRow );
    const int i = mdd->indexDofToEqno( indexRow );

    // Dirichlet boundary
    if( isDirichletBoundary( mesh, i, E ) ) {
        matrixRow.setElement( 0, indexRow, 1.0 );
        b[ indexRow ] = static_cast<const ModelImplementation*>(this)->getDirichletValue( mesh, i, E, time );
    }
    // Neumann boundary
    else {
        // for boundary faces returns only one valid cell index
        IndexType cellIndexes[ 2 ];
        int numCells = getCellsForFace( mesh, E, cellIndexes );
        const IndexType & K = cellIndexes[ 0 ];

        tnlAssert( numCells == 1,
                cerr << "assertion numCells == 1 failed" << endl; );

        // prepare face indexes
        IndexType faceIndexes[ 4 ];
        getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

        // find local index of face E
        // TODO: simplify?
        int e = 0;
        for( int xxx = 0; xxx < mdd->facesPerCell; xxx++ ) {
            if( faceIndexes[ xxx ] == E ) {
                e = xxx;
                break;
            }
        }

        // set right hand side value
        RealType bValue = - static_cast<const ModelImplementation*>(this)->getNeumannValue( mesh, i, E, time );
        bValue += mdd->w_iKe( i, K, e );
        for( int j = 0; j < mdd->n; j++ ) {
            bValue += mdd->b_ijKe( i, j, K, e ) * mdd->R_iK( j, K );
        }
        b[ indexRow ] = bValue;

        // set non-zero elements
        // FIXME: on the Neumann boundary, either q_KE = \rho_K u_KE or q_KE = \rho_E^upw u_KE,
        // but the getValue method returns only B_KEF
        for( int j = 0; j < mdd->n; j++ ) {
            for( int f = 0; f < mdd->facesPerCell; f++ ) {
                matrixRow.setElement( j * mdd->facesPerCell + f, mdd->getDofIndex( j, faceIndexes[ f ] ), getValue( i, j, E, e, faceIndexes[ f ], f, K ) );
            }
        }
    }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
typename MeshDependentData::RealType
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
getValue( const int & i,
          const int & j,
          const IndexType & E,
          const int & e,
          const IndexType & F,
          const int & f,
          const IndexType & K ) const
{
    RealType value = 0.0;
    for( int xxx = 0; xxx < mdd->n; xxx++ ) {
        value -= mdd->b_ijKe( i, xxx, K, e ) * mdd->R_ijKe( xxx, j, K, f );
        if( xxx == j && E == F )
            value += mdd->b_ijKe( i, xxx, K, e );
    }
    return value;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
isNeumannBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const
{
    if( ! isBoundaryFace( mesh, face ) )
        return false;
    return ! isDirichletBoundary( mesh, i, face );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData, ModelImplementation >::
isDirichletBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const
{
    if( ! isBoundaryFace( mesh, face ) )
        return false;
    return dirichletTags[ i * mesh.template getNumberOfFaces< 1, 1 >() + face ];
}

} // namespace mhfem
