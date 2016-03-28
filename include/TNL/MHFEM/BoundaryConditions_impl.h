#pragma once

#include "BoundaryConditions.h"
#include "../mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
void
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
bindMeshDependentData( MeshDependentDataType* mdd )
{
    this->mdd = mdd;
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
typename MeshDependentData::IndexType
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
getLinearSystemRowLength( const MeshType & mesh,
                          const IndexType & indexDof,
                          const CoordinatesType & coordinates ) const
{
    const IndexType faces = FacesCounter< MeshType >::getNumberOfFaces( mesh );

    // TODO: completely depends on the indexation of vectors in MeshDependentData, probably should be generalized
    const IndexType E = indexDof % faces;
    const int i = indexDof / faces;

    if( this->isDirichletBoundary( mesh, i, E ) )
        return 1;
    return MeshDependentDataType::FacesPerCell * MeshDependentDataType::NumberOfEquations;
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
    template< typename Vector, typename Matrix >
__cuda_callable__
void
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
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
                   cerr << "assertion numCells == 1 failed" << endl
                        << "E = " << E << endl
                        << "K0 = " << cellIndexes[ 0 ] << endl
                        << "K1 = " << cellIndexes[ 1 ] << endl; );

        // prepare face indexes
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );

        // find local index of face E
        // TODO: simplify?
        int e = 0;
        for( int xxx = 0; xxx < mdd->FacesPerCell; xxx++ ) {
            if( faceIndexes[ xxx ] == E ) {
                e = xxx;
                break;
            }
        }

        // set right hand side value
        RealType bValue = - static_cast<const ModelImplementation*>(this)->getNeumannValue( mesh, i, E, time ) * getFaceSurface( mesh, E );
        bValue += mdd->w_iKe( i, K, e );
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            SharedVectorType mass_matrix_storage( mdd->b_ijK( i, j, K ), MeshDependentDataType::MassMatrix::size );
            bValue += MeshDependentDataType::MassMatrix::get( e, mass_matrix_storage ) * mdd->R_iK( j, K );
        }
        b[ indexRow ] = bValue;

        // set non-zero elements
        // FIXME: on the Neumann boundary, either q_KE = \rho_K u_KE or q_KE = \rho_E^upw u_KE,
        // but the getValue method returns only B_KEF
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            for( int f = 0; f < mdd->FacesPerCell; f++ ) {
                matrixRow.setElement( j * mdd->FacesPerCell + f, mdd->getDofIndex( j, faceIndexes[ f ] ), hybrid::A_ijKEF( *mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
            }
        }
    }
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
isNeumannBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const
{
    if( ! isBoundaryFace( mesh, face ) )
        return false;
    return ! isDirichletBoundary( mesh, i, face );
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
isDirichletBoundary( const MeshType & mesh, const int & i, const IndexType & face ) const
{
    if( ! isBoundaryFace( mesh, face ) )
        return false;
    return dirichletTags[ i * FacesCounter< MeshType >::getNumberOfFaces( mesh ) + face ];
}

} // namespace mhfem
