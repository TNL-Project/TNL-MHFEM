#pragma once

#include "BoundaryConditions.h"
#include "../lib_general/mesh_helpers.h"

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
    const IndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();

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
        const int numCells = getCellsForFace( mesh, E, cellIndexes );
        const IndexType & K = cellIndexes[ 0 ];

        tnlAssert( numCells == 1,
                   cerr << "assertion numCells == 1 failed" << endl
                        << "E = " << E << endl
                        << "K0 = " << cellIndexes[ 0 ] << endl
                        << "K1 = " << cellIndexes[ 1 ] << endl; );

        // prepare face indexes
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );
        const int e = getLocalIndex( faceIndexes, E );

        // set right hand side value
        RealType bValue = - static_cast<const ModelImplementation*>(this)->getNeumannValue( mesh, i, E, time ) * getFaceSurface( mesh, E );

        bValue += mdd->w_iKe( i, K, e );
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            bValue += MeshDependentDataType::MassMatrix::b_ijKe( *mdd, i, j, K, e ) * mdd->R_iK( j, K );
        }
        b[ indexRow ] = bValue;

        // set non-zero elements
        for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ ) {
            for( int f = 0; f < mdd->FacesPerCell; f++ ) {
                matrixRow.setElement( j * mdd->FacesPerCell + f, mdd->getDofIndex( j, faceIndexes[ f ] ), coeff::A_ijKEF( *mdd, i, j, K, E, e, faceIndexes[ f ], f ) );
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
    const IndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();
    return dirichletTags[ i * faces + face ];
}

} // namespace mhfem
