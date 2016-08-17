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
                          const IndexType & E,
                          const typename MeshType::Face & entity,
                          const int & i ) const
{
    if( this->isDirichletBoundary( mesh, i, entity ) )
        return 1;
    return MeshDependentDataType::FacesPerCell * MeshDependentDataType::NumberOfEquations;
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
    template< typename DofVectorPointer, typename Vector, typename Matrix >
__cuda_callable__
void
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
setMatrixElements( DofVectorPointer & u,
                   const typename MeshType::Face & entity,
                   const RealType & time,
                   const RealType & tau,
                   const int & i,
                   Matrix & matrix,
                   Vector & b ) const
{
    const IndexType E = entity.getIndex();
    const MeshType & mesh = entity.getMesh();
    const IndexType indexRow = i * mesh.template getEntitiesCount< typename MeshType::Face >() + E;

    typename Matrix::MatrixRow matrixRow = matrix.getRow( indexRow );

    // Dirichlet boundary
    if( isDirichletBoundary( mesh, i, entity ) ) {
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
                   std::cerr << "assertion numCells == 1 failed" << std::endl
                             << "E = " << E << std::endl
                             << "K0 = " << cellIndexes[ 0 ] << std::endl
                             << "K1 = " << cellIndexes[ 1 ] << std::endl; );

        // prepare face indexes
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );
        const int e = getLocalIndex( faceIndexes, E );

        // set right hand side value
        RealType bValue = - static_cast<const ModelImplementation*>(this)->getNeumannValue( mesh, i, E, time ) * getFaceSurface( mesh, entity );

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
isNeumannBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const
{
    if( ! face.isBoundaryEntity() )
        return false;
    return ! isDirichletBoundary( mesh, i, face );
}

template< typename Mesh,
          typename MeshDependentData,
          typename ModelImplementation >
__cuda_callable__
bool
BoundaryConditions< Mesh, MeshDependentData, ModelImplementation >::
isDirichletBoundary( const MeshType & mesh, const int & i, const typename Mesh::Face & face ) const
{
    if( ! face.isBoundaryEntity() )
        return false;
    const IndexType faces = mesh.template getEntitiesCount< typename Mesh::Face >();
    return dirichletTags[ i * faces + face.getIndex() ];
}

} // namespace mhfem
