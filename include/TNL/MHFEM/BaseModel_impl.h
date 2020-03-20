#pragma once

#include "BaseModel.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          typename MassMatrix >
void
BaseModel< Mesh, Real, NumberOfEquations, MassMatrix >::
allocate( const MeshType & mesh )
{
    numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();
    numberOfFaces = mesh.template getEntitiesCount< typename Mesh::Face >();

    Z_iF.setSizes( 0, numberOfFaces );
    Z_iK.setSizes( 0, numberOfCells );

    N_ijK.setSizes( 0, 0, numberOfCells );
    u_ijKe.setSizes( 0, 0, numberOfCells, 0 );
    m_iK.setSizes( 0, numberOfCells );
    // NOTE: only for D isotropic (represented by scalar value)
//    D.setSize( n * d * n * d * cells );
    D_ijK.setSizes( 0, 0, numberOfCells );
    w_iKe.setSizes( 0, numberOfCells, 0 );
    a_ijKe.setSizes( 0, 0, numberOfCells, 0 );
    r_ijK.setSizes( 0, 0, numberOfCells );
    f_iK.setSizes( 0, numberOfCells );

    v_iKe.setSizes( 0, numberOfCells, 0 );
    m_iE_upw.setSizes( 0, numberOfFaces );
    Z_ijE_upw.setSizes( 0, 0, numberOfFaces );

    b_ijK_storage.setSizes( 0, 0, numberOfCells, 0 );
    R_ijKe.setSizes( 0, 0, numberOfCells, 0 );
    R_iK.setSizes( 0, numberOfCells );
}

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          typename MassMatrix >
    template< typename MeshOrdering >
void
BaseModel< Mesh, Real, NumberOfEquations, MassMatrix >::
reorderDofs( const MeshOrdering & meshOrdering, bool inverse )
{
    DofVectorType Z;
    for( int i = 0; i < NumberOfEquations; i++ ) {
        // TODO: this depends on the specific layout of Z_iK, general reordering of NDArray is needed
        Z.bind( Z_iK.getStorageArray().getData() + i * numberOfCells, numberOfCells );
        meshOrdering.template reorderVector< Mesh::getMeshDimension() >( Z, inverse );
    }
}

} // namespace mhfem
