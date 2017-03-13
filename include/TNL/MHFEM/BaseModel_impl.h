#pragma once

#include "BaseModel.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          typename Index,
          int NumberOfEquations,
          typename ModelImplementation,
          typename MassMatrix >
bool
BaseModel< Mesh, Real, Index, NumberOfEquations, ModelImplementation, MassMatrix >::
allocate( const MeshType & mesh )
{
    numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();
    numberOfFaces = mesh.template getEntitiesCount< typename Mesh::Face >();

    if( ! Z_iK.setSizes( 0, numberOfCells ) )
        return false;

    if( ! N_ijK.setSizes( 0, 0, numberOfCells ) )
        return false;
    if( ! u_ijKe.setSizes( 0, 0, numberOfCells, 0 ) )
        return false;
    if( ! m_iK.setSizes( 0, numberOfCells ) )
        return false;
    // NOTE: only for D isotropic (represented by scalar value)
//    if( ! D.setSize( n * d * n * d * cells ) )
//        return false;
    if( ! D_ijK.setSizes( 0, 0, numberOfCells ) )
        return false;
    if( ! w_iKe.setSizes( 0, numberOfCells, 0 ) )
        return false;
    if( ! a_ijKe.setSizes( 0, 0, numberOfCells, 0 ) )
        return false;
    if( ! r_ijK.setSizes( 0, 0, numberOfCells ) )
        return false;
    if( ! f_iK.setSizes( 0, numberOfCells ) )
        return false;

    if( ! v_iKe.setSizes( 0, numberOfCells, 0 ) )
        return false;
    if( ! m_iE_upw.setSizes( 0, numberOfFaces ) )
        return false;
    if( ! Z_ijE_upw.setSizes( 0, 0, numberOfFaces ) )
        return false;

    if( ! b_ijK_storage.setSizes( 0, 0, numberOfCells, 0 ) )
        return false;
    if( ! R_ijKe.setSizes( 0, 0, numberOfCells, 0 ) )
        return false;
    if( ! R_iK.setSizes( 0, numberOfCells ) )
        return false;

    return true;
}

template< typename Mesh,
          typename Real,
          typename Index,
          int NumberOfEquations,
          typename ModelImplementation,
          typename MassMatrix >
    template< typename MeshOrdering >
bool
BaseModel< Mesh, Real, Index, NumberOfEquations, ModelImplementation, MassMatrix >::
reorderDofs( const MeshOrdering & meshOrdering, bool inverse )
{
    bool status = true;
    DofVectorType Z;
    for( int i = 0; i < NumberOfEquations; i++ ) {
        // TODO: this depends on the specific layout of Z_iK, general reordering of NDArray is needed
        Z.bind( Z_iK.getStorageArray().getData() + i * numberOfCells, numberOfCells );
        status &= meshOrdering.reorder_cells( Z, inverse );
    }
    if( ! status )
        std::cerr << "Failed to reorder the DOF vector." << std::endl;
    return status;
}

} // namespace mhfem
