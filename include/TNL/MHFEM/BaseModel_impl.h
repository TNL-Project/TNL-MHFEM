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

    if( ! b.setSize( n * n * numberOfCells * MassMatrix::size ) )
        return false;
    if( ! R1.setSize( n * n * numberOfCells * FacesPerCell ) )
        return false;
    if( ! R2.setSize( n * numberOfCells ) )
        return false;

    return true;
}

} // namespace mhfem
