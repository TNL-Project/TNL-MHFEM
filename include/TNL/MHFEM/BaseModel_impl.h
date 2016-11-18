#pragma once

#include "BaseModel.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          typename Index,
          typename ModelImplementation,
          typename MassMatrix >
bool
BaseModel< Mesh, Real, Index, ModelImplementation, MassMatrix >::
allocate( const MeshType & mesh )
{
    numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();
    numberOfFaces = mesh.template getEntitiesCount< typename Mesh::Face >();

    if( ! Z.setSize( n * numberOfCells ) )
        return false;

    if( ! N.setSize( n * n * numberOfCells ) )
        return false;
    if( ! m.setSize( n * numberOfCells ) )
        return false;
    // NOTE: only for D isotropic (represented by scalar value)
//    if( ! D.setSize( n * d * n * d * cells ) )
//        return false;
    if( ! D.setSize( n * n * numberOfCells ) )
        return false;
    if( ! w.setSize( n * numberOfCells * FacesPerCell ) )
        return false;
    if( ! f.setSize( n * numberOfCells ) )
        return false;

    if( ! m_upw.setSize( n * numberOfFaces ) )
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
