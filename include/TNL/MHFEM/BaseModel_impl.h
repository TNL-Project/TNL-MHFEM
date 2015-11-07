#pragma once

#include "BaseModel.h"

namespace mhfem
{

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename ModelImplementation >
bool
BaseModel< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, ModelImplementation >::
allocate( const MeshType & mesh )
{
    numberOfCells = mesh.getNumberOfCells();
    numberOfFaces = mesh.template getNumberOfFaces< 1, 1 >();

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
    if( ! w.setSize( n * numberOfCells * facesPerCell ) )
        return false;
    if( ! f.setSize( n * numberOfCells ) )
        return false;

    if( ! m_upw.setSize( n * numberOfFaces ) )
        return false;
    // TODO check this
    if( ! b.setSize( n * n * numberOfCells * facesPerCell ) )
        return false;

    if( ! R1.setSize( n * n * numberOfCells * facesPerCell ) )
        return false;
    if( ! R2.setSize( n * numberOfCells ) )
        return false;

    return true;
}

} // namespace mhfem
