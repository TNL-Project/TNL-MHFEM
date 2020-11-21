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

template< typename Array, typename NDArray >
void
setInitialCondition_fuck_you_nvcc( const int i, const Array & sourceArray, NDArray & localArray )
{
    using IndexType = typename Array::IndexType;
    using DeviceType = typename Array::DeviceType;

    const auto source_view = sourceArray.getConstView();
    auto view = localArray.getView();

    TNL::Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, source_view.getSize(),
        [=] __cuda_callable__ ( IndexType K ) mutable {
            view( i, K ) = source_view[ K ];
    });
}
template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          typename MassMatrix >
    template< typename StdVector >
void
BaseModel< Mesh, Real, NumberOfEquations, MassMatrix >::
setInitialCondition( const int i, const StdVector & vector )
{
    if( (IndexType) vector.size() != numberOfCells )
        throw std::length_error( "wrong vector length for the initial condition: expected " + std::to_string(numberOfCells) + " elements, got "
                                 + std::to_string(vector.size()));
    using Array = TNL::Containers::Array< RealType, DeviceType, IndexType >;
    Array deviceArray( vector );
    setInitialCondition_fuck_you_nvcc( i, deviceArray, this->Z_iK );
}

} // namespace mhfem
