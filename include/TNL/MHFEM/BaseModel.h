#pragma once

#include <TNL/Containers/NDArray.h>

#include "MassMatrix.h"
#include "../lib_general/FacesPerCell.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          // this is not a non-typename parameter due to deficiency in TypeResolver
          typename MassLumpingTag = MassLumpingEnabledTag >
class BaseModel
{
public:
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;

    using MassMatrix = mhfem::MassMatrix< typename MeshType::Cell, MassLumpingTag::lumping >;

    using FPC = ::FacesPerCell< typename MeshType::Cell >;
    static constexpr int FacesPerCell = FPC::value;

    // NOTE: children of BaseModel must implement these methods
//    bool init( const tnlParameterContainer & parameters,
//               const MeshType & mesh );
//
//    __cuda_callable__
//    void
//    updateNonLinearTerms( const MeshType & mesh,
//                          const IndexType & K,
//                          const CoordinatesType & coordinates );
//
//    bool makeSnapshot( const RealType time,
//                       const IndexType step,
//                       const MeshType & mesh,
//                       const TNL::String & outputPrefix ) const

    void allocate( const MeshType & mesh );

    template< typename MeshOrdering >
    void reorderDofs( const MeshOrdering & meshOrdering, bool inverse );

    // hooks
    virtual void preIterate( const RealType time, const RealType tau ) {}
    virtual void postIterate( const RealType time, const RealType tau ) {}

    // indexing wrapper method
    __cuda_callable__
    IndexType getDofIndex( const int i, const IndexType indexFace ) const
    {
        return Z_iF.getStorageIndex( i, indexFace );
    }

    template< typename SizesHolder,
              typename HostPermutation,
              typename CudaPermutation >
    using NDArray = TNL::Containers::NDArray< RealType,
                                              SizesHolder,
                                              std::conditional_t< std::is_same< DeviceType, TNL::Devices::Cuda >::value,
                                                                  CudaPermutation,
                                                                  HostPermutation >,
                                              DeviceType >;
    // host NDArray - intended for output/buffering only
    template< typename SizesHolder,
              typename HostPermutation >
    using HostNDArray = TNL::Containers::NDArray< RealType,
                                                  SizesHolder,
                                                  HostPermutation,
                                                  TNL::Devices::Host >;

    // main dofs
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, F
             std::index_sequence< 0, 1 >,   // i, F  (host)
             std::index_sequence< 0, 1 > >  // i, F  (cuda)
        Z_iF;

    // accessor for auxiliary dofs
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
             std::index_sequence< 0, 1 >,   // i, K  (host)
             std::index_sequence< 0, 1 > >  // i, K  (cuda)
        Z_iK;

    // accessors for coefficients
    // TODO: optimize for models that don't use all coefficients
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
             std::index_sequence< 0, 2, 1 >,   // i, K, j  (host)
             std::index_sequence< 0, 1, 2 > >  // i, j, K  (cuda)
        N_ijK;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
             std::index_sequence< 0, 2, 1, 3 >,   // i, K, j, e  (host)
             std::index_sequence< 0, 1, 3, 2 > >  // i, j, e, K  (cuda)
        u_ijKe;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
             std::index_sequence< 0, 1 >,   // i, K  (host)
             std::index_sequence< 0, 1 > >  // i, K  (cuda)
        m_iK;

    // NOTE: only for D isotropic (represented by scalar value)
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
             std::index_sequence< 0, 2, 1 >,   // i, K, j  (host)
             std::index_sequence< 0, 1, 2 > >  // i, j, K  (cuda)
        D_ijK;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
             std::index_sequence< 0, 1, 2 >,   // i, K, e  (host)
             std::index_sequence< 0, 2, 1 > >  // i, e, K  (cuda)
        w_iKe;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
             std::index_sequence< 0, 2, 1, 3 >,   // i, K, j, e  (host)
             std::index_sequence< 0, 1, 3, 2 > >  // i, j, e, K  (cuda)
        a_ijKe;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
             std::index_sequence< 0, 2, 1 >,   // i, K, j  (host)
             std::index_sequence< 0, 1, 2 > >  // i, j, K  (cuda)
        r_ijK;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
             std::index_sequence< 0, 1 >,   // i, K  (host)
             std::index_sequence< 0, 1 > >  // i, K  (cuda)
        f_iK;


    // coefficients specific to the MHFEM scheme

    // conservative velocities for upwind: \vec v_i = - \sum_j \mat D_ij \grad Z_j + \vec w_i
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
             std::index_sequence< 0, 1, 2 >,   // i, K, e  (host)
             std::index_sequence< 0, 2, 1 > >  // i, e, K  (cuda)
        v_iKe;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, E
             std::index_sequence< 0, 1 >,   // i, E  (host)
             std::index_sequence< 0, 1 > >  // i, E  (cuda)
        m_iE_upw;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, E
             // NOTE: this must match the manual indexing in the UpwindZ class
             std::index_sequence< 0, 1, 2 >,   // i, j, E  (host)
             std::index_sequence< 0, 1, 2 > >  // i, j, E  (cuda)
        Z_ijE_upw;

    // values with different 's' represent the local matrix b_ijK
    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, MassMatrix::size >,  // i, j, K, s
             std::index_sequence< 0, 2, 1, 3 >,   // i, K, j, s  (host)
             std::index_sequence< 0, 1, 3, 2 > >  // i, j, s, K  (cuda)
        b_ijK_storage;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
             std::index_sequence< 0, 2, 1, 3 >,   // i, K, j, e  (host)
             std::index_sequence< 0, 1, 3, 2 > >  // i, j, e, K  (cuda)
        R_ijKe;

    NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
             std::index_sequence< 0, 1 >,   // i, K  (host)
             std::index_sequence< 0, 1 > >  // i, K  (cuda)
        R_iK;

protected:
    // number of entities of the mesh for which the vectors are allocated
    IndexType numberOfCells = 0;
    IndexType numberOfFaces = 0;
};

} // namespace mhfem

#include "BaseModel_impl.h"
