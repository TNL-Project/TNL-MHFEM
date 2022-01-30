#pragma once

#include <TNL/Containers/NDArray.h>
#include <TNL/Logger.h>

#include "MassMatrix.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          typename MassLumpingTag >
struct DefaultArrayTypes
{
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;

    using MassMatrix = mhfem::MassMatrix< typename MeshType::Cell, MassLumpingTag::lumping >;

    using FPC = ::FacesPerCell< typename MeshType::Cell >;
    static constexpr int FacesPerCell = FPC::value;
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

    // main and auxiliary dofs
    using Z_iF_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, F
                 // NOTE: order enforced by the DistributedMeshSynchronizer
                 std::index_sequence< 1, 0 >,    // F, i  (host)
                 std::index_sequence< 1, 0 > >;  // F, i  (cuda)
    using Z_iK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                 std::index_sequence< 0, 1 >,    // i, K  (host)
                 std::index_sequence< 0, 1 > >;  // i, K  (cuda)

    // accessors for coefficients
    using N_ijK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                 std::index_sequence< 0, 2, 1 >,    // i, K, j  (host)
                 std::index_sequence< 0, 1, 2 > >;  // i, j, K  (cuda)
    using u_ijKe_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                 std::index_sequence< 0, 2, 1, 3 >,    // i, K, j, e  (host)
                 std::index_sequence< 0, 1, 3, 2 > >;  // i, j, e, K  (cuda)
    using m_iK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                 std::index_sequence< 0, 1 >,    // i, K  (host)
                 std::index_sequence< 0, 1 > >;  // i, K  (cuda)
    // NOTE: only for D isotropic (represented by scalar value)
    using D_ijK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                 std::index_sequence< 0, 2, 1 >,    // i, K, j  (host)
                 std::index_sequence< 0, 1, 2 > >;  // i, j, K  (cuda)
    using w_iKe_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
                 std::index_sequence< 0, 1, 2 >,    // i, K, e  (host)
                 std::index_sequence< 0, 2, 1 > >;  // i, e, K  (cuda)
    using a_ijKe_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                 std::index_sequence< 0, 2, 1, 3 >,    // i, K, j, e  (host)
                 std::index_sequence< 0, 1, 3, 2 > >;  // i, j, e, K  (cuda)
    using r_ijK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                 std::index_sequence< 0, 2, 1 >,    // i, K, j  (host)
                 std::index_sequence< 0, 1, 2 > >;  // i, j, K  (cuda)
    using f_iK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                 std::index_sequence< 0, 1 >,    // i, K  (host)
                 std::index_sequence< 0, 1 > >;  // i, K  (cuda)

    // coefficients specific to the MHFEM scheme

    // conservative velocities for upwind: \vec v_i = - \sum_j \mat D_ij \grad Z_j + \vec w_i
    using v_iKe_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
                 std::index_sequence< 0, 1, 2 >,    // i, K, e  (host)
                 std::index_sequence< 0, 2, 1 > >;  // i, e, K  (cuda)
    using m_iE_upw_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, E
                 // NOTE: order enforced by the DistributedMeshSynchronizer
                 std::index_sequence< 1, 0 >,    // E, i  (host)
                 std::index_sequence< 1, 0 > >;  // E, i  (cuda)
    using Z_ijE_upw_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, E
                 // NOTE: order enforced by the DistributedMeshSynchronizer
                 std::index_sequence< 2, 1, 0 >,    // E, j, i  (host)
                 std::index_sequence< 2, 1, 0 > >;  // E, j, i  (cuda)
    // values with different 's' represent the local matrix b_ijK
    using b_ijK_storage_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, MassMatrix::size >,  // i, j, K, s
                 std::index_sequence< 0, 2, 1, 3 >,    // i, K, j, s  (host)
                 std::index_sequence< 0, 1, 3, 2 > >;  // i, j, s, K  (cuda)
    using R_ijKe_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                 std::index_sequence< 0, 2, 1, 3 >,    // i, K, j, e  (host)
                 std::index_sequence< 0, 1, 3, 2 > >;  // i, j, e, K  (cuda)
    using R_iK_t =
        NDArray< TNL::Containers::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                 std::index_sequence< 0, 1 >,    // i, K  (host)
                 std::index_sequence< 0, 1 > >;  // i, K  (cuda)
};

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          // this is not a non-typename parameter due to deficiency in TypeResolver
          typename MassLumpingTag = MassLumpingEnabledTag,
          typename ArrayTypes_ = DefaultArrayTypes< Mesh, Real, NumberOfEquations, MassLumpingTag > >
class BaseModel
{
public:
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;
    using ArrayTypes = ArrayTypes_;

    using MassMatrix = typename ArrayTypes::MassMatrix;
    static constexpr int FacesPerCell = ArrayTypes::FacesPerCell;

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

    // this can be overridden in child classes which use a constant mobility coefficient
    static constexpr bool do_mobility_upwind = true;

    // this can be overridden in child classes
    static void writeProlog( TNL::Logger& logger ) {}

    void allocate( const MeshType & mesh );

    template< typename DistributedHostMeshType >
    static std::size_t estimateMemoryDemands( const DistributedHostMeshType & mesh );

    template< typename StdVector >
    void setInitialCondition( const int i, const StdVector & vector );

    // hooks
    virtual void preIterate( const RealType time, const RealType tau ) {}
    virtual void postIterate( const RealType time, const RealType tau ) {}

    // indexing wrapper method
    __cuda_callable__
    IndexType getDofIndex( const int i, const IndexType indexFace ) const
    {
        return Z_iF.getStorageIndex( i, indexFace );
    }

    __cuda_callable__
    IndexType getRowIndex( const int i, const IndexType indexFace ) const
    {
        return Z_iF.getStorageIndex( i, indexFace );
    }


    // main and auxiliary dofs
    typename ArrayTypes::Z_iF_t Z_iF;
    typename ArrayTypes::Z_iK_t Z_iK;

    // accessors for coefficients
    typename ArrayTypes::N_ijK_t N_ijK;
    typename ArrayTypes::u_ijKe_t u_ijKe;
    typename ArrayTypes::m_iK_t m_iK;
    typename ArrayTypes::D_ijK_t D_ijK;
    typename ArrayTypes::w_iKe_t w_iKe;
    typename ArrayTypes::a_ijKe_t a_ijKe;
    typename ArrayTypes::r_ijK_t r_ijK;
    typename ArrayTypes::f_iK_t f_iK;

    // coefficients specific to the MHFEM scheme
    typename ArrayTypes::v_iKe_t v_iKe;
    typename ArrayTypes::m_iE_upw_t m_iE_upw;
    typename ArrayTypes::Z_ijE_upw_t Z_ijE_upw;
    typename ArrayTypes::b_ijK_storage_t b_ijK_storage;
    typename ArrayTypes::R_ijKe_t R_ijKe;
    typename ArrayTypes::R_iK_t R_iK;

protected:
    // number of entities of the mesh for which the vectors are allocated
    IndexType numberOfCells = 0;
    IndexType numberOfFaces = 0;
};

} // namespace mhfem

#include "BaseModel_impl.h"
