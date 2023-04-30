#pragma once

#include <TNL/Containers/NDArray.h>
#include <TNL/Logger.h>

#include "MassMatrix.h"
#include "mesh_helpers.h"

namespace mhfem
{

enum class AdvectionDiscretization {
    explicit_upwind,
    implicit_upwind,
    implicit_trace
};

template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          MassLumping massLumping >
struct DefaultArrayTypes
{
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;

    using MassMatrix = mhfem::MassMatrix< typename MeshType::Cell, massLumping >;

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
    template< typename NDArray >
    using HostNDArray = TNL::Containers::NDArray< RealType,
                                                  typename NDArray::SizesHolderType,
                                                  typename NDArray::PermutationType,
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
          MassLumping massLumping = MassLumping::enabled,
          AdvectionDiscretization advection = AdvectionDiscretization::explicit_upwind,
          typename ArrayTypes_ = DefaultArrayTypes< Mesh, Real, NumberOfEquations, massLumping > >
class BaseModel
{
public:
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = typename MeshType::GlobalIndexType;
    using ArrayTypes = ArrayTypes_;

    static constexpr mhfem::AdvectionDiscretization AdvectionDiscretization = advection;

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

    void allocate( const MeshType & mesh )
    {
        numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();
        numberOfFaces = mesh.template getEntitiesCount< typename Mesh::Face >();

        Z_iF.setSizes( 0, numberOfFaces );
        Z_iK.setSizes( 0, numberOfCells );

        N_ijK.setSizes( 0, 0, numberOfCells );
        u_ijKe.setSizes( 0, 0, numberOfCells, 0 );
        m_iK.setSizes( 0, numberOfCells );
        // NOTE: only for D isotropic (represented by scalar value)
        //D.setSize( NumberOfEquations * d * NumberOfEquations * d * cells );
        D_ijK.setSizes( 0, 0, numberOfCells );
        w_iKe.setSizes( 0, numberOfCells, 0 );
        a_ijKe.setSizes( 0, 0, numberOfCells, 0 );
        r_ijK.setSizes( 0, 0, numberOfCells );
        f_iK.setSizes( 0, numberOfCells );

        v_iKe.setSizes( 0, numberOfCells, 0 );
        m_iE_upw.setSizes( 0, numberOfFaces );
        if constexpr( AdvectionDiscretization == AdvectionDiscretization::explicit_upwind )
            Z_ijE_upw.setSizes( 0, 0, numberOfFaces );

        b_ijK_storage.setSizes( 0, 0, numberOfCells, 0 );
        R_ijKe.setSizes( 0, 0, numberOfCells, 0 );
        R_iK.setSizes( 0, numberOfCells );
    }

    template< typename DistributedHostMeshType >
    static std::size_t estimateMemoryDemands( const DistributedHostMeshType & mesh )
    {
        const auto & localMesh = mesh.getLocalMesh();
        const std::size_t cells = localMesh.template getEntitiesCount< typename MeshType::Cell >();
        const std::size_t faces = localMesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();

        std::size_t mdd_size =
            // Z_iF
            + NumberOfEquations * faces
            // Z_iK
            + NumberOfEquations * cells
            // N_ijK
            + NumberOfEquations * NumberOfEquations * cells
            // u_ijKe
            + NumberOfEquations * NumberOfEquations * cells * FacesPerCell
            // m_iK
            + NumberOfEquations * cells
            // D_ijK  NOTE: only for D isotropic (represented by scalar value)
            + NumberOfEquations * NumberOfEquations * cells
            // w_iKe
            + NumberOfEquations * cells * FacesPerCell
            // a_ijKe
            + NumberOfEquations * NumberOfEquations * cells * FacesPerCell
            // r_ijK
            + NumberOfEquations * NumberOfEquations * cells
            // f_iK
            + NumberOfEquations * cells
            // v_iKe
            + NumberOfEquations * cells * FacesPerCell
            // m_iE_upw
            + NumberOfEquations * faces
            // b_ijK_storage
            + NumberOfEquations * NumberOfEquations * cells * MassMatrix::size
            // R_ijKe
            + NumberOfEquations * NumberOfEquations * cells * FacesPerCell
            // R_iK
            + NumberOfEquations * cells
        ;

        if constexpr( AdvectionDiscretization == AdvectionDiscretization::explicit_upwind )
            // Z_ijE_upw
            mdd_size += NumberOfEquations * NumberOfEquations * faces;

        mdd_size *= sizeof(RealType);
        return mdd_size;
    }

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

template< typename Array, typename NDArray >
void
setInitialCondition_fuck_you_nvcc( const int i, const Array & sourceArray, NDArray & localArray )
{
    using IndexType = typename Array::IndexType;
    using DeviceType = typename Array::DeviceType;

    const auto source_view = sourceArray.getConstView();
    auto view = localArray.getView();

    TNL::Algorithms::parallelFor< DeviceType >( 0, source_view.getSize(),
        [=] __cuda_callable__ ( IndexType K ) mutable {
            view( i, K ) = source_view[ K ];
    });
}
template< typename Mesh,
          typename Real,
          int NumberOfEquations,
          MassLumping massLumping,
          AdvectionDiscretization advection,
          typename ArrayTypes >
    template< typename StdVector >
void
BaseModel< Mesh, Real, NumberOfEquations, massLumping, advection, ArrayTypes >::
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
