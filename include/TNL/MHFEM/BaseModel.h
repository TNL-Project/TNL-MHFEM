#pragma once

#include <TNL/Object.h>

#include "MassMatrix.h"
#include "../lib_general/FacesPerCell.h"
#include "../lib_general/ndarray.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          typename Index,
          // This can't be taken from ModelImplementation as a compile-time constant,
          // because it inherits from BaseModel so it is not known at this time.
          int NumberOfEquations,
          typename ModelImplementation,
          // this is not a non-typename parameter due to deficiency in TypeResolver
//          MassLumping lumping = MassLumping::enabled >
          typename MassMatrix_ = mhfem::MassMatrix< typename Mesh::Cell, MassLumping::enabled > >
class BaseModel :
    public TNL::Object
{
public:
    // TODO: for some arcane reason 'using ModelImplementation::MeshType' does not work, but 'IndexType n = ModelImplementation::NumberOfEquations' does
    // (using typedefs from children would greatly simplify the parametrization of BaseModel)
    using MeshType = Mesh;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = Index;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

//    using MassMatrix = mhfem::MassMatrix< MeshType, lumping >;
    using MassMatrix = MassMatrix_;

    using FPC = ::FacesPerCell< typename MeshType::Cell >;
    static constexpr int FacesPerCell = FPC::value;

    // NOTE: children of BaseModel (i.e. ModelImplementation) must implement these methods
//    bool init( const tnlParameterContainer & parameters,
//               const MeshType & mesh );
//
//    __cuda_callable__
//    void
//    updateNonLinearTerms( const MeshType & mesh,
//                          const IndexType & K,
//                          const CoordinatesType & coordinates );
//
//    bool makeSnapshot( const RealType & time,
//                       const IndexType & step,
//                       const MeshType & mesh ) const;

    bool allocate( const MeshType & mesh );

    // indexing functions
    __cuda_callable__
    IndexType indexDofToFace( const IndexType & indexDof ) const
    {
//        return indexDof / n;
        return indexDof % numberOfFaces;
    }

    __cuda_callable__
    IndexType indexDofToEqno( const IndexType & indexDof ) const
    {
//        return indexDof % n;
        return indexDof / numberOfFaces;
    }

    __cuda_callable__
    IndexType getDofIndex( const int & i, const IndexType & indexFace ) const
    {
//        return n * indexFace + i;
        return i * numberOfFaces + indexFace;
    }

    // needed in makeSnapshot
    __cuda_callable__
    IndexType getCellDofIndex( const int & i, const IndexType & indexCell ) const
    {
//        return n * indexCell + i;
        return i * numberOfCells + indexCell;
    }

    // accessor for auxiliary dofs
    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                       std::index_sequence< 0, 1 >,  // i, K  (host)
                       std::index_sequence< 0, 1 >,  // i, K  (cuda)
                       DeviceType >
                Z_iK;

    // accessors for coefficients
    // TODO: optimize for models that don't use all coefficients
    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                       std::index_sequence< 0, 2, 1 >,  // i, K, j  (host)
                       std::index_sequence< 0, 1, 2 >,  // i, j, K  (cuda)
                       DeviceType >
                N_ijK;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                       std::index_sequence< 0, 2, 1, 3 >,  // i, K, j, e  (host)
                       std::index_sequence< 0, 1, 3, 2 >,  // i, j, e, K  (cuda)
                       DeviceType >
                u_ijKe;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                       std::index_sequence< 0, 1 >,  // i, K  (host)
                       std::index_sequence< 0, 1 >,  // i, K  (cuda)
                       DeviceType >
                m_iK;

    // NOTE: only for D isotropic (represented by scalar value)
    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                       std::index_sequence< 0, 2, 1 >,  // i, K, j  (host)
                       std::index_sequence< 0, 1, 2 >,  // i, j, K  (cuda)
                       DeviceType >
                D_ijK;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
                       std::index_sequence< 0, 1, 2 >,  // i, K, e  (host)
                       std::index_sequence< 0, 2, 1 >,  // i, e, K  (cuda)
                       DeviceType >
                w_iKe;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                       std::index_sequence< 0, 2, 1, 3 >,  // i, K, j, e  (host)
                       std::index_sequence< 0, 1, 3, 2 >,  // i, j, e, K  (cuda)
                       DeviceType >
                a_ijKe;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, K
                       std::index_sequence< 0, 2, 1 >,  // i, K, j  (host)
                       std::index_sequence< 0, 1, 2 >,  // i, j, K  (cuda)
                       DeviceType >
                r_ijK;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                       std::index_sequence< 0, 1 >,  // i, K  (host)
                       std::index_sequence< 0, 1 >,  // i, K  (cuda)
                       DeviceType >
                f_iK;


    // coefficients specific to the MHFEM scheme

    // conservative velocities for upwind: \vec v_i = - \sum_j \mat D_ij \grad Z_j + \vec w_i
    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0, FacesPerCell >,  // i, K, e
                       std::index_sequence< 0, 1, 2 >,  // i, K, e  (host)
                       std::index_sequence< 0, 2, 1 >,  // i, e, K  (cuda)
                       DeviceType >
                v_iKe;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, E
                       std::index_sequence< 0, 1 >,  // i, E  (host)
                       std::index_sequence< 0, 1 >,  // i, E  (cuda)
                       DeviceType >
                m_iE_upw;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0 >,  // i, j, E
                       // NOTE: this must match the manual indexing in the UpwindZ class
                       std::index_sequence< 0, 1, 2 >,  // i, j, E  (host)
                       std::index_sequence< 0, 1, 2 >,  // i, j, E  (cuda)
                       DeviceType >
                Z_ijE_upw;

    // values with different 's' represent the local matrix b_ijK
    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, MassMatrix::size >,  // i, j, K, s
                       std::index_sequence< 0, 2, 1, 3 >,  // i, K, j, s  (host)
                       std::index_sequence< 0, 1, 3, 2 >,  // i, j, s, K  (cuda)
                       DeviceType >
                b_ijK_storage;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, NumberOfEquations, 0, FacesPerCell >,  // i, j, K, e
                       std::index_sequence< 0, 2, 1, 3 >,  // i, K, j, e  (host)
                       std::index_sequence< 0, 1, 3, 2 >,  // i, j, e, K  (cuda)
                       DeviceType >
                R_ijKe;

    sndarray::NDArray< RealType,
                       sndarray::SizesHolder< IndexType, NumberOfEquations, 0 >,  // i, K
                       std::index_sequence< 0, 1 >,  // i, K  (host)
                       std::index_sequence< 0, 1 >,  // i, K  (cuda)
                       DeviceType >
                R_iK;

//protected:
    
    // FIXME: nasty hack to pass tau to LocalUpdaters
    RealType current_tau;
    // FIXME: nasty hack to pass time to CompositionalModel::r_X
    RealType current_time;

    // FIXME: needed only to pass dofs to LocalUpdaters::update_v
    DofVectorType Z_iF;

protected:
    // number of entities of the mesh for which the vectors are allocated
    IndexType numberOfCells = 0;
    IndexType numberOfFaces = 0;
};

} // namespace mhfem

#include "BaseModel_impl.h"
