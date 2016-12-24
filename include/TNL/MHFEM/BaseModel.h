#pragma once

#include <TNL/Object.h>

#include "MassMatrix.h"
#include "../lib_general/FacesPerCell.h"

namespace mhfem
{

template< typename Mesh,
          typename Real,
          typename Index,
          typename ModelImplementation,
          // this is not a non-typename parameter due to deficiency in TypeResolver
//          MassLumping lumping = MassLumping::enabled >
          typename MassMatrix_ = mhfem::MassMatrix< Mesh, MassLumping::enabled > >
class BaseModel :
    public TNL::Object
{
public:
    // TODO: for some arcane reason 'using ModelImplementation::MeshType' does not work, but 'IndexType n = ModelImplementation::NumberOfEquations' does
    // (using typedefs from children would greatly simplify the parametrization of BaseModel)
    using MeshType = Mesh;
    using CoordinatesType = typename MeshType::CoordinatesType;
    using RealType = Real;
    using DeviceType = typename MeshType::DeviceType;
    using IndexType = Index;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

//    using MassMatrix = mhfem::MassMatrix< MeshType, lumping >;
    using MassMatrix = MassMatrix_;

    using FPC = ::FacesPerCell< MeshType >;
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

    // needed in makeSnapshot, FaceAverageFunction
    __cuda_callable__
    IndexType getCellDofIndex( const int & i, const IndexType & indexCell ) const
    {
//        return n * indexCell + i;
        return i * numberOfCells + indexCell;
    }

    // accessor for auxiliary dofs
    __cuda_callable__
    RealType & Z_iK( const int & i, const IndexType & K )
    {
        return Z[ i * numberOfCells + K ];
    }
    __cuda_callable__
    const RealType & Z_iK( const int & i, const IndexType & K ) const
    {
        return Z[ i * numberOfCells + K ];
    }

    // accessors for coefficients
    // TODO: write accessors for m, m_upw, f
    __cuda_callable__
    RealType & N_ijK( const int & i, const int & j, const IndexType & K )
    {
        return N[ n * n * K + i * n + j ];
    }

    __cuda_callable__
    RealType & m_iK( const int & i, const IndexType & K )
    {
//        return m[ n * K + i ];
        return m[ i * numberOfCells + K ];
    }

    __cuda_callable__
    const RealType & m_iK( const int & i, const IndexType & K ) const
    {
//        return m[ n * K + i ];
        return m[ i * numberOfCells + K ];
    }

    // NOTE: only for D isotropic (represented by scalar value)
    __cuda_callable__
    RealType & D_ijK( const int & i, const int & j, const IndexType & K )
    {
        return D[ n * n * K + i * n + j ];
    }

    __cuda_callable__
    RealType & w_iKe( const int & i, const IndexType & K, const int & e )
    {
        return w[ n * K * FacesPerCell + i * FacesPerCell + e ];
    }
    __cuda_callable__
    const RealType & w_iKe( const int & i, const IndexType & K, const int & e ) const
    {
        return w[ n * K * FacesPerCell + i * FacesPerCell + e ];
    }

    // accessors for local matrices/vectors
    __cuda_callable__
    RealType* b_ijK( const int & i, const int & j, const IndexType & K )
    {
        // returns address of the first element of the mass matrix b_ijK
        return &b[ ((K * n + i) * n + j) * MassMatrix::size ];
    }
    __cuda_callable__
    const RealType* b_ijK( const int & i, const int & j, const IndexType & K ) const
    {
        // returns address of the first element of the mass matrix b_ijK
        return &b[ ((K * n + i) * n + j) * MassMatrix::size ];
    }

    __cuda_callable__
    RealType & R_ijKe( const int & i, const int & j, const IndexType & K, const int & e )
    {
//        return R1[ n * n * K * FacesPerCell + i * n * FacesPerCell + j * FacesPerCell + e ];
        // stored in column-major orientation with respect to i,j
        return R1[ n * n * K * FacesPerCell + n * j * FacesPerCell + n * e + i ];
    }
    __cuda_callable__
    const RealType & R_ijKe( const int & i, const int & j, const IndexType & K, const int & e ) const
    {
//        return R1[ n * n * K * FacesPerCell + i * n * FacesPerCell + j * FacesPerCell + e ];
        // stored in column-major orientation with respect to i,j
        return R1[ n * n * K * FacesPerCell + n * j * FacesPerCell + n * e + i ];
    }

    __cuda_callable__
    RealType & R_iK( const int & i, const IndexType & K )
    {
        return R2[ n * K + i ];
    }

    __cuda_callable__
    const RealType & R_iK( const int & i, const IndexType & K ) const
    {
        return R2[ n * K + i ];
    }

//protected:
    // auxiliary dofs
    DofVectorType Z;

    // coefficients
    DofVectorType N;
    DofVectorType m;
    DofVectorType D;
    DofVectorType w;
    DofVectorType f;

    // specific to MHFEM scheme
    DofVectorType m_upw;
    DofVectorType b;    // each "row" represents the local matrix (b_ijK)_EF
    
    DofVectorType R1;   // R_KF
    DofVectorType R2;   // R_K

    // FIXME: nasty hack to pass tau to QRupdater
    RealType current_tau;

protected:
    // number of entities of the mesh for which the vectors are allocated
    IndexType numberOfCells = 0;
    IndexType numberOfFaces = 0;

private:
    // FIXME: n can't be static constexpr because according to nvcc, ModelImplementation is an incomplete type (works in GCC though)
    const int n = ModelImplementation::NumberOfEquations;
    static constexpr int d = MeshType::getMeshDimension();
};

} // namespace mhfem

#include "BaseModel_impl.h"
