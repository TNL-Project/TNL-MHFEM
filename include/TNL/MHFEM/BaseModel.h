#pragma once

#include <core/tnlObject.h>
#include <mesh/tnlGrid.h>
#include <matrices/tnlDenseMatrix.h>

namespace mhfem
{

template< typename Mesh,
          typename Real,
          typename Index,
          typename ModelImplementation >
class BaseModel :
    public tnlObject
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename ModelImplementation >
class BaseModel< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, ModelImplementation > :
    public tnlObject
{
public:
    // TODO: for some arcane reason 'using ModelImplementation::MeshType' does not work, but 'IndexType n = ModelImplementation::NumberOfEquations' does
    // (using typedefs from children would greatly simplify the parametrization of BaseModel)
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType > DofVectorType;

    bool allocate( const MeshType & mesh );

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
//    bool makeSnapshot( const RealType & time,
//                       const IndexType & step,
//                       const MeshType & mesh ) const;

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

    // accessor for auxiliary dofs
    __cuda_callable__
    RealType & Z_iK( const int & i, const IndexType & K )
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

    // NOTE: only for D isotropic (represented by scalar value)
    __cuda_callable__
    RealType & D_ijK( const int & i, const int & j, const IndexType & K )
    {
        return D[ n * n * K + i * n + j ];
    }

    __cuda_callable__
    RealType & w_iKe( const int & i, const IndexType & K, const int & e )
    {
        return w[ n * K * facesPerCell + i * facesPerCell + e ];
    }

    // accessors for local matrices/vectors
    __cuda_callable__
    RealType & b_ijKe( const int & i, const int & j, const IndexType & K, const int & e )
    {
        return b[ n * n * K * facesPerCell + i * n * facesPerCell + j * facesPerCell + e ];
    }

    __cuda_callable__
    // TODO: should return reference, but getElementFast returns by value
//    RealType & Q_ijK( const int & i, const int & j, const IndexType & K )
    RealType Q_ijK( const int & i, const int & j, const IndexType & K )
    {
//        return Q[ n * n * K + i * n + j ];
        return Q[ K ].getElementFast( i, j );
    }

    __cuda_callable__
    RealType & R_ijKe( const int & i, const int & j, const IndexType & K, const int & e )
    {
//        return R1[ n * n * K * facesPerCell + i * n * facesPerCell + j * facesPerCell + e ];
        // stored in column-major orientation with respect to i,j
        return R1[ n * n * K * facesPerCell + n * j * facesPerCell + n * e + i ];
    }

    __cuda_callable__
    RealType & R_iK( const int & i, const IndexType & K )
    {
        return R2[ n * K + i ];
    }

//protected:
    // TODO: generalize
    const int n = ModelImplementation::NumberOfEquations;
    const int facesPerCell = 4;

    // auxiliary dofs
    DofVectorType Z;

    // coefficients
    DofVectorType N;    // porosity * idealGasCoefficient
    DofVectorType m;    // density = idealGasCoefficient * pressure
    DofVectorType D;    // permeability / viscosity
    DofVectorType w;    // idealGasCoefficient * pressure * g
    DofVectorType f;

    // specific to MHFEM scheme
    DofVectorType m_upw;
    DofVectorType b;    // each "row" represents local diagonal matrix (b_ijK)_EF
    
    // TODO: need shared matrix
//    DofVectorType Q;
    tnlArray< tnlDenseMatrix< RealType, DeviceType, IndexType >, DeviceType, IndexType > Q;
    DofVectorType R1;   // R_KF
    DofVectorType R2;   // R_K

    // FIXME: nasty hack to pass tau to QRupdater
    RealType current_tau;

protected:
    const int d = 2;

    // number of entities of the mesh for which the vectors are allocated
    IndexType numberOfCells = 0;
    IndexType numberOfFaces = 0;
};

} // namespace mhfem

#include "BaseModel_impl.h"
