#pragma once

#include <core/vectors/tnlSharedVector.h>

#include "MassMatrixDependentCode.h"
#include "../lib_general/mesh_helpers.h"
#include "../lib_general/LU.h"
#include "../lib_general/StaticMatrix.h"

// TODO: bind with mesh-dependent data, e.g. as a subclass or local typedef

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class QRupdater
{
public:
    typedef Mesh MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentData::DeviceType DeviceType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef tnlStaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;
    typedef StaticMatrix< MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations, RealType > LocalMatrixType;
    typedef typename MeshDependentDataType::MassMatrix MassMatrix;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;

//    template< int EntityDimension >
//    __cuda_callable__
//    static void processEntity( const MeshType & mesh,
//                               MeshDependentDataType & mdd,
//                               const IndexType & indexCell,
//                               const CoordinatesType & coordinates )
//    {
//        static_assert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

        // get face indexes
//        FaceVectorType faceIndexes;
//        getFacesForCell( mesh, indexCell, faceIndexes );

//        update_bR( mesh, mdd, indexCell, faceIndexes );
//        update_R_K( mesh, mdd, indexCell, faceIndexes );
//        update_Q( mesh, mdd, indexCell, faceIndexes );
//    }

//protected:

    // TODO: split into update_b, update_RKF, update_RK for better parallelism
    struct update_R
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & index,
                                   const CoordinatesType & coordinates )
        {
            static_assert( EntityDimension == MeshType::Dimensions, "wrong EntityDimension in QRupdater::processEntity");

            const IndexType cells = mesh.getNumberOfCells();
            const IndexType K = index % cells;
            const int i = index / cells;

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            // update coefficients b_ijKE and R_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                MassMatrix::update( mesh, mdd, i, j, K );

                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    // assuming that the b_ijKe coefficient (accessed with MassMatrix::get( e, storage ) )
                    // can be cached in L1 or L2 cache even on CUDA
                    mdd.R_ijKe( i, j, K, e ) = coeff::R_ijKe( mdd, faceIndexes, i, j, K, e );
                }
            }

            // update coefficient R_iK
            RealType R = 0.0;
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                R += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
            }
            R += mdd.f[ i * K ] * mdd.current_tau;
            R *= getCellVolume( mesh, K );
            for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                const IndexType & E = faceIndexes[ e ];
                // TODO: simplify updating the w coefficient
                const RealType w_iKe = mdd.update_w( mesh, i, K, e );
                R -= mdd.m_upw[ mdd.getDofIndex( i, E ) ] * w_iKe * mdd.current_tau;
            }
            mdd.R_iK( i, K ) = R;
        }
    };

    struct update_Q
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & K,
                                   const CoordinatesType & coordinates )
        {
            static_assert( EntityDimension == MeshType::Dimensions, "wrong EntityDimension in QRupdater::processEntity");

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            LocalMatrixType Q;
            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                    RealType value = getCellVolume( mesh, K ) * mdd.N_ijK( i, j, K );
                    for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                        const IndexType & E = faceIndexes[ e ];
                        value += mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::b_ijKe( mdd, i, j, K, e ) * mdd.current_tau;
                    }
                    Q.setElementFast( i, j, value );

                    // update singularity state
                    if( value != 0.0 )
                        singular = false;
                }

                // check for singularity
                if( singular ) {
                    Q.setElementFast( i, i, 1.0 );
                    mdd.R_iK( i, K ) += mdd.Z_iK( i, K );
                }
            }

            LU_factorize( Q );

            SharedVectorType rk( &mdd.R_iK( 0, K ), mdd.NumberOfEquations );
            LU_solve( Q, rk, rk );
            for( int j = 0; j < mdd.NumberOfEquations; j++ )
                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    SharedVectorType rke( &mdd.R_ijKe( 0, j, K, e ), mdd.NumberOfEquations );
                    LU_solve( Q, rke, rke );
                }
        }
    };
};

} // namespace mhfem
