#pragma once

#include "MassMatrixDependentCode.h"
#include "../lib_general/mesh_helpers.h"
#include "../lib_general/LU.h"
#include "../lib_general/StaticMatrix.h"
#include "../lib_general/StaticSharedArray.h"

// TODO: bind with mesh-dependent data, e.g. as a subclass or local type

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class QRupdater
{
public:
    using MeshType = Mesh;
    using CoordinatesType = typename MeshType::CoordinatesType;
    using MeshDependentDataType = MeshDependentData;
    using RealType = typename MeshDependentDataType::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType>;
    using FaceVectorType = TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType >;
    using LocalMatrixType = StaticMatrix< MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations, RealType >;
    using MassMatrix = typename MeshDependentDataType::MassMatrix;
    using coeff = MassMatrixDependentCode< MeshDependentDataType >;

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

    struct update_b
    {
        template< typename EntityType >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const EntityType & entity,
                                   const int & i )
        {
            static_assert( EntityType::getDimensions() == MeshType::meshDimensions,
                           "wrong entity dimensions in QRupdater::processEntity");

            const IndexType K = entity.getIndex();

            // update coefficients b_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                MassMatrix::update( mesh, mdd, i, j, K );
            }
        }
    };

    // TODO: split into update_RKF, update_RK for better parallelism?
    struct update_R
    {
        template< typename EntityType >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const EntityType & entity,
                                   const int & i )
        {
            static_assert( EntityType::getDimensions() == MeshType::meshDimensions,
                           "wrong entity dimensions in QRupdater::processEntity");

            const IndexType K = entity.getIndex();

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            // update coefficients R_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
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
            R *= entity.getMeasure();
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
        template< typename EntityType >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const EntityType & entity,
                                   const int & _component )
        {
            static_assert( EntityType::getDimensions() == MeshType::meshDimensions,
                           "wrong entity dimensions in QRupdater::processEntity");

            const IndexType K = entity.getIndex();

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

#ifndef __CUDA_ARCH__
            LocalMatrixType Q;
#else
            // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
            // (we'll need to pass custom launch configuration to the traverser)
            __shared__ LocalMatrixType __Qs[ ( MeshType::getMeshDimensions() < 3 ) ? 256 : 512 ];
            LocalMatrixType& Q = __Qs[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];
#endif

            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                    RealType value = entity.getMeasure() * mdd.N_ijK( i, j, K );
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

            using SharedVectorType = StaticSharedArray< MeshDependentDataType::NumberOfEquations, RealType >;
            SharedVectorType rk( &mdd.R_iK( 0, K ) );
            LU_solve( Q, rk, rk );
            for( int j = 0; j < mdd.NumberOfEquations; j++ )
                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    SharedVectorType rke( &mdd.R_ijKe( 0, j, K, e ) );
                    LU_solve( Q, rke, rke );
                }
        }
    };
};

} // namespace mhfem
