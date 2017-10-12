#pragma once

#include "SecondaryCoefficients.h"
#include "../lib_general/mesh_helpers.h"
#include "../lib_general/LU.h"
#include "../lib_general/StaticMatrix.h"
#include "../lib_general/StaticSharedArray.h"

// TODO: bind with mesh-dependent data, e.g. as a subclass or local type

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class LocalUpdaters
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using RealType = typename MeshDependentDataType::RealType;
    using coeff = SecondaryCoefficients< MeshDependentDataType >;

    struct update_b
    {
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const typename MeshType::Cell & entity,
                                   const int & i )
        {
            using MassMatrix = typename MeshDependentDataType::MassMatrix;

            // update coefficients b_ijKEF
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                MassMatrix::update( mesh, entity, mdd, i, j );
            }
        }
    };

    // TODO: split into update_RKF, update_RK for better parallelism?
    struct update_R
    {
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const typename MeshType::Cell & entity,
                                   const int & i )
        {
            const auto K = entity.getIndex();

            // get face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );

            // update coefficients R_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ )
                for( int e = 0; e < mdd.FacesPerCell; e++ )
                    mdd.R_ijKe( i, j, K, e ) = coeff::R_ijKe( mdd, faceIndexes, i, j, K, e );

            // update coefficient R_iK
            mdd.R_iK( i, K ) = coeff::R_iK( mdd, mesh, entity, faceIndexes, i, K );
        }
    };

    struct update_Q
    {
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const typename MeshType::Cell & entity,
                                   const int & _component )
        {
            const auto K = entity.getIndex();

            // get face indexes
            const auto faceIndexes = getFacesForCell( mesh, K );

            using LocalMatrixType = StaticMatrix< MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations, RealType >;
#ifndef __CUDA_ARCH__
            LocalMatrixType Q;
//            RealType rhs[ MeshDependentDataType::NumberOfEquations ];
#else
            // TODO: use dynamic allocation via Devices::Cuda::getSharedMemory
            // (we'll need to pass custom launch configuration to the traverser)
            // TODO: the traverser for Mesh will use 256 threads per block even in 3D
            __shared__ LocalMatrixType __Qs[ ( MeshType::getMeshDimension() < 3 ) ? 256 : 512 ];
            LocalMatrixType& Q = __Qs[ ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x ];

            // TODO: this limits the kernel to 1 block on Fermi - maybe cudaFuncCachePreferShared specifically for this kernel would help
//            __shared__ RealType __rhss[ MeshDependentDataType::NumberOfEquations * ( ( MeshType::getMeshDimension() < 3 ) ? 256 : 512 ) ];
//            RealType* rhs = &__rhss[ MeshDependentDataType::NumberOfEquations * (
//                                        ( ( threadIdx.z * blockDim.y ) + threadIdx.y ) * blockDim.x + threadIdx.x
//                                    ) ];
#endif

            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                    const RealType value = coeff::Q_ijK( mdd, mesh, entity, faceIndexes, i, j, K );
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

//            using SharedVectorType = StaticSharedArray< MeshDependentDataType::NumberOfEquations, RealType >;
//            SharedVectorType rk( &mdd.R_iK( 0, K ) );
//            LU_solve( Q, rk, rk );
//            for( int j = 0; j < mdd.NumberOfEquations; j++ )
//                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
//                    SharedVectorType rke( &mdd.R_ijKe( 0, j, K, e ) );
//                    LU_solve( Q, rke, rke );
//                }

            RealType rhs[ MeshDependentDataType::NumberOfEquations ];

            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                rhs[ i ] = mdd.R_iK( i, K );
            LU_solve_inplace( Q, rhs );
            for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                mdd.R_iK( i, K ) = rhs[ i ];

            for( int j = 0; j < mdd.NumberOfEquations; j++ )
                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        rhs[ i ] = mdd.R_ijKe( i, j, K, e );
                    LU_solve_inplace( Q, rhs );
                    for( int i = 0; i < MeshDependentDataType::NumberOfEquations; i++ )
                        mdd.R_ijKe( i, j, K, e ) = rhs[ i ];
                }
        }
    };

    struct update_v
    {
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const typename MeshType::Cell & entity,
                                   const int & i )
        {
            const auto K = entity.getIndex();
            const auto faceIndexes = getFacesForCell( mesh, K );

            for( int e = 0; e < MeshDependentDataType::FacesPerCell; e++ )
                mdd.v_iKe( i, K, e ) = coeff::v_iKE( mdd, faceIndexes, i, K, faceIndexes[ e ], e );
        }
    };
};

} // namespace mhfem
