#pragma once

#include <mesh/tnlGrid.h>
#include <core/vectors/tnlSharedVector.h>

#include "../mesh_helpers.h"
#include "../GEM.h"
#include "../StaticMatrix.h"

// TODO: bind with mesh-dependent data, e.g. as a subclass or local typedef
// TODO: zkontrolovat vzhledem ke konečné aritmetice (odčítání blízkých čísel)

namespace mhfem
{

template< typename Mesh,
          typename MeshDependentData >
class QRupdater
{
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename MeshDependentData >
class QRupdater< tnlGrid< 2, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef tnlStaticVector< 4, IndexType > FaceVectorType;
    typedef StaticMatrix< MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations, RealType > LocalMatrixType;

//    template< int EntityDimension >
//    __cuda_callable__
//    static void processEntity( const MeshType & mesh,
//                               MeshDependentDataType & mdd,
//                               const IndexType & indexCell,
//                               const CoordinatesType & coordinates )
//    {
//        tnlStaticAssert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

        // get face indexes
//        FaceVectorType faceIndexes;
//        getFacesForCell( mesh, indexCell, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

//        update_bR( mesh, mdd, indexCell, faceIndexes );
//        update_R_K( mesh, mdd, indexCell, faceIndexes );
//        update_Q( mesh, mdd, indexCell, faceIndexes );
//    }

//protected:

    struct update_bR
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & index,
                                   const CoordinatesType & coordinates )
        {
            tnlStaticAssert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

            const IndexType cells = mesh.getNumberOfCells();
            const IndexType K = index % cells;
            const int i = index / cells;

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

//            for( int i = 0; i < mdd.n; i++ ) {
//                for( int j = 0; j < mdd.n; j++ ) {
                    // NOTE: assumes that b_ijK is diagonal
//                    for( int e = 0; e < mdd.facesPerCell; e++ ) {
//                        const IndexType & E = faceIndexes[ e ];
                        // NOTE: only for D isotropic (represented by scalar value)
//                        RealType b = 2 * mdd.D_ijK( i, j, indexCell )
//                                     * (( isHorizontalFace( mesh, E ) ) ? mesh.getHx() * mesh.getHyInverse()
//                                                                        : mesh.getHy() * mesh.getHxInverse() );
//                        mdd.b_ijKe( i, j, indexCell, e ) = b;
//                        mdd.R_ijKe( i, j, indexCell, e ) = mdd.m_upw[ mdd.getDofIndex( i, E ) ] * b * mdd.current_tau; // TODO: - u_ijKe
//                    }
//                }
//            }

            // NOTE: assumes that b_ijK is diagonal
            for( int e = 0; e < mdd.facesPerCell; e++ ) {
                const IndexType & E = faceIndexes[ e ];
                // TODO: isVerticalFace <==> e < 2
                const RealType h = ( isHorizontalFace( mesh, E ) ) ? mesh.getHx() * mesh.getHyInverse()
                                                                   : mesh.getHy() * mesh.getHxInverse();
                for( int j = 0; j < mdd.n; j++ ) {
                    // NOTE: only for D isotropic (represented by scalar value)
                    RealType b = 2 * mdd.D_ijK( i, j, K ) * h;
                    mdd.b_ijKe( i, j, K, e ) = b;
                    mdd.R_ijKe( i, j, K, e ) = mdd.m_upw[ mdd.getDofIndex( i, E ) ] * b * mdd.current_tau; // TODO: - u_ijKe
                }
            }
        }
    };

    struct update_R_K
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & index,
                                   const CoordinatesType & coordinates )
        {
            tnlStaticAssert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

            const IndexType cells = mesh.getNumberOfCells();
            const IndexType K = index % cells;
            const int i = index / cells;

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

//            for( int i = 0; i < mdd.n; i++ ) {
                RealType value = 0.0;
                for( int j = 0; j < mdd.n; j++ ) {
                    value += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
                }
                value *= mesh.getHxHy();
                value += mesh.getHxHy() * mdd.f[ i * K ] * mdd.current_tau;
                for( int e = 0; e < mdd.facesPerCell; e++ ) {
                    const IndexType & E = faceIndexes[ e ];
                    value -= mdd.m_upw[ mdd.getDofIndex( i, E ) ] * mdd.w_iKe( i, K, e ) * mdd.current_tau;
                }
                mdd.R_iK( i, K ) = value;
//            }
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
            tnlStaticAssert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes[ 0 ], faceIndexes[ 1 ], faceIndexes[ 2 ], faceIndexes[ 3 ] );

            LocalMatrixType Q;
            for( int i = 0; i < mdd.n; i++ ) {
                for( int j = 0; j < mdd.n; j++ ) {
                    RealType value = mesh.getHxHy() * mdd.N_ijK( i, j, K );
                    for( int e = 0; e < mdd.facesPerCell; e++ ) {
                        const IndexType & E = faceIndexes[ e ];
                        value += mdd.m_upw[ mdd.getDofIndex( i, E ) ] * mdd.b_ijKe( i, j, K, e ) * mdd.current_tau;
                    }
                    Q.setElementFast( i, j, value );
                }
            }

            // TODO: simplify passing right hand sides
            SharedVectorType rk( &mdd.R_iK( 0, K ), mdd.n );
            if( mdd.n == 1 ) {
                SharedVectorType rke1( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.n );
                SharedVectorType rke2( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.n );
                SharedVectorType rke3( &mdd.R_ijKe( 0, 0, K, 2 ), mdd.n );
                SharedVectorType rke4( &mdd.R_ijKe( 0, 0, K, 3 ), mdd.n );
                GEM( Q, rk, rke1, rke2, rke3, rke4 );
            }
            else if( mdd.n == 2 ) {
                SharedVectorType rke11( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.n );
                SharedVectorType rke12( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.n );
                SharedVectorType rke13( &mdd.R_ijKe( 0, 0, K, 2 ), mdd.n );
                SharedVectorType rke14( &mdd.R_ijKe( 0, 0, K, 3 ), mdd.n );
                SharedVectorType rke21( &mdd.R_ijKe( 0, 1, K, 0 ), mdd.n );
                SharedVectorType rke22( &mdd.R_ijKe( 0, 1, K, 1 ), mdd.n );
                SharedVectorType rke23( &mdd.R_ijKe( 0, 1, K, 2 ), mdd.n );
                SharedVectorType rke24( &mdd.R_ijKe( 0, 1, K, 3 ), mdd.n );
                GEM( Q, rk, rke11, rke12, rke13, rke14,
                            rke21, rke22, rke23, rke24 );
            }
        }
    };
};

} // namespace mhfem
