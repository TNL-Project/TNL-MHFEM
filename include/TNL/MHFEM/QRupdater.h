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
class QRupdater< tnlGrid< 1, MeshReal, Device, MeshIndex >, MeshDependentData >
{
public:
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef Device DeviceType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
    typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVectorType;
    typedef tnlStaticVector< 2, IndexType > FaceVectorType;
    typedef StaticMatrix< MeshDependentDataType::NumberOfEquations, MeshDependentDataType::NumberOfEquations, RealType > LocalMatrixType;
    typedef typename MeshDependentDataType::MassMatrix MassMatrix;

    struct update_R
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & index,
                                   const CoordinatesType & coordinates )
        {
            static_assert( EntityDimension == 1, "wrong EntityDimension in QRupdater::processEntity");

            const IndexType cells = mesh.getNumberOfCells();
            const IndexType K = index % cells;
            const int i = index / cells;

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            // update coefficients b_ijKE and R_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                SharedVectorType storage( mdd.b_ijK( i, j, K ), MassMatrix::size );
                MassMatrix::update( mesh, mdd.D_ijK( i, j, K ), storage );

                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    const IndexType & E = faceIndexes[ e ];
                    // assuming that the b_ijKe coefficient (accessed with MassMatrix::get( e, storage ) )
                    // can be cached in L1 or L2 cache even on CUDA
                    mdd.R_ijKe( i, j, K, e ) = mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::get( e, storage ) * mdd.current_tau; // TODO: - u_ijKe
                }
            }

            // update coefficient R_iK
            RealType R = 0.0;
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                R += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
            }
            R *= mesh.getHx();
            R += mesh.getHx() * mdd.f[ i * K ] * mdd.current_tau;
            for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                const IndexType & E = faceIndexes[ e ];
                R -= mdd.m_upw[ mdd.getDofIndex( i, E ) ] * mdd.w_iKe( i, K, e ) * mdd.current_tau;
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
            static_assert( EntityDimension == 1, "wrong EntityDimension in QRupdater::processEntity");

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            LocalMatrixType Q;
            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                    RealType value = mesh.getHx() * mdd.N_ijK( i, j, K );
                    const SharedVectorType b_storage( mdd.b_ijK( i, j, K ), MassMatrix::size );
                    for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                        const IndexType & E = faceIndexes[ e ];
                        value += mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::get( e, b_storage ) * mdd.current_tau;
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

            // TODO: move to StaticMatrix class as operator<<
//            cout << "Q[ " << K << " ] = " << endl << "[";
//            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
//                cout << "[";
//                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
//                    cout << Q.getElementFast( i, j );
//                    if( j < mdd.NumberOfEquations - 1 )
//                        cout << ", ";
//                }
//                cout << "]";
//                if( i < mdd.NumberOfEquations - 1 )
//                    cout << endl;
//            }
//            cout << "]" << endl;

            // TODO: simplify passing right hand sides
            SharedVectorType rk( &mdd.R_iK( 0, K ), mdd.NumberOfEquations );
            if( mdd.NumberOfEquations == 1 ) {
                SharedVectorType rke1( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke2( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.NumberOfEquations );
                GEM( Q, rk, rke1, rke2 );
            }
            else if( mdd.NumberOfEquations == 2 ) {
                SharedVectorType rke11( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke12( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.NumberOfEquations );
                SharedVectorType rke21( &mdd.R_ijKe( 0, 1, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke22( &mdd.R_ijKe( 0, 1, K, 1 ), mdd.NumberOfEquations );
                GEM( Q, rk, rke11, rke12,
                            rke21, rke22 );
            }
        }
    };
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
    typedef typename MeshDependentDataType::MassMatrix MassMatrix;

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

    struct update_R
    {
        template< int EntityDimension >
        __cuda_callable__
        static void processEntity( const MeshType & mesh,
                                   MeshDependentDataType & mdd,
                                   const IndexType & index,
                                   const CoordinatesType & coordinates )
        {
            static_assert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

            const IndexType cells = mesh.getNumberOfCells();
            const IndexType K = index % cells;
            const int i = index / cells;

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            // update coefficients b_ijKE and R_ijKE
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                SharedVectorType storage( mdd.b_ijK( i, j, K ), MassMatrix::size );
                MassMatrix::update( mesh, mdd.D_ijK( i, j, K ), storage );

                for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                    const IndexType & E = faceIndexes[ e ];
                    // assuming that the b_ijKe coefficient (accessed with MassMatrix::get( e, storage ) )
                    // can be cached in L1 or L2 cache even on CUDA
                    mdd.R_ijKe( i, j, K, e ) = mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::get( e, storage ) * mdd.current_tau; // TODO: - u_ijKe
                }
            }

            // update coefficient R_iK
            RealType R = 0.0;
            for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                R += mdd.N_ijK( i, j, K ) * mdd.Z_iK( j, K );
            }
            R *= mesh.getHxHy();
            R += mesh.getHxHy() * mdd.f[ i * K ] * mdd.current_tau;
            for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                const IndexType & E = faceIndexes[ e ];
                R -= mdd.m_upw[ mdd.getDofIndex( i, E ) ] * mdd.w_iKe( i, K, e ) * mdd.current_tau;
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
            static_assert( EntityDimension == 2, "wrong EntityDimension in QRupdater::processEntity");

            // get face indexes
            FaceVectorType faceIndexes;
            getFacesForCell( mesh, K, faceIndexes );

            LocalMatrixType Q;
            for( int i = 0; i < mdd.NumberOfEquations; i++ ) {
                // Q is singular if it has a row with all elements equal to zero
                bool singular = true;

                for( int j = 0; j < mdd.NumberOfEquations; j++ ) {
                    RealType value = mesh.getHxHy() * mdd.N_ijK( i, j, K );
                    const SharedVectorType b_storage( mdd.b_ijK( i, j, K ), MassMatrix::size );
                    for( int e = 0; e < mdd.FacesPerCell; e++ ) {
                        const IndexType & E = faceIndexes[ e ];
                        value += mdd.m_upw[ mdd.getDofIndex( i, E ) ] * MassMatrix::get( e, b_storage ) * mdd.current_tau;
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

            // TODO: simplify passing right hand sides
            SharedVectorType rk( &mdd.R_iK( 0, K ), mdd.NumberOfEquations );
            if( mdd.NumberOfEquations == 1 ) {
                SharedVectorType rke1( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke2( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.NumberOfEquations );
                SharedVectorType rke3( &mdd.R_ijKe( 0, 0, K, 2 ), mdd.NumberOfEquations );
                SharedVectorType rke4( &mdd.R_ijKe( 0, 0, K, 3 ), mdd.NumberOfEquations );
                GEM( Q, rk, rke1, rke2, rke3, rke4 );
            }
            else if( mdd.NumberOfEquations == 2 ) {
                SharedVectorType rke11( &mdd.R_ijKe( 0, 0, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke12( &mdd.R_ijKe( 0, 0, K, 1 ), mdd.NumberOfEquations );
                SharedVectorType rke13( &mdd.R_ijKe( 0, 0, K, 2 ), mdd.NumberOfEquations );
                SharedVectorType rke14( &mdd.R_ijKe( 0, 0, K, 3 ), mdd.NumberOfEquations );
                SharedVectorType rke21( &mdd.R_ijKe( 0, 1, K, 0 ), mdd.NumberOfEquations );
                SharedVectorType rke22( &mdd.R_ijKe( 0, 1, K, 1 ), mdd.NumberOfEquations );
                SharedVectorType rke23( &mdd.R_ijKe( 0, 1, K, 2 ), mdd.NumberOfEquations );
                SharedVectorType rke24( &mdd.R_ijKe( 0, 1, K, 3 ), mdd.NumberOfEquations );
                GEM( Q, rk, rke11, rke12, rke13, rke14,
                            rke21, rke22, rke23, rke24 );
            }
        }
    };
};

} // namespace mhfem
