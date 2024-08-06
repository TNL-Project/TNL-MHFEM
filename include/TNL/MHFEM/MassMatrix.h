#pragma once

#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Matrices/StaticMatrix.h>
#include <TNL/Matrices/Factorization/LUsequential.h>

#include "mesh_helpers.h"

namespace TNL::MHFEM {

enum class MassLumping {
    enabled,
    disabled
};


template< typename MeshEntity, MassLumping >
class MassMatrix
{};

// NOTE: everything is only for D isotropic (represented by scalar value)

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Edge >, MassLumping::enabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Edge >;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::enabled;
    static constexpr bool is_diagonal = true;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        const auto h_x = getEntityMeasure( mesh, entity );
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) / h_x;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e == f )
            return mdd.b_ijK_storage( i, j, K, 0 );
        return 0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return mdd.b_ijK_storage( i, j, K, 0 );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Edge >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Edge >;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 1;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        const auto h_x = getEntityMeasure( mesh, entity );
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) / h_x;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e == f )
            return 2 * mdd.b_ijK_storage( i, j, K, 0 );
        return mdd.b_ijK_storage( i, j, K, 0 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return 3 * mdd.b_ijK_storage( i, j, K, 0 );
    }
};


// NOTE: this is *not* for a general quadrilateral, we assume a rectangle
template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle >, MassLumping::enabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle >;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::enabled;
    static constexpr bool is_diagonal = true;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        // left bottom front vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top back vertex
        const PointType v_2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) ).getPoint();
        const auto h_x = v_2.x() - v_0.x();
        const auto h_y = v_2.y() - v_0.y();

        // value for horizontal faces (e=0, e=2)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * h_x / h_y;
        // value for vertical faces (e=1, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * h_y / h_x;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        // non-diagonal entries
        if( e != f )
            return 0;
        // diagonal entries - equal to b_ijKe for a diagonal matrix
        return b_ijKe( mdd, i, j, K, e );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        // horizontal face (e=0, e=2)
        if( e % 2 == 0 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // vertical face (e=1, e=3)
        return mdd.b_ijK_storage( i, j, K, 1 );
    }
};

// NOTE: this is *not* for a general quadrilateral, we assume a rectangle
template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Quadrangle >;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 2;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        // left bottom front vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top back vertex
        const PointType v_2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) ).getPoint();
        const auto h_x = v_2.x() - v_0.x();
        const auto h_y = v_2.y() - v_0.y();

        // value for horizontal faces (e=0, e=2)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * h_x / h_y;
        // value for vertical faces (e=1, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * h_y / h_x;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        // horizontal faces (e,f = 0 or 2)
        if( e % 2 == 0 && f % 2 == 0 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 0 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 0 );
        }
        // vertical faces (e,f = 1 or 3)
        if( e % 2 == 1 && f % 2 == 1 ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 1 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 1 );
        }
        // non-diagonal blocks
        return 0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        // horizontal face (e=0, e=2)
        if( e % 2 == 0 )
            return 3 * mdd.b_ijK_storage( i, j, K, 0 );
        // vertical face (e=1, e=3)
        return 3 * mdd.b_ijK_storage( i, j, K, 1 );
    }
};


// NOTE: this is *not* for a general hexahedron, we assume a voxel
template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron >, MassLumping::enabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron >;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::enabled;
    static constexpr bool is_diagonal = true;

    // number of independent values defining the matrix
    static constexpr int size = 3;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        // left bottom front vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top back vertex
        const PointType v_6 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 6 ) ).getPoint();
        const auto h_x = v_6.x() - v_0.x();
        const auto h_y = v_6.y() - v_0.y();
        const auto h_z = v_6.z() - v_0.z();

        // value for n_x faces (e=2, e=4)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * h_y * h_z / h_x;
        // value for n_y faces (e=1, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * h_x * h_z / h_y;
        // value for n_z faces (e=0, e=5)
        mdd.b_ijK_storage( i, j, K, 2 ) = 2 * mdd.D_ijK( i, j, K ) * h_x * h_y / h_z;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        // non-diagonal entries
        if( e != f )
            return 0;
        // diagonal entries - equal to b_ijKe for a diagonal matrix
        return b_ijKe( mdd, i, j, K, e );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        // n_x face (e=2, e=4)
        if( e == 2 || e == 4 )
            return mdd.b_ijK_storage( i, j, K, 0 );
        // n_y face (e=1, e=3)
        if( e == 1 || e == 3 )
            return mdd.b_ijK_storage( i, j, K, 1 );
        // n_z face (e=0, e=5)
        return mdd.b_ijK_storage( i, j, K, 2 );
    }
};

// NOTE: this is *not* for a general hexahedron, we assume a voxel
template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Hexahedron >;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 3;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );
        // left bottom front vertex
        const PointType v_0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) ).getPoint();
        // right top back vertex
        const PointType v_6 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 6 ) ).getPoint();
        const auto h_x = v_6.x() - v_0.x();
        const auto h_y = v_6.y() - v_0.y();
        const auto h_z = v_6.z() - v_0.z();

        // value for n_x faces (e=2, e=4)
        mdd.b_ijK_storage( i, j, K, 0 ) = 2 * mdd.D_ijK( i, j, K ) * h_y * h_z / h_x;
        // value for n_y faces (e=1, e=3)
        mdd.b_ijK_storage( i, j, K, 1 ) = 2 * mdd.D_ijK( i, j, K ) * h_x * h_z / h_y;
        // value for n_z faces (e=0, e=5)
        mdd.b_ijK_storage( i, j, K, 2 ) = 2 * mdd.D_ijK( i, j, K ) * h_x * h_y / h_z;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             const LocalIndex e,
             const LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        // n_x faces (e,f = 2 or 4)
        if( (e == 2 || e == 4) && (f == 2 || f == 4) ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 0 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 0 );
        }
        // n_y faces (e,f = 1 or 3)
        if( (e == 1 || e == 3) && (f == 1 || f == 3) ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 1 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 1 );
        }
        // n_z faces (e,f = 0 or 5)
        if( (e == 0 || e == 5) && (f == 0 || f == 5) ) {
            if( e == f )
                // diagonal
                return 2 * mdd.b_ijK_storage( i, j, K, 2 );
            // non-diagonal
            return mdd.b_ijK_storage( i, j, K, 2 );
        }
        // non-diagonal blocks
        return 0;
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        // n_x face (e=2, e=4)
        if( e == 2 || e == 4 )
            return 3 * mdd.b_ijK_storage( i, j, K, 0 );
        // n_y face (e=1, e=3)
        if( e == 1 || e == 3 )
            return 3 * mdd.b_ijK_storage( i, j, K, 1 );
        // n_z face (e=0, e=5)
        return 3 * mdd.b_ijK_storage( i, j, K, 2 );
    }
};


template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Triangle >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Triangle >;
    using RealType = typename Mesh::RealType;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 9;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );

        // TNL orders the subentities such that i-th subvertex is the opposite vertex of i-th subface
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );

        const PointType P0 = v0.getPoint() - v2.getPoint();
        const PointType P1 = v1.getPoint() - v2.getPoint();
        // P2 = 0

        const RealType P00 = (P0, P0);
        const RealType P01 = (P0, P1);
        const RealType P11 = (P1, P1);

        const auto denominator = 24 * getEntityMeasure( mesh, entity );

        // LU decomposition is stable
        // TODO: use Cholesky instead
        using namespace TNL::Matrices::Factorization;

        TNL::Matrices::StaticMatrix< typename Mesh::RealType, 3, 3 > matrix;
        TNL::Containers::StaticVector< 3, typename Mesh::RealType > v;

        matrix( 0, 0 ) = ( 3 * P00 + P11 - 3 * P01 ) / denominator;
        matrix( 1, 1 ) = ( P00 + 3 * P11 - 3 * P01 ) / denominator;
        matrix( 2, 2 ) = ( P00 + P11 + P01 ) / denominator;
        matrix( 0, 1 ) = ( 3 * P01 - P00 - P11 ) / denominator;
        matrix( 0, 2 ) = ( P11 - P00 - P01 ) / denominator;
        matrix( 1, 2 ) = ( P00 - P11 - P01 ) / denominator;

        matrix( 1, 0 ) = matrix( 0, 1 );
        matrix( 2, 0 ) = matrix( 0, 2 );
        matrix( 2, 1 ) = matrix( 1, 2 );

        LU_sequential_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * mdd.D_ijK( i, j, K );

        // the last 3 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 6 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 );
        mdd.b_ijK_storage( i, j, K, 7 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 );
        mdd.b_ijK_storage( i, j, K, 8 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return mdd.b_ijK_storage( i, j, K, 6 + e );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Triangle >, MassLumping::enabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Triangle >;
    using RealType = typename Mesh::RealType;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::enabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 9;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );

        // TNL orders the subentities such that i-th subvertex is the opposite vertex of i-th subface
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );

        const PointType P_01 = v0.getPoint() - v1.getPoint();
        const PointType P_02 = v0.getPoint() - v2.getPoint();
        const PointType P_12 = v1.getPoint() - v2.getPoint();

        const RealType P_0101 = (P_01, P_01);
        const RealType P_0202 = (P_02, P_02);
        const RealType P_1212 = (P_12, P_12);
        const RealType P_0102 = (P_01, P_02);
        const RealType P_0112 = (P_01, P_12);
        const RealType P_0212 = (P_02, P_12);

        const auto denominator = 12 * getEntityMeasure( mesh, entity );

        // LU decomposition is stable
        // TODO: use Cholesky instead
        using namespace TNL::Matrices::Factorization;

        TNL::Matrices::StaticMatrix< typename Mesh::RealType, 3, 3 > matrix;
        TNL::Containers::StaticVector< 3, typename Mesh::RealType > v;

        matrix( 0, 0 ) = ( P_0101 + P_0202 ) / denominator;
        matrix( 1, 1 ) = ( P_0101 + P_1212 ) / denominator;
        matrix( 2, 2 ) = ( P_0202 + P_1212 ) / denominator;
        matrix( 0, 1 ) = ( P_0212 ) / denominator;
        matrix( 0, 2 ) = ( - P_0112 ) / denominator;
        matrix( 1, 2 ) = ( P_0102 ) / denominator;

        matrix( 1, 0 ) = matrix( 0, 1 );
        matrix( 2, 0 ) = matrix( 0, 2 );
        matrix( 2, 1 ) = matrix( 1, 2 );

        LU_sequential_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * mdd.D_ijK( i, j, K );

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * mdd.D_ijK( i, j, K );
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * mdd.D_ijK( i, j, K );

        // the last 3 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 6 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 );
        mdd.b_ijK_storage( i, j, K, 7 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 );
        mdd.b_ijK_storage( i, j, K, 8 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return mdd.b_ijK_storage( i, j, K, 6 + e );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Tetrahedron >, MassLumping::disabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Tetrahedron >;
    using RealType = typename Mesh::RealType;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::disabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 14;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );

        // TNL orders the subentities such that i-th subvertex is the opposite vertex of i-th subface
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
        const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );

        const PointType P0 = v0.getPoint() - v3.getPoint();
        const PointType P1 = v1.getPoint() - v3.getPoint();
        const PointType P2 = v2.getPoint() - v3.getPoint();
        // P3 = 0

        const RealType P00 = (P0, P0);
        const RealType P11 = (P1, P1);
        const RealType P22 = (P2, P2);
        const RealType P01 = (P0, P1);
        const RealType P02 = (P0, P2);
        const RealType P12 = (P1, P2);

        const auto D = mdd.D_ijK( i, j, K );
        const auto denominator = 180 * getEntityMeasure( mesh, entity );

        // LU decomposition is stable
        // TODO: use Cholesky instead
        using namespace TNL::Matrices::Factorization;

        TNL::Matrices::StaticMatrix< typename Mesh::RealType, 4, 4 > matrix;
        TNL::Containers::StaticVector< 4, typename Mesh::RealType > v;

        matrix( 0, 0 ) = ( 12 * P00 +  2 * P11 +  2 * P22 - 8 * P01 - 8 * P02 + 2 * P12 ) / denominator;
        matrix( 1, 1 ) = (  2 * P00 + 12 * P11 +  2 * P22 - 8 * P01 + 2 * P02 - 8 * P12 ) / denominator;
        matrix( 2, 2 ) = (  2 * P00 +  2 * P11 + 12 * P22 + 2 * P01 - 8 * P02 - 8 * P12 ) / denominator;
        matrix( 3, 3 ) = 2 * ( P00 + P11 + P22 + P01 + P02 + P12 ) / denominator;
        matrix( 0, 1 ) = ( - 3 * P00 - 3 * P11 + 2 * P22 + 12 * P01 -  3 * P02 -  3 * P12 ) / denominator;
        matrix( 0, 2 ) = ( - 3 * P00 + 2 * P11 - 3 * P22 -  3 * P01 + 12 * P02 -  3 * P12 ) / denominator;
        matrix( 1, 2 ) = (   2 * P00 - 3 * P11 - 3 * P22 -  3 * P01 -  3 * P02 + 12 * P12 ) / denominator;
        matrix( 0, 3 ) = ( - 3 * P00 + 2 * P11 + 2 * P22 - 3 * P01 - 3 * P02 + 2 * P12 ) / denominator;
        matrix( 1, 3 ) = (   2 * P00 - 3 * P11 + 2 * P22 - 3 * P01 + 2 * P02 - 3 * P12 ) / denominator;
        matrix( 2, 3 ) = (   2 * P00 + 2 * P11 - 3 * P22 + 2 * P01 - 3 * P02 - 3 * P12 ) / denominator;

        matrix( 1, 0 ) = matrix( 0, 1 );
        matrix( 2, 0 ) = matrix( 0, 2 );
        matrix( 2, 1 ) = matrix( 1, 2 );
        matrix( 3, 0 ) = matrix( 0, 3 );
        matrix( 3, 1 ) = matrix( 1, 3 );
        matrix( 3, 2 ) = matrix( 2, 3 );

        LU_sequential_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * D;

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * D;

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * D;
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * D;

        v.setValue( 0.0 );
        v[ 3 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 6 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 7 ) = v[ 1 ] * D;
        mdd.b_ijK_storage( i, j, K, 8 ) = v[ 2 ] * D;
        mdd.b_ijK_storage( i, j, K, 9 ) = v[ 3 ] * D;

        // the last 4 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 10 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 6 );
        mdd.b_ijK_storage( i, j, K, 11 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 7 );
        mdd.b_ijK_storage( i, j, K, 12 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 ) + mdd.b_ijK_storage( i, j, K, 8 );
        mdd.b_ijK_storage( i, j, K, 13 ) = mdd.b_ijK_storage( i, j, K, 6 ) + mdd.b_ijK_storage( i, j, K, 7 ) + mdd.b_ijK_storage( i, j, K, 8 ) + mdd.b_ijK_storage( i, j, K, 9 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return mdd.b_ijK_storage( i, j, K, 10 + e );
    }
};

template< typename MeshConfig, typename Device >
class MassMatrix< TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Tetrahedron >, MassLumping::enabled >
{
public:
    using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;
    using MeshEntity = TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Tetrahedron >;
    using RealType = typename Mesh::RealType;
    using PointType = typename Mesh::PointType;
    using LocalIndex = typename Mesh::LocalIndexType;
    using GlobalIndex = typename Mesh::GlobalIndexType;
    static constexpr MassLumping lumping = MassLumping::enabled;
    static constexpr bool is_diagonal = false;

    // number of independent values defining the matrix
    static constexpr int size = 14;

    template< typename MeshDependentData >
    __cuda_callable__
    static void
    update( const Mesh & mesh,
            MeshDependentData & mdd,
            const GlobalIndex K,
            const LocalIndex i,
            const LocalIndex j )
    {
        const auto& entity = mesh.template getEntity< typename Mesh::Cell >( K );

        // TNL orders the subentities such that i-th subvertex is the opposite vertex of i-th subface
        const auto& v0 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 2 ) );
        const auto& v3 = mesh.template getEntity< 0 >( entity.template getSubentityIndex< 0 >( 3 ) );

        const PointType P_01 = v0.getPoint() - v1.getPoint();
        const PointType P_02 = v0.getPoint() - v2.getPoint();
        const PointType P_03 = v0.getPoint() - v3.getPoint();
        const PointType P_12 = v1.getPoint() - v2.getPoint();
        const PointType P_13 = v1.getPoint() - v3.getPoint();
        const PointType P_23 = v2.getPoint() - v3.getPoint();

        const RealType P_0101 = (P_01, P_01);
        const RealType P_0202 = (P_02, P_02);
        const RealType P_0303 = (P_03, P_03);
        const RealType P_1212 = (P_12, P_12);
        const RealType P_1313 = (P_13, P_13);
        const RealType P_2323 = (P_23, P_23);

        const auto D = mdd.D_ijK( i, j, K );
        const auto denominator = 36 * getEntityMeasure( mesh, entity );

        // LU decomposition is stable
        // TODO: use Cholesky instead
        using namespace TNL::Matrices::Factorization;

        TNL::Matrices::StaticMatrix< typename Mesh::RealType, 4, 4 > matrix;
        TNL::Containers::StaticVector< 4, typename Mesh::RealType > v;

        matrix( 0, 0 ) = ( P_0101 + P_0202 + P_0303 ) / denominator;
        matrix( 1, 1 ) = ( P_0101 + P_1212 + P_1313 ) / denominator;
        matrix( 2, 2 ) = ( P_0202 + P_1212 + P_2323 ) / denominator;
        matrix( 3, 3 ) = ( P_0303 + P_1313 + P_2323 ) / denominator;
        matrix( 0, 1 ) = (   (P_02, P_12) + (P_03, P_13) ) / denominator;
        matrix( 0, 2 ) = ( - (P_01, P_12) + (P_03, P_23) ) / denominator;
        matrix( 0, 3 ) = ( - (P_01, P_13) - (P_02, P_23) ) / denominator;
        matrix( 1, 2 ) = (   (P_01, P_02) + (P_13, P_23) ) / denominator;
        matrix( 1, 3 ) = (   (P_01, P_03) - (P_12, P_23) ) / denominator;
        matrix( 2, 3 ) = (   (P_02, P_03) + (P_12, P_13) ) / denominator;

        matrix( 1, 0 ) = matrix( 0, 1 );
        matrix( 2, 0 ) = matrix( 0, 2 );
        matrix( 2, 1 ) = matrix( 1, 2 );
        matrix( 3, 0 ) = matrix( 0, 3 );
        matrix( 3, 1 ) = matrix( 1, 3 );
        matrix( 3, 2 ) = matrix( 2, 3 );

        LU_sequential_factorize( matrix );

        // store the inverse in the packed format (upper triangle, column by column)
        // see: http://www.netlib.org/lapack/lug/node123.html

        v.setValue( 0.0 );
        v[ 0 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 0 ) = v[ 0 ] * D;

        v.setValue( 0.0 );
        v[ 1 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 1 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 2 ) = v[ 1 ] * D;

        v.setValue( 0.0 );
        v[ 2 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 3 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 4 ) = v[ 1 ] * D;
        mdd.b_ijK_storage( i, j, K, 5 ) = v[ 2 ] * D;

        v.setValue( 0.0 );
        v[ 3 ] = 1.0;
        LU_sequential_solve_inplace( matrix, v );
        mdd.b_ijK_storage( i, j, K, 6 ) = v[ 0 ] * D;
        mdd.b_ijK_storage( i, j, K, 7 ) = v[ 1 ] * D;
        mdd.b_ijK_storage( i, j, K, 8 ) = v[ 2 ] * D;
        mdd.b_ijK_storage( i, j, K, 9 ) = v[ 3 ] * D;

        // the last 4 values are the sums for the b_ijK coefficients
        mdd.b_ijK_storage( i, j, K, 10 ) = mdd.b_ijK_storage( i, j, K, 0 ) + mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 6 );
        mdd.b_ijK_storage( i, j, K, 11 ) = mdd.b_ijK_storage( i, j, K, 1 ) + mdd.b_ijK_storage( i, j, K, 2 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 7 );
        mdd.b_ijK_storage( i, j, K, 12 ) = mdd.b_ijK_storage( i, j, K, 3 ) + mdd.b_ijK_storage( i, j, K, 4 ) + mdd.b_ijK_storage( i, j, K, 5 ) + mdd.b_ijK_storage( i, j, K, 8 );
        mdd.b_ijK_storage( i, j, K, 13 ) = mdd.b_ijK_storage( i, j, K, 6 ) + mdd.b_ijK_storage( i, j, K, 7 ) + mdd.b_ijK_storage( i, j, K, 8 ) + mdd.b_ijK_storage( i, j, K, 9 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKef( const MeshDependentData & mdd,
             const LocalIndex i,
             const LocalIndex j,
             const GlobalIndex K,
             LocalIndex e,
             LocalIndex f )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );
        TNL_ASSERT_LT( f, getFacesPerCell< MeshEntity >(), "" );

        if( e > f )
            TNL::swap( e, f );

        return mdd.b_ijK_storage( i, j, K, e + ( f * (f+1) ) / 2 );
    }

    template< typename MeshDependentData >
    __cuda_callable__
    static typename MeshDependentData::RealType
    b_ijKe( const MeshDependentData & mdd,
            const LocalIndex i,
            const LocalIndex j,
            const GlobalIndex K,
            const LocalIndex e )
    {
        TNL_ASSERT_LT( e, getFacesPerCell< MeshEntity >(), "" );

        return mdd.b_ijK_storage( i, j, K, 10 + e );
    }
};

} // namespace TNL::MHFEM
