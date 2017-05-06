#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>
#include <TNL/SharedPointer.h>

#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

// TODO: make mdd->getBoundaryMobility a parameter
//       (it should be possible to upwind any quantity, not just mobility)
template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions >
class Upwind
    : public TNL::Functions::Domain< Mesh::getMeshDimension(), TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using RealType = typename MeshDependentDataType::RealType;
    using DeviceType = typename MeshDependentDataType::DeviceType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType>;

    static constexpr int getEntitiesDimensions() { return Mesh::getMeshDimension() - 1; }
 
    void bind( const TNL::SharedPointer< MeshType > & mesh,
               TNL::SharedPointer< MeshDependentDataType > mdd,
               TNL::SharedPointer< BoundaryConditions > & bc )
    {
        this->mesh = mesh;
        this->mdd = mdd;
        this->bc = bc;
    }

    // FIXME: template needed due to limitation of FunctionAdapter (for grids MeshType::Cell is
    // different from MeshType::template EntityType< d >, because it has non-default Config parameter)
    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getEntityDimension() == getEntitiesDimensions(),
                       "This function is defined on faces." );

        // dereference the smart pointer on device
        const auto & mdd = this->mdd.template getData< DeviceType >();
        const auto & mesh = this->mesh.template getData< DeviceType >();

        const IndexType E = entity.getIndex();

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, entity, cellIndexes );
        
        // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
        const IndexType & K1 = cellIndexes[ 0 ];

        // find local index of face E
        const auto faceIndexes = getFacesForCell( mesh, K1 );
        const int e = getLocalIndex( faceIndexes, E );

        if( numCells == 1 ) {
            // dereference the smart pointer on device
            const auto & bc = this->bc.template getData< DeviceType >();

            // We need to check inflow of ALL phases!
            // FIXME: this assumes two-phase model, general system might be coupled differently or even decoupled
            bool inflow = false;
            for( int j = 0; j < MeshDependentDataType::NumberOfEquations; j++ )
                // Taking the boundary value increases the error, for example in the mcwh3d problem
                // on cubes, so we need to use mdd.v_iKe instead of bc.getNeumannValue
                if( mdd.v_iKe( j, K1, e ) < 0 ) {
                    inflow = true;
                    break;
                }

            if( inflow )
                // The velocity might be negative even on faces with 0 Neumann condition (probably
                // due to rounding errors), so the model must check if the value is available and
                // otherwise return m_iK( i, K1 ).
                return mdd.getBoundaryMobility( mesh, bc, i, entity, time );
            return mdd.m_iK( i, K1 );
        }
        else {
            const IndexType & K2 = cellIndexes[ 1 ];
            // Theoretically, v_iKE is conservative so one might expect that `vel = mdd.v_iKe( i, K1, e )`
            // is enough, but there might be numerical errors. Perhaps more importantly, the main equation
            // might not be based on balancing v_iKE, but some other quantity. We also use a dummy equation
            // if Q_K is singular, so this has significant effect on the error.
            const RealType vel = mdd.v_iKe( i, K1, e ) - mdd.v_iKe( i, K2, e );

            if( vel >= 0.0 )
                return mdd.m_iK( i, K1 );
            else
                return mdd.m_iK( i, K2 );
        }
    }

protected:
    TNL::SharedPointer< MeshType > mesh;
    TNL::SharedPointer< MeshDependentDataType > mdd;
    TNL::SharedPointer< BoundaryConditions > bc;
};


template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions >
class UpwindZ
    : public TNL::Functions::Domain< Mesh::getMeshDimension(), TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations * MeshDependentData::NumberOfEquations >
{
public:
    using MeshType = Mesh;
    using MeshDependentDataType = MeshDependentData;
    using RealType = typename MeshDependentDataType::RealType;
    using DeviceType = typename MeshDependentDataType::DeviceType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType>;

    static constexpr int getEntitiesDimensions() { return Mesh::getMeshDimension() - 1; }
 
    void bind( const TNL::SharedPointer< MeshType > & mesh,
               TNL::SharedPointer< MeshDependentDataType > mdd,
               DofVectorType & Z_iF )
    {
        this->mesh = mesh;
        this->mdd = mdd;
        this->Z_iF.bind( Z_iF );
    }

    // FIXME: template needed due to limitation of FunctionAdapter, otherwise we would use MeshType::Face
    // (for grids it is different from MeshType::template EntityType< d >, because it has non-default Config parameter)
    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         // NOTE: xxx should vary between 0 and (MeshDependentData::NumberOfEquations)^2
                         const int & xxx ) const
    {
        static_assert( EntityType::getEntityDimension() == getEntitiesDimensions(),
                       "This function is defined on faces." );

        const int i = xxx / MeshDependentData::NumberOfEquations;
        const int j = xxx % MeshDependentData::NumberOfEquations;

        // dereference the smart pointer on device
        const auto & mdd = this->mdd.template getData< DeviceType >();
        const auto & mesh = this->mesh.template getData< DeviceType >();

        const IndexType E = entity.getIndex();

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, entity, cellIndexes );
        
        // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
        const IndexType & K1 = cellIndexes[ 0 ];

        // find local index of face E
        const auto faceIndexes = getFacesForCell( mesh, K1 );
        const int e = getLocalIndex( faceIndexes, E );

        const RealType a_plus_u = mdd.a_ijKe( i, j, K1, e ) + mdd.u_ijKe( i, j, K1, e );

        if( a_plus_u > 0.0 ) {
            return mdd.Z_iK( j, K1 );
        }
        else if( a_plus_u == 0.0 )
            return 0.0;
        else if( numCells == 2 ) {
            const IndexType & K2 = cellIndexes[ 1 ];
            return mdd.Z_iK( j, K2 );
        }
        else {
            // TODO: this matches the Dirichlet condition, but what happens on Neumann boundary?
            // TODO: at time=0 the value on Neumann boundary is indeterminate
            return Z_iF[ mdd.getDofIndex( j, E ) ];
        }
    }

protected:
    TNL::SharedPointer< MeshType > mesh;
    TNL::SharedPointer< MeshDependentDataType > mdd;
    DofVectorType Z_iF;
};

} // namespace mhfem
