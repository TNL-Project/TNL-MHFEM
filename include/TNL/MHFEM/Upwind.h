#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Functions/Range.h>
#include <TNL/SharedPointer.h>

#include "MassMatrixDependentCode.h"
#include "../lib_general/mesh_helpers.h"

namespace mhfem
{

// TODO: make mdd->getBoundaryMobility a parameter
//       (it should be possible to upwind any quantity, not just mobility)
template< typename Mesh,
          typename MeshDependentData,
          typename BoundaryConditions >
class Upwind
    : public TNL::Functions::Domain< Mesh::meshDimensions, TNL::Functions::MeshDomain >,
      public TNL::Functions::Range< typename MeshDependentData::RealType, MeshDependentData::NumberOfEquations >
{
public:
    typedef Mesh MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef MeshDependentData MeshDependentDataType;
    typedef typename MeshDependentDataType::RealType RealType;
    typedef typename MeshDependentDataType::DeviceType DeviceType;
    typedef typename MeshDependentDataType::IndexType IndexType;
    typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
    typedef TNL::Containers::StaticVector< MeshDependentDataType::FacesPerCell, IndexType > FaceVectorType;
    typedef MassMatrixDependentCode< MeshDependentDataType > coeff;

    static constexpr int getEntitiesDimensions() { return Mesh::meshDimensions - 1; }
 
    void bind( MeshDependentDataType* mdd,
               TNL::SharedPointer< BoundaryConditions > & bc,
               TNL::SharedPointer< DofVectorType > & Z_iF )
    {
        this->mdd = mdd;
        this->bc = bc;
        this->Z_iF = Z_iF;
    }

    // FIXME: velocities should be pre-calculated before determining upwind values and updating the pressure values
    __cuda_callable__
    RealType getVelocity( const MeshType & mesh,
                          const int & i,
                          const IndexType & K,
                          const IndexType & E ) const
    {
        // find local index of face E
        FaceVectorType faceIndexes;
        getFacesForCell( mesh, K, faceIndexes );
        const int e = getLocalIndex( faceIndexes, E );

        // dereference the smart pointer on device
        const auto & Z_iF = this->Z_iF.template getData< DeviceType >();

        return coeff::v_iKE( *mdd, Z_iF, faceIndexes, i, K, E, e );
    }

    template< typename EntityType >
    __cuda_callable__
    RealType operator()( const EntityType & entity,
                         const RealType & time,
                         const int & i ) const
    {
        static_assert( EntityType::getDimensions() == getEntitiesDimensions(),
                       "This function is defined on faces." );

        const MeshType & mesh = entity.getMesh();
        const IndexType E = entity.getIndex();

        IndexType cellIndexes[ 2 ];
        const int numCells = getCellsForFace( mesh, entity, cellIndexes );
        
        // index of the main element (left/bottom if indexFace is inner face, otherwise the element next to the boundary face)
        const IndexType & K1 = cellIndexes[ 0 ];

        if( getVelocity( mesh, i, K1, E ) >= 0 ) {
            return mdd->m_iK( i, K1 );
        }
        else if( numCells == 2 ) {
            const IndexType & K2 = cellIndexes[ 1 ];
            return mdd->m_iK( i, K2 );
        }
        else {
            // dereference the smart pointer on device
            const auto & bc = this->bc.template getData< DeviceType >();

            // TODO: check if the value is available (we need to know the density on \Gamma_c ... part of the boundary where the fluid flows in)
//            return mdd->m_iK( i, K1 );
            return mdd->getBoundaryMobility( mesh, bc, i, entity, time );
        }
    }

protected:
    MeshDependentDataType* mdd;
    TNL::SharedPointer< BoundaryConditions > bc;
    TNL::SharedPointer< DofVectorType > Z_iF;
};

} // namespace mhfem
