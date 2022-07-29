#pragma once

#include <TNL/Containers/Array.h>

#include "BoundaryConditionsType.h"
#include "BoundaryConditionsStorage.h"

namespace mhfem
{

template< typename MeshDependentData,
          typename BoundaryModel >
class BoundaryConditions
{
public:
    using MeshType = typename MeshDependentData::MeshType;
    using MeshDependentDataType = MeshDependentData;
    using DeviceType = typename MeshType::DeviceType;
    using RealType = typename MeshDependentDataType::RealType;
    using IndexType = typename MeshDependentDataType::IndexType;
    using TagArrayType = TNL::Containers::Array< BoundaryConditionsType, DeviceType, IndexType >;
    using ValueArrayType = TNL::Containers::Array< RealType, DeviceType, IndexType >;

    void init( const BoundaryConditionsStorage< RealType > & storage );

    __cuda_callable__
    IndexType getLinearSystemRowLength( const MeshType & mesh,
                                        const IndexType E,
                                        const int i ) const;

    __cuda_callable__
    IndexType getLinearSystemRowLengthDiag( const MeshType & mesh,
                                            const IndexType E,
                                            const int i ) const;

    template< typename Matrix, typename Vector >
    __cuda_callable__
    void setMatrixElements( const MeshType & mesh,
                            const MeshDependentDataType & mdd,
                            const IndexType rowIndex,
                            const IndexType E,
                            const int i,
                            const RealType time,
                            const RealType tau,
#ifdef HAVE_HYPRE
                            Matrix & diag,
                            Matrix & offd,
#else
                            Matrix & matrix,
#endif
                            Vector & b ) const;


    __cuda_callable__
    RealType
    getNeumannValue( const MeshType & mesh,
                     const int i,
                     const IndexType E,
                     const RealType time,
                     const RealType tau ) const
    {
        return BoundaryModel::getNeumannValue( *this, mesh, i, E, time, tau );
    }

    __cuda_callable__
    RealType
    getDirichletValue( const MeshType & mesh,
                       const int i,
                       const IndexType E,
                       const RealType time,
                       const RealType tau ) const
    {
        return BoundaryModel::getDirichletValue( *this, mesh, i, E, time, tau );
    }

    // array holding tags to differentiate the boundary condition based on the face index
    TagArrayType tags;

    // arrays holding the values to be interpreted by the boundary condition specified on each face
    ValueArrayType values;
    // Dirichlet values are "special" - boundary values may be necessary even for flux-based conditions (upwind on inflow)
    ValueArrayType dirichletValues;
};

} // namespace mhfem

#include "BoundaryConditions_impl.h"
