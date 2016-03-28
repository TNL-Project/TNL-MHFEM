#pragma once

#include <core/vectors/tnlSharedVector.h>

#include "MassMatrix.h"

namespace mhfem
{

template< typename MeshDependentData, MassLumping = MeshDependentData::MassMatrix::lumping >
class MassMatrixDependentCode
{};

template< typename MeshDependentData >
class MassMatrixDependentCode< MeshDependentData, MassLumping::enabled >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;
    using SharedVectorType = tnlSharedVector< RealType, DeviceType, IndexType >;

    // NOTE: MeshDependentData cannot be const, because tnlSharedVector expects RealType*, does not work with const RealType*
    __cuda_callable__
    static inline RealType
    A_ijKEF( MeshDependentData & mdd,
             const int & i,
             const int & j,
             const IndexType & K,
             const IndexType & E,
             const int & e,
             const IndexType & F,
             const int & f )
    {
        RealType value = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            SharedVectorType storage( mdd.b_ijK( i, xxx, K ), MassMatrix::size );
            value -= MassMatrix::get( e, storage ) * mdd.R_ijKe( xxx, j, K, f );
            // TODO: maybe the condition is useless, if we happen to add 0.0, there is no additional overhead involving global memory read
            if( xxx == j && E == F )
                value += MassMatrix::get( e, storage );
        }
        return value;
    }
};

template< typename MeshDependentData >
class MassMatrixDependentCode< MeshDependentData, MassLumping::disabled >
{
public:
    using RealType = typename MeshDependentData::RealType;
    using DeviceType = typename MeshDependentData::DeviceType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;
    using SharedVectorType = tnlSharedVector< RealType, DeviceType, IndexType >;

    // NOTE: MeshDependentData cannot be const, because tnlSharedVector expects RealType*, does not work with const RealType*
    __cuda_callable__
    static inline RealType
    A_ijKEF( MeshDependentData & mdd,
             const int & i,
             const int & j,
             const IndexType & K,
             const IndexType & E,
             const int & e,
             const IndexType & F,
             const int & f )
    {
        RealType value = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            SharedVectorType storage( mdd.b_ijK( i, xxx, K ), MassMatrix::size );
            value += MassMatrix::get( e, f, storage );
            value -= MassMatrix::get( e, storage ) * mdd.R_ijKe( xxx, j, K, f );
        }
        return value;
    }
};

} // namespace mhfem
