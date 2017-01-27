#pragma once

#include "MassLumpingDependentCoefficients.h"

namespace mhfem {

template< typename MeshDependentData >
struct SecondaryCoefficients
    : public MassLumpingDependentCoefficients< MeshDependentData >
{
    using RealType = typename MeshDependentData::RealType;
    using IndexType = typename MeshDependentData::IndexType;
    using MassMatrix = typename MeshDependentData::MassMatrix;

    __cuda_callable__
    static inline RealType
    A_ijKEF( const MeshDependentData & mdd,
             const int & i,
             const int & j,
             const IndexType & K,
             const IndexType & E,
             const int & e,
             const IndexType & F,
             const int & f )
    {
//        RealType value = MassMatrix::b_ijKef( mdd, i, j, K, e, f );
//        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
//            value -= MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
//        }
//        return value;
        // more careful version with only one subtraction to avoid catastrophic truncation
        RealType sum = 0.0;
        for( int xxx = 0; xxx < MeshDependentData::NumberOfEquations; xxx++ ) {
            sum += MassMatrix::b_ijKe( mdd, i, xxx, K, e ) * mdd.R_ijKe( xxx, j, K, f );
        }
        return MassMatrix::b_ijKef( mdd, i, j, K, e, f ) - sum;
    }
};

} // namespace mhfem
