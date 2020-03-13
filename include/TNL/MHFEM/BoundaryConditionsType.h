#pragma once

#include <cstdint>

namespace mhfem {

enum class BoundaryConditionsType
: std::uint8_t
{
    // fixed-value (Dirichlet) boundary condition
    // (sets Z_iE := value)
    FixedValue = 0,

    // fixed-flux (Neumann) boundary condition
    // (solves  (-m_i \vec v_i + \sum_j Z_j \vec a_ij) \cdot \vec n = J_Neu  on the face,
    // where  \vec v_i = - \sum_j D_ij \grad Z_j + \vec w_i  is the diffusive velocity
    // WARNING:
    //  - m_i must be non-zero (positive) on the boundary if \vec v_i \cdot n_E < 0
    //  - the coefficient \vec u_ij is not included in this boundary condition
    FixedFlux = 1,

    // advective outflow boundary condition
    // (sets  \vec v_i = 0  on the face, where \vec v_i is the diffusive velocity
    // defined as  \vec v_i = - \sum_j D_ij \grad Z_j + \vec w_i)
    AdvectiveOutflow = 2
};

} // namespace mhfem
