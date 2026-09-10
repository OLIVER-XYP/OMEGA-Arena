#ifndef UNITS_HPP
#define UNITS_HPP

namespace hlt {

/** The type of dimension values across all Halite classes using dimension. */
using dimension_type = long;

/** The type of energy/production values across all Halite classes. */
// int (32-bit): the CUDA batched dump atomics static_assert sizeof(energy_type)
// == sizeof(int), and Halite values (cell/cargo/bank/scores <= ~300k) fit int32.
using energy_type = int;

}

#endif // UNITS_HPP
