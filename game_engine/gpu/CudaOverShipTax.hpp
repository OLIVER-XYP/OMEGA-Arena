#ifndef CUDAOVERSHIPTAX_HPP
#define CUDAOVERSHIPTAX_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_over_ship_tax(StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAOVERSHIPTAX_HPP
