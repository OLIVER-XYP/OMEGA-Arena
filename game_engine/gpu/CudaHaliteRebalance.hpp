#ifndef CUDAHALITEREBALANCE_HPP
#define CUDAHALITEREBALANCE_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_halite_rebalance(StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAHALITEREBALANCE_HPP
