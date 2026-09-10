#include "CudaHaliteRebalance.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_halite_rebalance(StateFrame &state_frame, const GameConfig &config) {
    (void)state_frame;
    (void)config;
    throw std::runtime_error("CUDA halite rebalance kernel was not built");
}

} // namespace hlt::gpu
