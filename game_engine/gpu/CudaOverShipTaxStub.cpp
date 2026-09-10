#include "CudaOverShipTax.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_over_ship_tax(StateFrame &state_frame, const GameConfig &config) {
    (void)state_frame;
    (void)config;
    throw std::runtime_error("CUDA over-ship tax kernel was not built");
}

} // namespace hlt::gpu
