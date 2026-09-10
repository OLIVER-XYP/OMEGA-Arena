#include "CudaRegen.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_regen(StateFrame &state_frame, const GameConfig &config) {
    (void)state_frame;
    (void)config;
    throw std::runtime_error("CUDA regen kernel was not built");
}

} // namespace hlt::gpu
