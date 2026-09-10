#include "CudaDefend.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_defend(StateFrame &state_frame, const CommandFrame &commands, const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA defend kernel was not built");
}

} // namespace hlt::gpu
