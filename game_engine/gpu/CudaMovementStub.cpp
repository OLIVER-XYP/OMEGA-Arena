#include "CudaMovement.hpp"

#include <stdexcept>

namespace hlt::gpu {

std::vector<MovementFrameDecision> run_cuda_movement_decisions(const StateFrame &state_frame,
                                                               const CommandFrame &commands,
                                                               const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA movement kernel was not built");
}

} // namespace hlt::gpu
