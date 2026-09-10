#include "CudaValidation.hpp"

#include <stdexcept>

namespace hlt::gpu {

std::vector<ValidationFrameDecision> run_cuda_validation_decisions(const StateFrame &state_frame,
                                                                   const CommandFrame &commands,
                                                                   const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA validation kernel was not built");
}

} // namespace hlt::gpu
