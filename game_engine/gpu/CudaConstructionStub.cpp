#include "CudaConstruction.hpp"

#include <stdexcept>

namespace hlt::gpu {

std::vector<ConstructionFrameDecision> run_cuda_construction_decisions(const StateFrame &state_frame,
                                                                       const CommandFrame &commands,
                                                                       const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA construction kernel was not built");
}

} // namespace hlt::gpu
