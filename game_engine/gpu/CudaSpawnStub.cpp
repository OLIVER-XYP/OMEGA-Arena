#include "CudaSpawn.hpp"

#include <stdexcept>

namespace hlt::gpu {

SpawnFrameDecisions run_cuda_spawn_decisions(const StateFrame &state_frame,
                                             const CommandFrame &commands,
                                             const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA spawn kernel was not built");
}

} // namespace hlt::gpu
