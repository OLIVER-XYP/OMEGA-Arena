#include "CudaCombat.hpp"

#include <stdexcept>

namespace hlt::gpu {

std::vector<CombatFrameDecision> run_cuda_combat_decisions(const StateFrame &state_frame,
                                                           const CommandFrame &commands,
                                                           const GameConfig &config) {
    (void)state_frame;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA combat kernel was not built");
}

} // namespace hlt::gpu
