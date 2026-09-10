#include "CudaMining.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_mining(StateFrame &state_frame,
                     const TurnFrame &turn_frame,
                     const GameConfig &config,
                     std::vector<MiningEffectFrameEntry> *effects) {
    (void)state_frame;
    (void)turn_frame;
    (void)config;
    (void)effects;
    throw std::runtime_error("CUDA mining kernel was not built");
}

void run_cuda_mining_and_plunder(StateFrame &state_frame,
                                 const TurnFrame &turn_frame,
                                 const GameConfig &config,
                                 std::vector<MiningEffectFrameEntry> *effects) {
    (void)state_frame;
    (void)turn_frame;
    (void)config;
    (void)effects;
    throw std::runtime_error("CUDA mining/plunder kernel was not built");
}

} // namespace hlt::gpu
