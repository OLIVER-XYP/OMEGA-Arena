#ifndef CUDAMINING_HPP
#define CUDAMINING_HPP

#include <vector>

#include "GameConfig.hpp"
#include "StateFrame.hpp"
#include "TurnFrame.hpp"

namespace hlt::gpu {

struct MiningEffectFrameEntry {
    Entity::id_type entity{Entity::None};
    Player::id_type owner{Player::None};
    Location location{0, 0};
    energy_type extracted{};
    energy_type gained{};
    bool was_captured{};
};

void run_cuda_mining(StateFrame &state_frame,
                     const TurnFrame &turn_frame,
                     const GameConfig &config,
                     std::vector<MiningEffectFrameEntry> *effects = nullptr);
void run_cuda_mining_and_plunder(StateFrame &state_frame,
                                 const TurnFrame &turn_frame,
                                 const GameConfig &config,
                                 std::vector<MiningEffectFrameEntry> *effects = nullptr);

} // namespace hlt::gpu

#endif // CUDAMINING_HPP
