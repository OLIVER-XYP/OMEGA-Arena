#ifndef CUDAMOVEMENT_HPP
#define CUDAMOVEMENT_HPP

#include <vector>

#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

struct MovementFrameDecision {
    Player::id_type player{Player::None};
    Entity::id_type entity{Entity::None};
    Location from{0, 0};
    Location to{0, 0};
    energy_type required{};
    energy_type current_energy{};
    bool command{};
    bool entity_missing{};
    bool insufficient_energy{};
};

std::vector<MovementFrameDecision> run_cuda_movement_decisions(const StateFrame &state_frame,
                                                               const CommandFrame &commands,
                                                               const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAMOVEMENT_HPP
