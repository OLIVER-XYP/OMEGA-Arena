#ifndef CUDASPAWN_HPP
#define CUDASPAWN_HPP

#include <vector>

#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

struct SpawnFrameDecision {
    Player::id_type player{Player::None};
    Location factory{0, 0};
    Entity::id_type existing_entity{Entity::None};
    Player::id_type cell_owner{Player::None};
    energy_type cost{};
    bool command{};
    bool factory_occupied{};
    bool self_collision{};
};

struct EmergencySpawnFrameDecision {
    Player::id_type player{Player::None};
    unsigned long count{};
    Location locations[5]{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
};

struct SpawnFrameDecisions {
    std::vector<SpawnFrameDecision> spawns;
    std::vector<EmergencySpawnFrameDecision> emergency_spawns;
};

SpawnFrameDecisions run_cuda_spawn_decisions(const StateFrame &state_frame,
                                             const CommandFrame &commands,
                                             const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDASPAWN_HPP
