#ifndef CUDACOMBAT_HPP
#define CUDACOMBAT_HPP

#include <vector>

#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

struct CombatFrameDecision {
    Player::id_type player{Player::None};
    Entity::id_type attacker{Entity::None};
    Location attacker_location{0, 0};
    Entity::id_type target{Entity::None};
    Player::id_type target_owner{Player::None};
    Location target_location{0, 0};
    bool command{};
    bool structure_target{};
    bool invalid{};
    bool valid_structure{};
    bool valid_ship{};
};

std::vector<CombatFrameDecision> run_cuda_combat_decisions(const StateFrame &state_frame,
                                                           const CommandFrame &commands,
                                                           const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDACOMBAT_HPP
