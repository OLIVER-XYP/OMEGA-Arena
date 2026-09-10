#ifndef CUDACONSTRUCTION_HPP
#define CUDACONSTRUCTION_HPP

#include <vector>

#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

struct ConstructionFrameDecision {
    Player::id_type player{Player::None};
    Entity::id_type entity{Entity::None};
    Location location{0, 0};
    energy_type cost{};
    Player::id_type cell_owner{Player::None};
    bool command{};
    bool entity_missing{};
    bool cell_owned{};
};

std::vector<ConstructionFrameDecision> run_cuda_construction_decisions(const StateFrame &state_frame,
                                                                       const CommandFrame &commands,
                                                                       const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDACONSTRUCTION_HPP
