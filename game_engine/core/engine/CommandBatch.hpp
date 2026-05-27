#ifndef COMMANDBATCH_HPP
#define COMMANDBATCH_HPP

#include <functional>
#include <vector>

#include "Command.hpp"
#include "Player.hpp"

namespace hlt {

struct CommandBatch {
    id_map<Player, std::vector<std::reference_wrapper<const ConstructCommand>>> constructs;
    id_map<Player, std::vector<std::reference_wrapper<const DefendCommand>>> defends;
    id_map<Player, std::vector<std::reference_wrapper<const MoveCommand>>> moves;
    id_map<Player, std::vector<std::reference_wrapper<const AttackCommand>>> attacks;
    id_map<Player, std::vector<std::reference_wrapper<const HealCommand>>> heals;
    id_map<Player, std::vector<std::reference_wrapper<const SpawnCommand>>> spawns;
};

} // namespace hlt

#endif // COMMANDBATCH_HPP
