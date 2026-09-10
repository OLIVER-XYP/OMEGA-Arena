#ifndef STEPRESULT_HPP
#define STEPRESULT_HPP

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "CommandBatch.hpp"
#include "Entity.hpp"
#include "GameEvent.hpp"
#include "Location.hpp"
#include "Player.hpp"

namespace hlt {

struct StepResult {
    std::vector<std::string> events;
    std::vector<std::string> non_fatal_errors;
    std::vector<Player::id_type> eliminated_players;
    std::vector<GameEvent> replay_events;
    std::optional<CommandBatch> validated_commands;
    std::unordered_set<Entity::id_type> changed_entities;
    std::unordered_set<Location> changed_cells;
};

} // namespace hlt

#endif // STEPRESULT_HPP
