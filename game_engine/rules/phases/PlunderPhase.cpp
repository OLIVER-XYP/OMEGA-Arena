#include "PlunderPhase.hpp"

#include <cmath>
#include <unordered_set>

namespace hlt::rules::phases {

// Plunder: ships within PLUNDER_RANGE of an enemy structure (shipyard or dropoff)
// gain PLUNDER_HALITE_PER_TURN halite each turn. Rewards aggressive positioning
// (camping near enemy structures) with passive income — purely asymmetric,
// benefiting aggro > eco > control.
void PlunderPhase::execute(RuleContext &context) const {
    const auto &economy = context.config().ruleset.economy;
    if (economy.plunder_range <= 0 || economy.plunder_halite_per_turn <= 0) {
        return;
    }

    auto &store = context.state().store_ref();
    const auto &map = context.state().map_ref();
    const auto plunder_range = economy.plunder_range;
    const auto plunder_halite = economy.plunder_halite_per_turn;

    // Collect enemy structure locations per player.
    std::unordered_map<Player::id_type, std::vector<Location>> player_structures;
    for (const auto &[pid, player] : store.players_ref()) {
        auto &vec = player_structures[pid];
        vec.push_back(player.factory); // shipyard
        for (const auto &dropoff : player.dropoffs) {
            vec.push_back(dropoff.location);
        }
    }

    for (auto &[entity_id, entity] : store.entities_ref()) {
        if (entity.energy >= economy.max_energy) continue;

        const auto &player = store.players_ref().at(entity.owner);
        const auto location = player.get_entity_location(entity_id);

        // Check if this ship is within plunder_range of any ENEMY structure.
        for (const auto &[pid, structures] : player_structures) {
            if (pid == entity.owner) continue; // skip own structures
            for (const auto &struct_loc : structures) {
                if (map.distance(location, struct_loc) <= plunder_range) {
                    entity.energy = std::min(economy.max_energy,
                                             entity.energy + plunder_halite);
                    goto next_entity; // one plunder bonus per ship per turn
                }
            }
        }
        next_entity:;
    }
}

} // namespace hlt::rules::phases
