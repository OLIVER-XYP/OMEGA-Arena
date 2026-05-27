#include "CapturePhase.hpp"

#include <optional>
#include <unordered_set>
#include <vector>

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

namespace {

struct CaptureDecision {
    Location location{0, 0};
    Player::id_type new_player_id{Player::None};
};

} // namespace

void CapturePhase::execute(RuleContext &context) const {
    const auto &capture = context.config().ruleset.capture;
    if (!capture.enabled) {
        return;
    }

    auto &state = context.state();
    auto &store = state.store_ref();
    auto &map = state.map_ref();

    std::vector<std::tuple<Player::id_type, Entity::id_type, Location>> entities;
    for (const auto &[player_id, player] : store.players_ref()) {
        for (const auto &[entity_id, location] : player.entities) {
            entities.emplace_back(player_id, entity_id, location);
        }
    }

    auto decisions = parallel_plan<CaptureDecision>(
        context,
        entities.size(),
        [&](std::size_t index, std::vector<std::optional<CaptureDecision>> &plans) {
            const auto &[player_id, entity_id, location] = entities[index];
            (void)entity_id;

            id_map<Player, unsigned long> ships_in_radius;
            for (const auto &[pid, _] : store.players_ref()) {
                ships_in_radius[pid] = 0;
            }

            std::unordered_set<Location> visited_locs{};
            std::unordered_set<Location> to_visit_locs{location};
            while (!to_visit_locs.empty()) {
                const Location cur = *to_visit_locs.begin();
                visited_locs.emplace(cur);
                to_visit_locs.erase(to_visit_locs.begin());

                Cell cur_cell = map.at(cur);
                if (cur_cell.entity != Entity::None) {
                    // A cell may carry a stale entity id (e.g. a ship killed in
                    // CombatPhase whose cell was not cleared).  get_entity()
                    // asserts on a missing id, which is a no-op in Release and
                    // dereferences end() -> access violation.  Guard with find().
                    const auto &all_entities = store.entities_ref();
                    const auto it = all_entities.find(cur_cell.entity);
                    if (it != all_entities.end()) {
                        ships_in_radius[it->second.owner]++;
                    }
                }

                for (const Location &neighbor : map.get_neighbors(cur)) {
                    if ((visited_locs.find(neighbor) == visited_locs.end())
                        && (to_visit_locs.find(neighbor) == to_visit_locs.end())) {
                        if (map.distance(location, neighbor) <= capture.radius) {
                            to_visit_locs.emplace(neighbor);
                        } else {
                            visited_locs.emplace(neighbor);
                        }
                    }
                }
            }

            unsigned long max_val = 0;
            Player::id_type max_id = Player::None;
            for (const auto &[pid, val] : ships_in_radius) {
                if (pid != player_id && val > max_val) {
                    max_val = val;
                    max_id = pid;
                }
            }
            if (ships_in_radius[player_id] + capture.ships_above_for_capture <= max_val) {
                plans[index] = CaptureDecision{location, max_id};
            }
        });

    std::unordered_map<Location, Player::id_type> entity_switches;
    for (const auto &decision : decisions) {
        if (decision.has_value()) {
            entity_switches[decision->location] = decision->new_player_id;
        }
    }

    for (const auto &[location, new_player_id] : entity_switches) {
        auto &cell = map.at(location);
        // Skip if the cell no longer holds a live entity (stale/cleared ref).
        if (cell.entity == Entity::None
                || store.entities_ref().find(cell.entity) == store.entities_ref().end()) {
            continue;
        }
        const auto entity = store.get_entity(cell.entity);
        const auto old_owner = entity.owner;
        const auto old_id = entity.id;
        const auto old_energy = entity.energy;

        store.get_player(old_owner).remove_entity(old_id);
        store.delete_entity(old_id);

        auto &new_entity = store.new_entity(old_energy, new_player_id);
        new_entity.spawn_turn = context.state().turn.number;
        cell.entity = new_entity.id;
        store.get_entity(new_entity.id).was_captured = true;
        store.get_player(new_player_id).add_entity(new_entity.id, location);
        context.result().changed_cells.emplace(location);
        store.changed_cells_ref().emplace(location);
        context.event_sink().emit(events::CapturedEvent{location, old_owner, old_id, new_player_id, new_entity.id});
    }
}

} // namespace hlt::rules::phases
