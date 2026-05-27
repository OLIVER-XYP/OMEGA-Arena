#include "InspirationPhase.hpp"

#include <optional>
#include <vector>

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

namespace {

struct InspirationDecision {
    Entity::id_type entity_id{Entity::None};
    bool inspired{};
};

} // namespace

void InspirationPhase::execute(RuleContext &context) const {
    const auto &config = context.config().ruleset.inspiration;
    if (!config.enabled) {
        return;
    }

    auto &state = context.state();
    auto &store = state.store_ref();
    auto &map = state.map_ref();
    const auto inspiration_radius = config.radius;
    const auto ships_threshold = config.ship_count;

    std::vector<std::tuple<Player::id_type, Entity::id_type, Location>> entities;
    for (const auto &[player_id, player] : store.players_ref()) {
        for (const auto &[entity_id, location] : player.entities) {
            entities.emplace_back(player_id, entity_id, location);
        }
    }

    auto decisions = parallel_plan<InspirationDecision>(
        context,
        entities.size(),
        [&](std::size_t index, std::vector<std::optional<InspirationDecision>> &plans) {
            const auto &[player_id, entity_id, location] = entities[index];
            id_map<Player, long> ships_in_radius;
            for (const auto &[pid, _] : store.players_ref()) {
                ships_in_radius[pid] = 0;
            }

            for (auto dx = -inspiration_radius; dx <= inspiration_radius; dx++) {
                for (auto dy = -inspiration_radius; dy <= inspiration_radius; dy++) {
                    const auto cur = Location{
                        (((location.x + dx) % map.width) + map.width) % map.width,
                        (((location.y + dy) % map.height) + map.height) % map.height
                    };

                    const auto &cur_cell = map.at(cur);
                    if (cur_cell.entity == Entity::None || map.distance(location, cur) > inspiration_radius) {
                        continue;
                    }
                    const auto &other_entity = store.get_entity(cur_cell.entity);
                    ships_in_radius[other_entity.owner]++;
                }
            }

            unsigned long opponent_entities = 0;
            for (const auto &[pid, count] : ships_in_radius) {
                if (pid != player_id) {
                    opponent_entities += count;
                }
            }

            plans[index] = InspirationDecision{entity_id, opponent_entities >= ships_threshold};
        });

    for_each_planned(decisions, [&](const auto &decision) {
        auto &entity = store.get_entity(decision.entity_id);
        entity.is_inspired = decision.inspired;
    });
}

} // namespace hlt::rules::phases
