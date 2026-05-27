#include "SpawnPhase.hpp"

#include <deque>
#include <optional>
#include <vector>

#include "PhaseHelpers.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

namespace {

struct SpawnPlan {
    Player::id_type player_id{Player::None};
    const SpawnCommand *command = nullptr;
    energy_type cost{};
};

} // namespace

void SpawnPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    static constexpr auto MAX_SPAWNS_PER_TURN = 1;

    std::vector<std::pair<Player::id_type, const SpawnCommand *>> commands;
    for (const auto &[player_id, spawns] : result.validated_commands->spawns) {
        if (spawns.size() > MAX_SPAWNS_PER_TURN) {
            std::deque<std::reference_wrapper<const Command>> spawns_deque{spawns.begin(), spawns.end()};
            const Command &legal = spawns_deque.front();
            spawns_deque.pop_front();
            const Command &illegal = spawns_deque.front();
            spawns_deque.pop_front();
            ErrorContext error_context;
            error_context.push_back(legal);
            for (const Command &spawn : spawns_deque) {
                error_context.push_back(spawn);
            }
            append_error(result, std::make_unique<ExcessiveSpawnsError>(player_id, illegal, error_context));
            continue;
        }
        for (const SpawnCommand &spawn : spawns) {
            commands.emplace_back(player_id, &spawn);
        }
    }

    auto plans = parallel_plan<SpawnPlan>(
        context,
        commands.size(),
        [&](std::size_t index, std::vector<std::optional<SpawnPlan>> &plans) {
            const auto [player_id, command] = commands[index];
            auto &player = store.get_player(player_id);
            SpawnPlan plan{};
            plan.player_id = player_id;
            plan.command = command;
            plan.cost = scaled_spawn_cost(
                context.config().ruleset.economy.new_entity_energy_cost,
                player.entities.size(),
                context.config().ruleset.economy.spawn_cost_growth,
                context.config().ruleset.economy.spawn_quad_threshold,
                context.config().ruleset.economy.spawn_quad_growth);
            plans[index] = plan;
        });

    for_each_planned(plans, [&](const auto &plan) {
        auto &player = store.get_player(plan.player_id);
        player.energy -= plan.cost;
        auto &cell = map.at(player.factory);
        auto &entity = store.new_entity(0, player.id);
        entity.spawn_turn = context.state().turn.number;
        player.add_entity(entity.id, player.factory);
        result.changed_entities.emplace(entity.id);
        context.event_sink().emit(events::SpawnedEvent{player.id, entity.id, player.factory, 0});
        if (cell.entity == Entity::None) {
            cell.entity = entity.id;
        } else {
            auto &existing_entity = store.get_entity(cell.entity);
            auto &existing_player = store.get_player(existing_entity.owner);
            auto &owner = store.get_player(cell.owner);
            if (existing_entity.owner == cell.owner) {
                append_error(result, std::make_unique<SelfCollisionError<SpawnCommand>>(plan.player_id, *plan.command, ErrorContext(), player.factory,
                                                                                        std::vector<Entity::id_type>{cell.entity, entity.id},
                                                                                        !context.config().match.strict_errors));
            }
            context.event_sink().emit(events::CollisionResolvedEvent{owner.factory, std::vector<Entity::id_type>{cell.entity, entity.id}});
            dump_energy(store, existing_entity, owner.factory, cell, existing_entity.energy);
            existing_player.remove_entity(cell.entity);
            store.delete_entity(cell.entity);
            player.remove_entity(entity.id);
            store.delete_entity(entity.id);
            cell.entity = Entity::None;
        }
        result.changed_cells.emplace(player.factory);
    });

    const auto &economy = context.config().ruleset.economy;
    const auto turn_no = context.state().turn.number;
    if (economy.emergency_spawn_enabled
            && economy.emergency_spawn_period > 0
            && economy.emergency_spawn_count > 0
            && turn_no > 0
            && turn_no % economy.emergency_spawn_period == 0) {
        for (auto &[player_id, player] : store.players_ref()) {
            (void)player_id;
            if (!player.entities.empty()) continue;
            if (player.energy >= economy.new_entity_energy_cost) continue;

            std::vector<Location> placements;
            auto &factory_cell = map.at(player.factory);
            if (factory_cell.entity == Entity::None) {
                placements.push_back(player.factory);
            }
            for (const auto &neighbour : map.get_neighbors(player.factory)) {
                if (placements.size() >= economy.emergency_spawn_count) break;
                auto &nc = map.at(neighbour);
                if (nc.entity == Entity::None) placements.push_back(neighbour);
            }
            if (placements.empty()) continue;

            const auto to_spawn = std::min<std::size_t>(placements.size(), economy.emergency_spawn_count);
            for (std::size_t i = 0; i < to_spawn; ++i) {
                const Location loc = placements[i];
                auto &cell = map.at(loc);
                auto &entity = store.new_entity(0, player.id);
                entity.spawn_turn = turn_no;
                entity.protection_turns = static_cast<int>(economy.emergency_protection_turns);
                entity.is_defending = entity.protection_turns > 0;
                player.add_entity(entity.id, loc);
                cell.entity = entity.id;
                result.changed_entities.emplace(entity.id);
                result.changed_cells.emplace(loc);
                context.event_sink().emit(events::SpawnedEvent{player.id, entity.id, loc, 0});
            }
            if (economy.emergency_halite_bonus > 0) {
                player.energy += economy.emergency_halite_bonus;
            }
        }
    }
}

} // namespace hlt::rules::phases
