#include "ConstructionPhase.hpp"

#include <optional>
#include <vector>

#include "PhaseHelpers.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

namespace {

struct ConstructionPlan {
    Player::id_type player_id{Player::None};
    const ConstructCommand *command = nullptr;
    Entity::id_type entity_id{Entity::None};
    Location location{0, 0};
    energy_type cost{};
    bool entity_missing = false;
    bool cell_owned = false;
    Player::id_type cell_owner{Player::None};
};

} // namespace

void ConstructionPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    const auto base_cost = context.config().ruleset.economy.dropoff_cost;
    const auto growth = context.config().ruleset.economy.dropoff_cost_growth;

    auto commands = flatten_player_commands<ConstructCommand>(result.validated_commands->constructs);
    auto plans = parallel_plan<ConstructionPlan>(
        context,
        commands.size(),
        [&](std::size_t index, std::vector<std::optional<ConstructionPlan>> &plans) {
            const auto [player_id, command] = commands[index];
            auto &player = store.get_player(player_id);
            ConstructionPlan plan{};
            plan.player_id = player_id;
            plan.command = command;
            plan.entity_id = command->entity;
            plan.cost = scaled_dropoff_cost(base_cost, player.dropoffs.size(), growth);

            if (!player.has_entity(command->entity)) {
                plan.entity_missing = true;
                plans[index] = plan;
                return;
            }

            plan.location = player.get_entity_location(command->entity);
            const auto &cell = map.at(plan.location);
            if (cell.owner != Player::None) {
                plan.cell_owned = true;
                plan.cell_owner = cell.owner;
            }
            plans[index] = plan;
        });

    for_each_planned(plans, [&](const auto &plan) {
        if (plan.entity_missing) {
            append_error(result, std::make_unique<EntityNotFoundError<ConstructCommand>>(plan.player_id, *plan.command));
            return;
        }
        if (plan.cell_owned) {
            append_error(result, std::make_unique<CellOwnedError<ConstructCommand>>(plan.player_id, *plan.command, plan.location, plan.cell_owner));
            return;
        }

        auto &player = store.get_player(plan.player_id);
        if (!player.has_entity(plan.entity_id)) {
            return;
        }
        const auto &entity = store.get_entity(plan.entity_id);
        auto &cell = map.at(plan.location);

        cell.owner = plan.player_id;
        player.dropoffs.emplace_back(store.new_dropoff(plan.location));
        auto &created_dropoff = player.dropoffs.back();
        created_dropoff.halite_pool = context.config().ruleset.economy.initial_dropoff_halite;
        created_dropoff.destroyed = false;
        store.map_total_energy -= cell.energy;

        const auto credit = cell.energy + entity.energy;
        {
            auto &mut_entity = store.get_entity(plan.entity_id);
            mut_entity.lifetime_deposited += mut_entity.energy;
        }
        cell.energy = 0;
        cell.entity = Entity::None;
        result.changed_cells.emplace(plan.location);
        dump_energy(store, plan.location, cell, credit);
        player.energy -= plan.cost;
        player.remove_entity(plan.entity_id);
        store.delete_entity(plan.entity_id);
        context.event_sink().emit(events::ConstructionResolvedEvent{plan.location, plan.player_id, plan.command->entity});
    });
}

} // namespace hlt::rules::phases
