#include "MovementPhase.hpp"

#include <deque>
#include <optional>
#include <vector>

#include "PhaseHelpers.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

namespace {

struct MovePlan {
    Player::id_type player_id{Player::None};
    const MoveCommand *command = nullptr;
    Entity::id_type entity_id{Entity::None};
    Location from{0, 0};
    Location to{0, 0};
    energy_type required{};
    bool entity_missing = false;
    bool insufficient_energy = false;
    energy_type current_energy{};
};

} // namespace

void MovementPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    std::unordered_map<Location, std::vector<Entity::id_type>> destinations;
    id_map<Entity, std::reference_wrapper<const MoveCommand>> causes;

    auto commands = flatten_player_commands_if<MoveCommand>(result.validated_commands->moves, [](const MoveCommand &command) {
        return command.direction != Direction::Still;
    });

    auto plans = parallel_plan<MovePlan>(
        context,
        commands.size(),
        [&](std::size_t index, std::vector<std::optional<MovePlan>> &plans) {
            const auto [player_id, command] = commands[index];
            auto &player = store.get_player(player_id);
            MovePlan plan{};
            plan.player_id = player_id;
            plan.command = command;
            plan.entity_id = command->entity;

            if (!player.has_entity(command->entity)) {
                plan.entity_missing = true;
                plans[index] = plan;
                return;
            }

            auto location = player.get_entity_location(command->entity);
            auto &source = map.at(location);
            auto &entity = store.get_entity(command->entity);
            const auto cost_ratio = entity.is_inspired ? context.config().ruleset.inspiration.move_cost_ratio
                                                       : context.config().ruleset.economy.move_cost_ratio;
            energy_type required = source.energy / cost_ratio;
            plan.from = location;
            plan.required = required;
            plan.current_energy = entity.energy;
            if (entity.energy < required) {
                plan.insufficient_energy = true;
                plans[index] = plan;
                return;
            }

            map.move_location(location, command->direction);
            plan.to = location;
            plans[index] = plan;
        });

    for_each_planned(plans, [&](const auto &plan) {
        if (plan.entity_missing) {
            append_error(result, std::make_unique<EntityNotFoundError<MoveCommand>>(plan.player_id, *plan.command));
            return;
        }
        if (plan.insufficient_energy) {
            append_error(result, std::make_unique<InsufficientEnergyError<MoveCommand>>(plan.player_id,
                                                                                        *plan.command,
                                                                                        plan.current_energy,
                                                                                        plan.required,
                                                                                        !context.config().match.strict_errors));
            return;
        }

        auto &entity = store.get_entity(plan.entity_id);
        auto &source = map.at(plan.from);
        causes.emplace(plan.entity_id, *plan.command);
        entity.energy -= plan.required;
        source.entity = Entity::None;
        destinations[plan.to].emplace_back(plan.entity_id);
        store.get_player(entity.owner).remove_entity(plan.entity_id);
        result.changed_entities.emplace(plan.entity_id);
        result.changed_cells.emplace(plan.from);
        result.changed_cells.emplace(plan.to);
        context.event_sink().emit(events::ShipMovedEvent{plan.entity_id, plan.from, plan.to});
    });

    for (auto &[destination, _] : destinations) {
        auto &cell = map.at(destination);
        if (cell.entity != Entity::None) {
            destinations[destination].emplace_back(cell.entity);
            store.get_player(store.get_entity(cell.entity).owner).remove_entity(cell.entity);
            cell.entity = Entity::None;
        }
    }

    static constexpr auto MAX_ENTITIES_PER_CELL = 1;
    for (auto &[destination, entities] : destinations) {
        auto &cell = map.at(destination);
        if (entities.size() > MAX_ENTITIES_PER_CELL) {
            const int collision_damage = context.config().ruleset.combat.collision_hp_damage;
            std::vector<Entity::id_type> collision_ids;
            id_map<Player, std::vector<Entity::id_type>> self_collisions;
            id_map<Player, std::deque<std::reference_wrapper<const MoveCommand>>> self_collision_commands;
            for (auto &entity_id : entities) {
                auto &entity = store.get_entity(entity_id);
                collision_ids.push_back(entity_id);
                self_collisions[entity.owner].emplace_back(entity_id);
                if (auto cause = causes.find(entity_id); cause != causes.end()) {
                    self_collision_commands[entity.owner].emplace_back(cause->second);
                }
                entity.hp -= collision_damage;
                result.changed_entities.emplace(entity_id);
            }
            for (const auto &[player_id, self_collision_entities] : self_collisions) {
                if (self_collision_entities.size() > MAX_ENTITIES_PER_CELL) {
                    auto &commands = self_collision_commands[player_id];
                    const MoveCommand &first = commands.front();
                    commands.pop_front();
                    const ErrorContext error_context{commands.begin(), commands.end()};
                    append_error(result, std::make_unique<SelfCollisionError<MoveCommand>>(player_id, first, error_context, destination,
                                                                                            self_collision_entities,
                                                                                            !context.config().match.strict_errors));
                }
            }

            context.event_sink().emit(events::CollisionResolvedEvent{destination, collision_ids});

            for (const auto &entity_id : collision_ids) {
                auto &entity = store.get_entity(entity_id);
                if (entity.hp <= 0) {
                    dump_energy(store, entity, destination, cell, entity.energy);
                    store.delete_entity(entity_id);
                } else {
                    store.get_player(entity.owner).add_entity(entity_id, destination);
                }
            }

            cell.entity = Entity::None;
            for (const auto &entity_id : collision_ids) {
                bool alive = false;
                for (auto &[eid, _ent] : store.all_entities()) {
                    if (eid == entity_id) {
                        alive = true;
                        break;
                    }
                }
                if (alive) {
                    cell.entity = entity_id;
                    break;
                }
            }
            result.changed_cells.emplace(destination);
        } else {
            auto &entity_id = entities.front();
            cell.entity = entity_id;
            store.get_player(store.get_entity(entity_id).owner).add_entity(entity_id, destination);
            result.changed_entities.emplace(entity_id);
            result.changed_cells.emplace(destination);
        }
    }
}

} // namespace hlt::rules::phases
