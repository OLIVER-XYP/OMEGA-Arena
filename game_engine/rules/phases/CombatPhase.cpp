#include "CombatPhase.hpp"

#include <optional>
#include <unordered_set>
#include <vector>

#include "PhaseHelpers.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

namespace {

struct PendingDeath {
    Entity::id_type id{Entity::None};
    Player::id_type owner{Player::None};
    Player::id_type killer{Player::None};
    Location loc{0, 0};
    energy_type energy{};           // energy dumped on cell (post-steal)
    energy_type pre_steal_energy{}; // energy used for kill credit (pre-steal)
};

struct StructureAttackPlan {
    Player::id_type attacker_owner{Player::None};
    Entity::id_type attacker_id{Entity::None};
    Location attacker_loc{0, 0};
    Player::id_type target_owner{Player::None};
    Location target_loc{0, 0};
    bool valid{};
};

struct ShipAttackPlan {
    Player::id_type attacker_owner{Player::None};
    Entity::id_type attacker_id{Entity::None};
    Location attacker_loc{0, 0};
    Entity::id_type target_id{Entity::None};
    Player::id_type target_owner{Player::None};
    Location target_loc{0, 0};
    bool valid{};
};

struct AttackPlan {
    Player::id_type player_id{Player::None};
    const AttackCommand *command = nullptr;
    bool is_structure_target{};
    std::optional<StructureAttackPlan> structure_plan;
    std::optional<ShipAttackPlan> ship_plan;
    bool invalid = false;
};

} // namespace

void CombatPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    const auto &combat = context.config().ruleset.combat;

    auto attack_commands = flatten_player_commands<AttackCommand>(result.validated_commands->attacks);
    auto plans = parallel_plan<AttackPlan>(
        context,
        attack_commands.size(),
        [&](std::size_t index, std::vector<std::optional<AttackPlan>> &plans) {
            const auto [player_id, command] = attack_commands[index];
            const auto &player = store.get_player(player_id);
            if (!player.has_entity(command->entity)) {
                return;
            }

            const auto &attacker = store.get_entity(command->entity);
            const auto attacker_loc = player.get_entity_location(command->entity);
            AttackPlan plan{};
            plan.player_id = player_id;
            plan.command = command;
            plan.is_structure_target = command->is_structure_target;

            if (command->is_structure_target) {
                if (command->target_structure_owner == Player::None || command->target_structure_owner == player_id) {
                    plan.invalid = true;
                    plans[index] = plan;
                    return;
                }

                const auto &target_cell = map.at(command->target_structure_location);
                if (target_cell.owner != command->target_structure_owner || map.distance(attacker_loc, command->target_structure_location) != 1) {
                    plan.invalid = true;
                    plans[index] = plan;
                    return;
                }

                plan.structure_plan = StructureAttackPlan{player_id,
                                                          command->entity,
                                                          attacker_loc,
                                                          command->target_structure_owner,
                                                          command->target_structure_location,
                                                          true};
                plans[index] = plan;
                return;
            }

            const Entity *target_ptr = nullptr;
            for (auto &[eid, ent] : store.all_entities()) {
                if (eid == command->target && ent.owner != player_id) {
                    target_ptr = &ent;
                    break;
                }
            }
            if (target_ptr == nullptr) {
                plan.invalid = true;
                plans[index] = plan;
                return;
            }

            const auto &target_player = store.get_player(target_ptr->owner);
            const auto target_loc = target_player.get_entity_location(command->target);
            if (map.distance(attacker_loc, target_loc) > combat.attack_range) {
                plan.invalid = true;
                plans[index] = plan;
                return;
            }

            plan.ship_plan = ShipAttackPlan{player_id,
                                            command->entity,
                                            attacker_loc,
                                            command->target,
                                            target_ptr->owner,
                                            target_loc,
                                            true};
            plans[index] = plan;
        });

    std::vector<PendingDeath> to_delete;
    for_each_planned(plans, [&](const auto &plan) {
        if (plan.invalid) {
            append_error(result, std::make_unique<EntityNotFoundError<AttackCommand>>(plan.player_id, *plan.command, !context.config().match.strict_errors));
            return;
        }

        auto &player = store.get_player(plan.player_id);
        if (!player.has_entity(plan.command->entity)) {
            return;
        }
        auto &attacker = store.get_entity(plan.command->entity);

        if (plan.is_structure_target && plan.structure_plan.has_value()) {
            const auto &structure = *plan.structure_plan;
            energy_type penalty = attacker.energy / 2;
            attacker.energy -= penalty;
            auto &target_player = store.get_player(structure.target_owner);
            energy_type *pool = nullptr;
            bool *destroyed_flag = nullptr;
            if (structure.target_loc == target_player.factory) {
                pool = &target_player.factory_halite;
                destroyed_flag = &target_player.factory_destroyed;
            } else {
                for (auto &dropoff : target_player.dropoffs) {
                    if (dropoff.location == structure.target_loc) {
                        pool = &dropoff.halite_pool;
                        destroyed_flag = &dropoff.destroyed;
                        break;
                    }
                }
            }
            if (pool == nullptr || destroyed_flag == nullptr) {
                append_error(result, std::make_unique<EntityNotFoundError<AttackCommand>>(plan.player_id, *plan.command, !context.config().match.strict_errors));
                return;
            }
            if (!(*destroyed_flag)) {
                if (penalty >= *pool) {
                    *pool = 0;
                    *destroyed_flag = true;
                } else {
                    *pool -= penalty;
                }
            }
            result.changed_entities.emplace(plan.command->entity);
            result.changed_cells.emplace(structure.target_loc);
            context.event_sink().emit(events::CombatResolvedEvent{structure.attacker_loc,
                                                                  structure.attacker_id,
                                                                  structure.target_loc,
                                                                  Entity::None,
                                                                  true});
            return;
        }

        if (!plan.ship_plan.has_value()) {
            return;
        }

        const auto &ship = *plan.ship_plan;
        auto &current_target = store.get_entity(ship.target_id);
        auto &target_player = store.get_player(current_target.owner);
        const auto current_target_loc = target_player.get_entity_location(ship.target_id);
        if (current_target.owner == plan.player_id || !(current_target_loc == ship.target_loc) || map.distance(ship.attacker_loc, current_target_loc) > combat.attack_range) {
            append_error(result, std::make_unique<EntityNotFoundError<AttackCommand>>(plan.player_id, *plan.command, !context.config().match.strict_errors));
            return;
        }
        if (current_target.is_defending) {
            // Defender reflects retaliation damage back to the attacker, but the
            // attack still lands (damage + theft).  This makes defend a soft
            // counter: attacking a turtled ship is costly but not impossible.
            // The structural rock-paper-scissors cycle depends on this — without
            // it aggro can never beat control and the metagame collapses to a
            // single transitive ordering.
            if (combat.defend_retaliation_damage > 0) {
                attacker.hp -= combat.defend_retaliation_damage;
                result.changed_entities.emplace(ship.attacker_id);
                if (player.has_entity(ship.attacker_id)) {
                    auto &att = store.get_entity(ship.attacker_id);
                    if (att.hp <= 0) {
                        to_delete.push_back({ship.attacker_id, plan.player_id, current_target.owner, ship.attacker_loc, att.energy, att.energy});
                    }
                }
            }
            // Fall through — attack still deals damage and steals cargo.
        }

        auto target_energy_pre_steal = current_target.energy;
        current_target.hp -= combat.attack_hp_damage;
        energy_type stolen = static_cast<energy_type>(static_cast<double>(current_target.energy) * combat.attack_halite_steal_ratio);
        current_target.energy -= stolen;
        attacker.energy = std::min(attacker.energy + stolen, context.config().ruleset.economy.max_energy);
        attacker.enemy_hp_dealt += combat.attack_hp_damage;
        attacker.enemy_halite_taken += stolen;
        if (combat.enable_attacker_self_damage) {
            attacker.hp -= combat.attack_hp_self_damage;
        }

        context.event_sink().emit(events::CombatResolvedEvent{ship.attacker_loc,
                                                              ship.attacker_id,
                                                              current_target_loc,
                                                              ship.target_id,
                                                              true});
        result.changed_entities.emplace(ship.attacker_id);
        result.changed_entities.emplace(ship.target_id);

        if (current_target.hp <= 0) {
            to_delete.push_back({ship.target_id, current_target.owner, plan.player_id,
                                 current_target_loc, current_target.energy, target_energy_pre_steal});
        }
        if (player.has_entity(ship.attacker_id)) {
            auto &att = store.get_entity(ship.attacker_id);
            if (att.hp <= 0) {
                to_delete.push_back({ship.attacker_id, plan.player_id, Player::None,
                                     ship.attacker_loc, att.energy, att.energy});
            }
        }
    });

    std::unordered_set<Entity::id_type> deleted;
    for (auto &[eid, owner_id, killer_id, loc, energy, pre_steal_energy] : to_delete) {
        if (deleted.count(eid)) {
            continue;
        }
        deleted.insert(eid);
        auto &cell = map.at(loc);
        if (combat.kill_credit_to_attacker && killer_id != Player::None && killer_id != owner_id) {
            energy_type base = pre_steal_energy > 0 ? pre_steal_energy : energy;
            energy_type credited = static_cast<energy_type>(base * (1.0 + combat.kill_halite_bonus_ratio));
            auto &killer_player = store.get_player(killer_id);
            killer_player.energy += credited;
            killer_player.total_energy_deposited += credited;
        }
        auto &entity = store.get_entity(eid);
        dump_energy(store, entity, loc, cell, entity.energy);
        store.get_player(owner_id).remove_entity(eid);
        store.delete_entity(eid);
        if (cell.entity == eid) {
            cell.entity = Entity::None;
        }
        result.changed_cells.emplace(loc);
    }
}

} // namespace hlt::rules::phases


