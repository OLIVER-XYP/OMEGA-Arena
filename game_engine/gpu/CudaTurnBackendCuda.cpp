#include "CudaTurnBackend.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CommandFrame.hpp"
#include "CudaCapture.hpp"
#include "CudaCombat.hpp"
#include "CudaConstruction.hpp"
#include "CudaDevice.hpp"
#include "CudaDefend.hpp"
#include "CudaDump.hpp"
#include "CudaHaliteRebalance.hpp"
#include "CudaInspiration.hpp"
#include "CudaMining.hpp"
#include "CudaMovement.hpp"
#include "CudaOverShipTax.hpp"
#include "CudaPlunder.hpp"
#include "CudaRegen.hpp"
#include "CudaSpawn.hpp"
#include "CudaValidation.hpp"
#include "RuleContext.hpp"
#include "StateFrame.hpp"
#include "TurnFrame.hpp"
#include "phases/PhaseHelpers.hpp"

namespace hlt::gpu {

namespace {

bool location_before(const Location &a, const Location &b) {
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.x < b.x;
}

std::string unavailable_reason() {
    const auto info = query_cuda_device();
    if (!info.runtime_available) {
        return std::string("CUDA runtime unavailable: ") + info.error_message;
    }
    return std::string("CUDA device '") + info.device_name + "' is available.";
}

template <typename Fn>
void run_profiled(TurnExecutionProfile *profile, const std::string &name, Fn &&fn) {
    const auto phase_start = std::chrono::steady_clock::now();
    fn();
    if (profile != nullptr) {
        const auto phase_end = std::chrono::steady_clock::now();
        profile->phase_timings.push_back(PhaseTimingEntry{
            name,
            std::chrono::duration_cast<std::chrono::nanoseconds>(phase_end - phase_start).count()});
    }
}

void run_cuda_inspiration_on_state(GameState &state, const GameConfig &config) {
    auto frame = capture_state_frame(state);
    ::hlt::gpu::run_cuda_inspiration(frame, config.ruleset.inspiration);
    apply_state_frame(state, frame);
}

void run_cuda_defend_on_state(GameState &state, StepResult &result, const GameConfig &config) {
    if (!result.validated_commands.has_value()) {
        return;
    }

    bool has_work = false;
    for (const auto &[player_id, defends] : result.validated_commands->defends) {
        (void)player_id;
        if (!defends.empty()) {
            has_work = true;
            break;
        }
    }
    if (!has_work) {
        for (const auto &[entity_id, entity] : state.store_ref().entities_ref()) {
            (void)entity_id;
            if (entity.is_defending || entity.protection_turns > 0) {
                has_work = true;
                break;
            }
        }
    }
    if (!has_work) {
        return;
    }

    auto frame = capture_state_frame(state);
    const auto commands = flatten_command_batch(*result.validated_commands);
    ::hlt::gpu::run_cuda_defend(frame, commands, config);
    apply_state_frame(state, frame);

    auto &store = state.store_ref();
    for (const auto &command : commands.commands) {
        if (command.type != FlatCommandType::Defend) {
            continue;
        }
        const auto entity_it = store.entities_ref().find(command.entity);
        if (entity_it != store.entities_ref().end() && entity_it->second.owner == command.player) {
            result.changed_entities.emplace(command.entity);
        }
    }
}

void run_cuda_validation_on_state(GameState &state,
                                  ActionBatch &actions,
                                  StepResult &result,
                                  const GameConfig &config) {
    const auto refs = flatten_validation_actions(actions, state.store_ref());
    const auto decisions = ::hlt::gpu::run_cuda_validation_decisions(capture_state_frame(state),
                                                                     command_frame_from_validation_refs(refs),
                                                                     config);
    apply_validation_decisions(state, actions, result, refs, decisions);
}

CommandFrame flatten_construct_commands(const std::vector<std::pair<Player::id_type, const ConstructCommand *>> &commands) {
    CommandFrame frame;
    frame.commands.reserve(commands.size());
    for (const auto &[player_id, command] : commands) {
        frame.commands.push_back(FlatCommand{player_id,
                                             command->entity,
                                             Entity::None,
                                             Player::None,
                                             Location{0, 0},
                                             Direction::Still,
                                             FlatCommandType::Construct});
    }
    return frame;
}

CommandFrame flatten_spawn_commands(const std::vector<std::pair<Player::id_type, const SpawnCommand *>> &commands) {
    CommandFrame frame;
    frame.commands.reserve(commands.size());
    for (const auto &[player_id, command] : commands) {
        (void)command;
        frame.commands.push_back(FlatCommand{player_id,
                                             Entity::None,
                                             Entity::None,
                                             Player::None,
                                             Location{0, 0},
                                             Direction::Still,
                                             FlatCommandType::Spawn});
    }
    return frame;
}

CommandFrame flatten_move_commands(const std::vector<std::pair<Player::id_type, const MoveCommand *>> &commands) {
    CommandFrame frame;
    frame.commands.reserve(commands.size());
    for (const auto &[player_id, command] : commands) {
        frame.commands.push_back(FlatCommand{player_id,
                                             command->entity,
                                             Entity::None,
                                             Player::None,
                                             Location{0, 0},
                                             command->direction,
                                             FlatCommandType::Move});
    }
    return frame;
}

CommandFrame flatten_attack_commands(const std::vector<std::pair<Player::id_type, const AttackCommand *>> &commands) {
    CommandFrame frame;
    frame.commands.reserve(commands.size());
    for (const auto &[player_id, command] : commands) {
        const auto type = command->is_structure_target ? FlatCommandType::AttackStructure : FlatCommandType::AttackShip;
        frame.commands.push_back(FlatCommand{player_id,
                                             command->entity,
                                             command->target,
                                             command->target_structure_owner,
                                             command->target_structure_location,
                                             Direction::Still,
                                             type});
    }
    return frame;
}

struct CudaPendingDeath {
    Entity::id_type id{Entity::None};
    Player::id_type owner{Player::None};
    Player::id_type killer{Player::None};
    Location loc{0, 0};
    energy_type energy{};
    energy_type pre_steal_energy{};
};

void run_cuda_construction_on_state(GameState &state,
                                    StepResult &result,
                                    events::EventSink &event_sink,
                                    const GameConfig &config) {
    if (!result.validated_commands.has_value()) {
        return;
    }

    const auto commands =
        rules::phases::flatten_player_commands<ConstructCommand>(result.validated_commands->constructs);
    if (commands.empty()) {
        return;
    }

    auto &store = state.store_ref();
    auto &map = state.map_ref();
    const auto command_frame = flatten_construct_commands(commands);
    const auto decisions = ::hlt::gpu::run_cuda_construction_decisions(capture_state_frame(state), command_frame, config);

    for (std::size_t index = 0; index < decisions.size() && index < commands.size(); ++index) {
        const auto [player_id, command] = commands[index];
        const auto &decision = decisions[index];
        if (!decision.command) {
            continue;
        }
        if (decision.entity_missing) {
            rules::phases::append_error(result, std::make_unique<EntityNotFoundError<ConstructCommand>>(player_id, *command));
            continue;
        }
        if (decision.cell_owned) {
            rules::phases::append_error(result,
                                        std::make_unique<CellOwnedError<ConstructCommand>>(player_id,
                                                                                          *command,
                                                                                          decision.location,
                                                                                          decision.cell_owner));
            continue;
        }

        auto &player = store.get_player(player_id);
        if (!player.has_entity(decision.entity)) {
            continue;
        }

        const auto &entity = store.get_entity(decision.entity);
        auto &cell = map.at(decision.location);

        cell.owner = player_id;
        player.dropoffs.emplace_back(store.new_dropoff(decision.location));
        auto &created_dropoff = player.dropoffs.back();
        created_dropoff.halite_pool = config.ruleset.economy.initial_dropoff_halite;
        created_dropoff.destroyed = false;
        store.map_total_energy -= cell.energy;

        const auto credit = cell.energy + entity.energy;
        {
            auto &mut_entity = store.get_entity(decision.entity);
            mut_entity.lifetime_deposited += mut_entity.energy;
        }
        cell.energy = 0;
        cell.entity = Entity::None;
        result.changed_cells.emplace(decision.location);
        rules::phases::dump_energy(store, decision.location, cell, credit);
        player.energy -= decision.cost;
        player.remove_entity(decision.entity);
        store.delete_entity(decision.entity);
        event_sink.emit(events::ConstructionResolvedEvent{decision.location, player_id, command->entity});
    }
}

void run_cuda_combat_on_state(GameState &state,
                              StepResult &result,
                              events::EventSink &event_sink,
                              const GameConfig &config) {
    if (!result.validated_commands.has_value()) {
        return;
    }

    const auto attack_commands =
        rules::phases::flatten_player_commands<AttackCommand>(result.validated_commands->attacks);
    if (attack_commands.empty()) {
        return;
    }

    auto &store = state.store_ref();
    auto &map = state.map_ref();
    const auto &combat = config.ruleset.combat;
    const auto decisions = ::hlt::gpu::run_cuda_combat_decisions(capture_state_frame(state),
                                                                 flatten_attack_commands(attack_commands),
                                                                 config);

    std::vector<CudaPendingDeath> to_delete;
    for (std::size_t index = 0; index < decisions.size() && index < attack_commands.size(); ++index) {
        const auto [player_id, command] = attack_commands[index];
        const auto &decision = decisions[index];
        if (!decision.command) {
            continue;
        }
        if (decision.invalid) {
            rules::phases::append_error(result,
                                        std::make_unique<EntityNotFoundError<AttackCommand>>(player_id,
                                                                                             *command,
                                                                                             !config.match.strict_errors));
            continue;
        }

        auto &player = store.get_player(player_id);
        if (!player.has_entity(command->entity)) {
            continue;
        }
        auto &attacker = store.get_entity(command->entity);

        if (decision.structure_target && decision.valid_structure) {
            energy_type penalty = attacker.energy / 2;
            attacker.energy -= penalty;
            auto &target_player = store.get_player(decision.target_owner);
            energy_type *pool = nullptr;
            bool *destroyed_flag = nullptr;
            if (decision.target_location == target_player.factory) {
                pool = &target_player.factory_halite;
                destroyed_flag = &target_player.factory_destroyed;
            } else {
                for (auto &dropoff : target_player.dropoffs) {
                    if (dropoff.location == decision.target_location) {
                        pool = &dropoff.halite_pool;
                        destroyed_flag = &dropoff.destroyed;
                        break;
                    }
                }
            }
            if (pool == nullptr || destroyed_flag == nullptr) {
                rules::phases::append_error(result,
                                            std::make_unique<EntityNotFoundError<AttackCommand>>(player_id,
                                                                                                 *command,
                                                                                                 !config.match.strict_errors));
                continue;
            }
            if (!(*destroyed_flag)) {
                if (penalty >= *pool) {
                    *pool = 0;
                    *destroyed_flag = true;
                } else {
                    *pool -= penalty;
                }
            }
            result.changed_entities.emplace(command->entity);
            result.changed_cells.emplace(decision.target_location);
            event_sink.emit(events::CombatResolvedEvent{decision.attacker_location,
                                                        decision.attacker,
                                                        decision.target_location,
                                                        Entity::None,
                                                        true});
            continue;
        }

        if (!decision.valid_ship) {
            continue;
        }

        auto &current_target = store.get_entity(decision.target);
        auto &target_player = store.get_player(current_target.owner);
        const auto current_target_loc = target_player.get_entity_location(decision.target);
        if (current_target.owner == player_id ||
            !(current_target_loc == decision.target_location) ||
            map.distance(decision.attacker_location, current_target_loc) > combat.attack_range) {
            rules::phases::append_error(result,
                                        std::make_unique<EntityNotFoundError<AttackCommand>>(player_id,
                                                                                             *command,
                                                                                             !config.match.strict_errors));
            continue;
        }
        if (current_target.is_defending) {
            if (combat.defend_retaliation_damage > 0) {
                attacker.hp -= combat.defend_retaliation_damage;
                result.changed_entities.emplace(decision.attacker);
                if (player.has_entity(decision.attacker)) {
                    auto &att = store.get_entity(decision.attacker);
                    if (att.hp <= 0) {
                        to_delete.push_back({decision.attacker,
                                             player_id,
                                             current_target.owner,
                                             decision.attacker_location,
                                             att.energy,
                                             att.energy});
                    }
                }
            }
        }

        auto target_energy_pre_steal = current_target.energy;
        current_target.hp -= combat.attack_hp_damage;
        energy_type stolen = static_cast<energy_type>(static_cast<double>(current_target.energy) * combat.attack_halite_steal_ratio);
        current_target.energy -= stolen;
        attacker.energy = std::min(attacker.energy + stolen, config.ruleset.economy.max_energy);
        attacker.enemy_hp_dealt += combat.attack_hp_damage;
        attacker.enemy_halite_taken += stolen;
        if (combat.enable_attacker_self_damage) {
            attacker.hp -= combat.attack_hp_self_damage;
        }

        event_sink.emit(events::CombatResolvedEvent{decision.attacker_location,
                                                    decision.attacker,
                                                    current_target_loc,
                                                    decision.target,
                                                    true});
        result.changed_entities.emplace(decision.attacker);
        result.changed_entities.emplace(decision.target);

        if (current_target.hp <= 0) {
            to_delete.push_back({decision.target,
                                 current_target.owner,
                                 player_id,
                                 current_target_loc,
                                 current_target.energy,
                                 target_energy_pre_steal});
        }
        if (player.has_entity(decision.attacker)) {
            auto &att = store.get_entity(decision.attacker);
            if (att.hp <= 0) {
                to_delete.push_back({decision.attacker,
                                     player_id,
                                     Player::None,
                                     decision.attacker_location,
                                     att.energy,
                                     att.energy});
            }
        }
    }

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
        rules::phases::dump_energy(store, entity, loc, cell, entity.energy);
        store.get_player(owner_id).remove_entity(eid);
        store.delete_entity(eid);
        if (cell.entity == eid) {
            cell.entity = Entity::None;
        }
        result.changed_cells.emplace(loc);
    }
}

void run_cuda_movement_on_state(GameState &state,
                                StepResult &result,
                                events::EventSink &event_sink,
                                const GameConfig &config) {
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto commands = rules::phases::flatten_player_commands_if<MoveCommand>(
        result.validated_commands->moves,
        [](const MoveCommand &command) {
            return command.direction != Direction::Still;
        });
    if (commands.empty()) {
        return;
    }

    auto &store = state.store_ref();
    auto &map = state.map_ref();
    std::unordered_map<Location, std::vector<Entity::id_type>> destinations;
    id_map<Entity, std::reference_wrapper<const MoveCommand>> causes;

    const auto decisions = ::hlt::gpu::run_cuda_movement_decisions(capture_state_frame(state), flatten_move_commands(commands), config);
    for (std::size_t index = 0; index < decisions.size() && index < commands.size(); ++index) {
        const auto [player_id, command] = commands[index];
        const auto &decision = decisions[index];
        if (!decision.command) {
            continue;
        }
        if (decision.entity_missing) {
            rules::phases::append_error(result, std::make_unique<EntityNotFoundError<MoveCommand>>(player_id, *command));
            continue;
        }
        if (decision.insufficient_energy) {
            rules::phases::append_error(result,
                                        std::make_unique<InsufficientEnergyError<MoveCommand>>(player_id,
                                                                                               *command,
                                                                                               decision.current_energy,
                                                                                               decision.required,
                                                                                               !config.match.strict_errors));
            continue;
        }

        auto &entity = store.get_entity(decision.entity);
        auto &source = map.at(decision.from);
        causes.emplace(decision.entity, *command);
        entity.energy -= decision.required;
        source.entity = Entity::None;
        destinations[decision.to].emplace_back(decision.entity);
        store.get_player(entity.owner).remove_entity(decision.entity);
        result.changed_entities.emplace(decision.entity);
        result.changed_cells.emplace(decision.from);
        result.changed_cells.emplace(decision.to);
        event_sink.emit(events::ShipMovedEvent{decision.entity, decision.from, decision.to});
    }

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
            const int collision_damage = config.ruleset.combat.collision_hp_damage;
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
                    auto &player_commands = self_collision_commands[player_id];
                    const MoveCommand &first = player_commands.front();
                    player_commands.pop_front();
                    const ErrorContext error_context{player_commands.begin(), player_commands.end()};
                    rules::phases::append_error(result,
                                                std::make_unique<SelfCollisionError<MoveCommand>>(player_id,
                                                                                                  first,
                                                                                                  error_context,
                                                                                                  destination,
                                                                                                  self_collision_entities,
                                                                                                  !config.match.strict_errors));
                }
            }

            event_sink.emit(events::CollisionResolvedEvent{destination, collision_ids});

            for (const auto &entity_id : collision_ids) {
                auto &entity = store.get_entity(entity_id);
                if (entity.hp <= 0) {
                    rules::phases::dump_energy(store, entity, destination, cell, entity.energy);
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

void resolve_spawn_collision(GameState &state,
                             StepResult &result,
                             events::EventSink &event_sink,
                             const GameConfig &config,
                             Player::id_type player_id,
                             const SpawnCommand &command,
                             Entity::id_type existing_entity_id,
                             Entity::id_type spawned_entity_id,
                             Location factory) {
    auto &store = state.store_ref();
    auto &map = state.map_ref();
    auto &cell = map.at(factory);
    auto &existing_entity = store.get_entity(existing_entity_id);
    auto &existing_player = store.get_player(existing_entity.owner);
    auto &owner = store.get_player(cell.owner);
    if (existing_entity.owner == cell.owner) {
        rules::phases::append_error(result,
                                    std::make_unique<SelfCollisionError<SpawnCommand>>(player_id,
                                                                                       command,
                                                                                       ErrorContext(),
                                                                                       factory,
                                                                                       std::vector<Entity::id_type>{existing_entity_id, spawned_entity_id},
                                                                                       !config.match.strict_errors));
    }
    event_sink.emit(events::CollisionResolvedEvent{owner.factory, std::vector<Entity::id_type>{existing_entity_id, spawned_entity_id}});
    rules::phases::dump_energy(store, existing_entity, owner.factory, cell, existing_entity.energy);
    existing_player.remove_entity(existing_entity_id);
    store.delete_entity(existing_entity_id);
    store.get_player(player_id).remove_entity(spawned_entity_id);
    store.delete_entity(spawned_entity_id);
    cell.entity = Entity::None;
}

bool emergency_spawn_possible(const GameState &state, const GameConfig &config) {
    const auto &economy = config.ruleset.economy;
    return economy.emergency_spawn_enabled &&
           economy.emergency_spawn_period > 0 &&
           economy.emergency_spawn_count > 0 &&
           state.turn.number > 0 &&
           state.turn.number % economy.emergency_spawn_period == 0;
}

void run_cuda_spawn_on_state(GameState &state,
                             StepResult &result,
                             events::EventSink &event_sink,
                             const GameConfig &config) {
    if (!result.validated_commands.has_value()) {
        return;
    }

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
            rules::phases::append_error(result, std::make_unique<ExcessiveSpawnsError>(player_id, illegal, error_context));
            continue;
        }
        for (const SpawnCommand &spawn : spawns) {
            commands.emplace_back(player_id, &spawn);
        }
    }

    const bool can_emergency_spawn = emergency_spawn_possible(state, config);
    if (commands.empty() && !can_emergency_spawn) {
        return;
    }

    ::hlt::gpu::SpawnFrameDecisions decisions;
    if (!commands.empty()) {
        decisions = ::hlt::gpu::run_cuda_spawn_decisions(capture_state_frame(state), flatten_spawn_commands(commands), config);
    }
    auto &store = state.store_ref();
    auto &map = state.map_ref();

    for (std::size_t index = 0; index < decisions.spawns.size() && index < commands.size(); ++index) {
        const auto [player_id, command] = commands[index];
        const auto &decision = decisions.spawns[index];
        if (!decision.command) {
            continue;
        }

        auto &player = store.get_player(player_id);
        player.energy -= decision.cost;
        auto &cell = map.at(decision.factory);
        auto &entity = store.new_entity(0, player.id);
        entity.spawn_turn = state.turn.number;
        const auto spawned_id = entity.id;
        player.add_entity(spawned_id, decision.factory);
        result.changed_entities.emplace(spawned_id);
        event_sink.emit(events::SpawnedEvent{player.id, spawned_id, decision.factory, 0});
        if (!decision.factory_occupied || cell.entity == Entity::None) {
            cell.entity = spawned_id;
        } else {
            resolve_spawn_collision(state,
                                    result,
                                    event_sink,
                                    config,
                                    player_id,
                                    *command,
                                    decision.existing_entity,
                                    spawned_id,
                                    decision.factory);
        }
        result.changed_cells.emplace(decision.factory);
    }

    if (!can_emergency_spawn) {
        return;
    }

    if (!commands.empty()) {
        CommandFrame no_spawn_commands;
        decisions = ::hlt::gpu::run_cuda_spawn_decisions(capture_state_frame(state), no_spawn_commands, config);
    } else {
        CommandFrame no_spawn_commands;
        decisions = ::hlt::gpu::run_cuda_spawn_decisions(capture_state_frame(state), no_spawn_commands, config);
    }
    const auto &economy = config.ruleset.economy;
    for (const auto &decision : decisions.emergency_spawns) {
        if (decision.count == 0) {
            continue;
        }
        auto player_it = store.players_ref().find(decision.player);
        if (player_it == store.players_ref().end()) {
            continue;
        }
        auto &player = player_it->second;
        const auto spawn_count = std::min<unsigned long>(decision.count, economy.emergency_spawn_count);
        for (unsigned long index = 0; index < spawn_count && index < 5; ++index) {
            const auto location = decision.locations[index];
            auto &cell = map.at(location);
            if (cell.entity != Entity::None) {
                continue;
            }
            auto &entity = store.new_entity(0, player.id);
            entity.spawn_turn = state.turn.number;
            entity.protection_turns = static_cast<int>(economy.emergency_protection_turns);
            entity.is_defending = entity.protection_turns > 0;
            player.add_entity(entity.id, location);
            cell.entity = entity.id;
            result.changed_entities.emplace(entity.id);
            result.changed_cells.emplace(location);
            event_sink.emit(events::SpawnedEvent{player.id, entity.id, location, 0});
        }
        if (economy.emergency_halite_bonus > 0) {
            player.energy += economy.emergency_halite_bonus;
        }
    }
}

bool should_mine_entity(const EntityFrameEntry &entity,
                        const StepResult &result,
                        const GameConfig &config) {
    if (!entity.alive || entity.energy >= config.ruleset.economy.max_energy) {
        return false;
    }

    const bool defend_mines = config.ruleset.combat.defend_allows_mining && entity.is_defending;
    const bool already_acted = result.changed_entities.find(entity.id) != result.changed_entities.end();
    return !already_acted || defend_mines;
}

void run_cuda_mining_on_state(GameState &state,
                              StepResult &result,
                              events::EventSink &event_sink,
                              const GameConfig &config) {
    auto before = capture_state_frame(state);
    auto after = before;
    const auto turn_frame = capture_turn_frame(result);
    std::vector<::hlt::gpu::MiningEffectFrameEntry> effects;
    ::hlt::gpu::run_cuda_mining(after, turn_frame, config, &effects);
    apply_state_frame(state, after);

    auto &store = state.store_ref();
    for (const auto &effect : effects) {
        if (effect.entity == Entity::None) {
            continue;
        }
        result.changed_cells.emplace(effect.location);
        store.changed_cells_ref().emplace(effect.location);
        event_sink.emit(events::MinedEvent{effect.entity,
                                           effect.owner,
                                           effect.location,
                                           effect.extracted,
                                           effect.gained,
                                           effect.was_captured});
    }
}

void emit_cuda_dump_events_from_frames(const StateFrame &before,
                                       const StateFrame &after,
                                       StepResult &result,
                                       events::EventSink &event_sink) {
    for (const auto &before_entity : before.entities) {
        if (!before_entity.alive) {
            continue;
        }
        const auto cell_index_value = cell_index(before, before_entity.location);
        if (before.cell_owner[cell_index_value] != before_entity.owner) {
            continue;
        }
        const auto after_slot_it = after.entity_slot_by_id.find(before_entity.id);
        if (after_slot_it == after.entity_slot_by_id.end()) {
            continue;
        }

        result.changed_cells.emplace(before_entity.location);
        result.changed_entities.emplace(before_entity.id);
        event_sink.emit(events::DepositedEvent{before_entity.owner, before_entity.location, before_entity.energy});
    }
}

void run_cuda_dump_on_state(GameState &state,
                            StepResult &result,
                            events::EventSink &event_sink,
                            const GameConfig &config) {
    const auto before = capture_state_frame(state);
    auto after = before;
    ::hlt::gpu::run_cuda_dump(after, config);
    apply_state_frame(state, after);
    emit_cuda_dump_events_from_frames(before, after, result, event_sink);
}

void apply_cuda_mining_effects_to_result(GameState &state,
                                         StepResult &result,
                                         events::EventSink &event_sink,
                                         const std::vector<::hlt::gpu::MiningEffectFrameEntry> &effects) {
    auto &store = state.store_ref();
    for (const auto &effect : effects) {
        if (effect.entity == Entity::None) {
            continue;
        }
        result.changed_cells.emplace(effect.location);
        store.changed_cells_ref().emplace(effect.location);
        event_sink.emit(events::MinedEvent{effect.entity,
                                           effect.owner,
                                           effect.location,
                                           effect.extracted,
                                           effect.gained,
                                           effect.was_captured});
    }
}

void record_cuda_regen_changes(const std::vector<energy_type> &before_energy,
                               const StateFrame &after,
                               StepResult &result) {
    for (dimension_type y = 0; y < after.height; ++y) {
        for (dimension_type x = 0; x < after.width; ++x) {
            const Location location{x, y};
            const auto index = cell_index(after, location);
            if (index < before_energy.size() && before_energy[index] != after.cell_energy[index]) {
                result.changed_cells.emplace(location);
            }
        }
    }
}

void run_cuda_regen_on_state(GameState &state, StepResult &result, const GameConfig &config) {
    auto before = capture_state_frame(state);
    auto after = before;
    ::hlt::gpu::run_cuda_regen(after, config);
    apply_state_frame(state, after);
    record_cuda_regen_changes(before.cell_energy, after, result);
}

void run_cuda_over_ship_tax_on_state(GameState &state, const GameConfig &config) {
    auto frame = capture_state_frame(state);
    ::hlt::gpu::run_cuda_over_ship_tax(frame, config);
    apply_state_frame(state, frame);
}

void run_cuda_plunder_on_state(GameState &state, const GameConfig &config) {
    auto frame = capture_state_frame(state);
    ::hlt::gpu::run_cuda_plunder(frame, config);
    apply_state_frame(state, frame);
}

void run_cuda_halite_rebalance_on_state(GameState &state, const GameConfig &config) {
    auto frame = capture_state_frame(state);
    ::hlt::gpu::run_cuda_halite_rebalance(frame, config);
    apply_state_frame(state, frame);
}

void apply_cuda_capture_decisions(GameState &state,
                                  StepResult &result,
                                  events::EventSink &event_sink,
                                  const GameConfig &config) {
    if (!config.ruleset.capture.enabled) {
        return;
    }

    auto &store = state.store_ref();
    auto &map = state.map_ref();
    const auto decisions = ::hlt::gpu::run_cuda_capture_decisions(capture_state_frame(state), config);

    std::unordered_map<Location, Player::id_type> entity_switches;
    for (const auto &decision : decisions) {
        if (decision.should_capture) {
            entity_switches[decision.location] = decision.new_owner;
        }
    }

    std::vector<std::pair<Location, Player::id_type>> ordered_switches;
    ordered_switches.reserve(entity_switches.size());
    for (const auto &[location, new_player_id] : entity_switches) {
        ordered_switches.emplace_back(location, new_player_id);
    }
    std::sort(ordered_switches.begin(), ordered_switches.end(), [](const auto &a, const auto &b) {
        return location_before(a.first, b.first);
    });

    for (const auto &[location, new_player_id] : ordered_switches) {
        auto &cell = map.at(location);
        if (cell.entity == Entity::None || store.entities_ref().find(cell.entity) == store.entities_ref().end()) {
            continue;
        }

        const auto entity = store.get_entity(cell.entity);
        const auto old_owner = entity.owner;
        const auto old_id = entity.id;
        const auto old_energy = entity.energy;

        store.get_player(old_owner).remove_entity(old_id);
        store.delete_entity(old_id);

        auto &new_entity = store.new_entity(old_energy, new_player_id);
        new_entity.spawn_turn = state.turn.number;
        cell.entity = new_entity.id;
        store.get_entity(new_entity.id).was_captured = true;
        store.get_player(new_player_id).add_entity(new_entity.id, location);
        result.changed_cells.emplace(location);
        store.changed_cells_ref().emplace(location);
        event_sink.emit(events::CapturedEvent{location, old_owner, old_id, new_player_id, new_entity.id});
    }
}

void run_cuda_post_spawn_frame_pipeline(GameState &state,
                                        StepResult &result,
                                        events::EventSink &event_sink,
                                        const GameConfig &config,
                                        TurnExecutionProfile *profile) {
    StateFrame frame;
    run_profiled(profile, "cuda:state-capture:post-spawn", [&] {
        frame = capture_state_frame(state);
    });
    run_profiled(profile, "cuda:mining+plunder", [&] {
        const auto turn_frame = capture_turn_frame(result);
        std::vector<::hlt::gpu::MiningEffectFrameEntry> effects;
        ::hlt::gpu::run_cuda_mining_and_plunder(frame, turn_frame, config, &effects);
        apply_cuda_mining_effects_to_result(state, result, event_sink, effects);
    });
    run_profiled(profile, "cuda:plunder-fused", [&] {
        (void)frame;
    });
    if (!config.runtime.skip_regen_phase) {
        run_profiled(profile, "cuda:regen", [&] {
            const auto before_regen_energy = frame.cell_energy;
            ::hlt::gpu::run_cuda_regen(frame, config);
            record_cuda_regen_changes(before_regen_energy, frame, result);
        });
    } else {
        run_profiled(profile, "cuda:regen-skipped", [&] {
            (void)state;
            (void)result;
        });
    }
    run_profiled(profile, "cuda:state-apply:post-spawn", [&] {
        apply_state_frame(state, frame);
    });
}

void run_cuda_end_economy_frame_pipeline(GameState &state,
                                         const GameConfig &config,
                                         TurnExecutionProfile *profile) {
    if (config.runtime.skip_end_economy_phases) {
        run_profiled(profile, "cuda:over-ship-tax-skipped", [&] {
            (void)state;
        });
        run_profiled(profile, "cuda:halite-rebalance-skipped", [&] {
            (void)state;
        });
        return;
    }

    StateFrame frame;
    run_profiled(profile, "cuda:state-capture:end-economy", [&] {
        frame = capture_state_frame(state);
    });
    run_profiled(profile, "cuda:over-ship-tax", [&] {
        ::hlt::gpu::run_cuda_over_ship_tax(frame, config);
    });
    run_profiled(profile, "cuda:halite-rebalance", [&] {
        ::hlt::gpu::run_cuda_halite_rebalance(frame, config);
    });
    run_profiled(profile, "cuda:state-apply:end-economy", [&] {
        apply_state_frame(state, frame);
    });
}

} // namespace

bool cuda_turn_backend_available() {
    static const auto info = query_cuda_device();
    return info.runtime_available;
}

const char *cuda_turn_backend_unavailable_reason() {
    static const std::string reason = unavailable_reason();
    return reason.c_str();
}

const char *cuda_turn_backend_capability_summary() {
    return "CUDA turn backend: validation-decision, inspiration, construction-decision, defend, movement-decision, combat-decision, dump, spawn-decision, mining, plunder, regen, capture-decision, over-ship-tax, halite-rebalance on GPU; CPU applies lifecycle, events, and error logs";
}

std::unique_ptr<ITurnBackend> make_cuda_turn_backend(TurnEngine &cpu_reference_engine, const GameConfig &config) {
    if (!cuda_turn_backend_available()) {
        return nullptr;
    }
    return std::make_unique<CudaTurnBackend>(cpu_reference_engine, config);
}

CudaTurnBackend::CudaTurnBackend(TurnEngine &cpu_reference_engine, const GameConfig &config)
    : cpu_reference_engine(cpu_reference_engine), config(config) {}

StepResult CudaTurnBackend::step(GameState &state,
                                 ActionBatch &actions,
                                 events::EventSink &event_sink,
                                 TaskExecutor &executor,
                                 TurnExecutionProfile *profile) {
    StepResult result{};
    return step_from_validated(state, actions, event_sink, executor, profile, std::move(result));
}

bool CudaTurnBackend::supports_prevalidated_step() const {
    return true;
}

bool CudaTurnBackend::supports_split_dump_step() const {
    return true;
}

StepResult CudaTurnBackend::step_from_validated(GameState &state,
                                                ActionBatch &actions,
                                                events::EventSink &event_sink,
                                                TaskExecutor &executor,
                                                TurnExecutionProfile *profile,
                                                StepResult result) {
    (void)cpu_reference_engine;
    const bool has_prevalidated_commands = result.validated_commands.has_value() ||
                                           !result.eliminated_players.empty() ||
                                           !result.non_fatal_errors.empty();
    executor.reset_run_stats();
    if (profile != nullptr) {
        profile->phase_timings.clear();
        profile->executor_type = std::string("cuda+") + executor.executor_name();
        profile->executor_thread_count = executor.executor_thread_count();
    }

    const auto turn_start = std::chrono::steady_clock::now();

    if (!config.runtime.skip_inspiration_phase) {
        run_profiled(profile, "cuda:inspiration", [&] {
            run_cuda_inspiration_on_state(state, config);
        });
    } else {
        run_profiled(profile, "cuda:inspiration-skipped", [&] {
            (void)state;
        });
    }
    if (!has_prevalidated_commands) {
        run_profiled(profile, "cuda:validation-decision", [&] {
            run_cuda_validation_on_state(state, actions, result, config);
        });
    } else {
        run_profiled(profile, "cuda:validation-prevalidated", [&] {
            (void)actions;
            (void)result;
        });
    }
    run_profiled(profile, "cuda:construction-decision", [&] {
        run_cuda_construction_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:defend", [&] {
        run_cuda_defend_on_state(state, result, config);
    });
    run_profiled(profile, "cuda:movement-decision", [&] {
        run_cuda_movement_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:combat-decision", [&] {
        run_cuda_combat_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:heal-noop", [&] {
        (void)result;
    });
    run_profiled(profile, "cuda:dump", [&] {
        run_cuda_dump_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:spawn-decision", [&] {
        run_cuda_spawn_on_state(state, result, event_sink, config);
    });
    run_cuda_post_spawn_frame_pipeline(state, result, event_sink, config, profile);
    run_profiled(profile, "cuda:capture-decision", [&] {
        apply_cuda_capture_decisions(state, result, event_sink, config);
    });
    run_cuda_end_economy_frame_pipeline(state, config, profile);

    if (profile != nullptr) {
        const auto turn_end = std::chrono::steady_clock::now();
        profile->total_turn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(turn_end - turn_start).count();
        profile->executor_stats = executor.run_stats();
    }

    return result;
}

StepResult CudaTurnBackend::step_until_dump(GameState &state,
                                            ActionBatch &actions,
                                            events::EventSink &event_sink,
                                            TaskExecutor &executor,
                                            TurnExecutionProfile *profile,
                                            StepResult result) {
    (void)cpu_reference_engine;
    const bool has_prevalidated_commands = result.validated_commands.has_value() ||
                                           !result.eliminated_players.empty() ||
                                           !result.non_fatal_errors.empty();
    executor.reset_run_stats();
    if (profile != nullptr) {
        profile->phase_timings.clear();
        profile->executor_type = std::string("cuda-split+") + executor.executor_name();
        profile->executor_thread_count = executor.executor_thread_count();
    }

    const auto turn_start = std::chrono::steady_clock::now();

    if (!config.runtime.skip_inspiration_phase) {
        run_profiled(profile, "cuda:inspiration", [&] {
            run_cuda_inspiration_on_state(state, config);
        });
    } else {
        run_profiled(profile, "cuda:inspiration-skipped", [&] {
            (void)state;
        });
    }
    if (!has_prevalidated_commands) {
        run_profiled(profile, "cuda:validation-decision", [&] {
            run_cuda_validation_on_state(state, actions, result, config);
        });
    } else {
        run_profiled(profile, "cuda:validation-prevalidated", [&] {
            (void)actions;
            (void)result;
        });
    }
    run_profiled(profile, "cuda:construction-decision", [&] {
        run_cuda_construction_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:defend", [&] {
        run_cuda_defend_on_state(state, result, config);
    });
    run_profiled(profile, "cuda:movement-decision", [&] {
        run_cuda_movement_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:combat-decision", [&] {
        run_cuda_combat_on_state(state, result, event_sink, config);
    });
    run_profiled(profile, "cuda:heal-noop", [&] {
        (void)result;
    });

    if (profile != nullptr) {
        const auto turn_end = std::chrono::steady_clock::now();
        profile->total_turn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(turn_end - turn_start).count();
        profile->executor_stats = executor.run_stats();
    }

    return result;
}

StepResult CudaTurnBackend::step_after_dump(GameState &state,
                                            ActionBatch &actions,
                                            events::EventSink &event_sink,
                                            TaskExecutor &executor,
                                            TurnExecutionProfile *profile,
                                            StepResult result) {
    (void)cpu_reference_engine;
    (void)actions;
    if (profile != nullptr) {
        profile->executor_type = std::string("cuda-split+") + executor.executor_name();
        profile->executor_thread_count = executor.executor_thread_count();
    }

    const auto turn_start = std::chrono::steady_clock::now();

    run_profiled(profile, "cuda:spawn-decision", [&] {
        run_cuda_spawn_on_state(state, result, event_sink, config);
    });
    run_cuda_post_spawn_frame_pipeline(state, result, event_sink, config, profile);
    run_profiled(profile, "cuda:capture-decision", [&] {
        apply_cuda_capture_decisions(state, result, event_sink, config);
    });
    run_cuda_end_economy_frame_pipeline(state, config, profile);

    if (profile != nullptr) {
        const auto turn_end = std::chrono::steady_clock::now();
        profile->total_turn_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(turn_end - turn_start).count();
        profile->executor_stats = executor.run_stats();
    }

    return result;
}

std::string CudaTurnBackend::backend_name() const {
    (void)cpu_reference_engine;
    return "cuda";
}

} // namespace hlt::gpu
