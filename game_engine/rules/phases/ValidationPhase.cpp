#include "ValidationPhase.hpp"

#include <iterator>
#include <vector>

#include "LocalStepResult.hpp"
#include "PhaseHelpers.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

namespace {

static constexpr auto MAX_COMMANDS_PER_ENTITY = 1;

bool check_move_ownership(const Player &player,
                          Entity::id_type entity,
                          const MoveCommand &command,
                          id_map<Player, std::vector<std::reference_wrapper<const MoveCommand>>> &faulty) {
    if (!player.has_entity(entity)) {
        faulty[player.id].emplace_back(command);
        return false;
    }
    return true;
}

bool check_construct_ownership(const Player &player,
                               Entity::id_type entity,
                               const ConstructCommand &command,
                               id_map<Player, std::vector<std::reference_wrapper<const ConstructCommand>>> &faulty) {
    if (!player.has_entity(entity)) {
        faulty[player.id].emplace_back(command);
        return false;
    }
    return true;
}

void add_occurrence(Entity::id_type entity,
                    const Command &command,
                    id_map<Entity, std::pair<int, ErrorContext>> &occurrences,
                    id_map<Entity, std::reference_wrapper<const Command>> &first_faulty) {
    auto &entry = occurrences[entity];
    if (entry.first++ == MAX_COMMANDS_PER_ENTITY) {
        first_faulty.emplace(entity, command);
    } else {
        entry.second.emplace_back(command);
    }
}

void add_expense(const Player &player,
                 const Command &command,
                 energy_type amount,
                 id_map<Player, std::pair<energy_type, ErrorContext>> &expenses,
                 id_map<Player, std::reference_wrapper<const Command>> &first_faulty) {
    auto &entry = expenses[player.id];
    if ((entry.first += amount) > player.energy) {
        if (auto [_, inserted] = first_faulty.emplace(player.id, command); !inserted) {
            entry.second.emplace_back(command);
        }
    } else {
        entry.second.emplace_back(command);
    }
}

struct ValidationShardResult {
    CommandBatch batch;
    LocalStepResult local_result;
    id_map<Entity, std::pair<int, ErrorContext>> occurrences;
    id_map<Entity, std::reference_wrapper<const Command>> occurrences_first_faulty;
    id_map<Player, std::pair<energy_type, ErrorContext>> expenses;
    id_map<Player, std::reference_wrapper<const Command>> expenses_first_faulty;
    id_map<Player, std::vector<std::reference_wrapper<const MoveCommand>>> move_ownership_faulty;
    id_map<Player, std::vector<std::reference_wrapper<const ConstructCommand>>> construct_ownership_faulty;
};

ValidationShardResult validate_player_commands(const GameState &state,
                                               const GameConfig &config,
                                               Player &player,
                                               const std::vector<std::unique_ptr<Command>> &command_list) {
    ValidationShardResult shard;
    auto &store = state.store_ref();
    auto &map = state.map_ref();

    for (const auto &command_ptr : command_list) {
        const Command &command = *command_ptr;
        if (auto move = dynamic_cast<const MoveCommand *>(&command)) {
            if (check_move_ownership(player, move->entity, *move, shard.move_ownership_faulty)) {
                add_occurrence(move->entity, *move, shard.occurrences, shard.occurrences_first_faulty);
                shard.batch.moves[player.id].emplace_back(*move);
            }
            continue;
        }
        if (auto construct = dynamic_cast<const ConstructCommand *>(&command)) {
            if (check_construct_ownership(player, construct->entity, *construct, shard.construct_ownership_faulty)) {
                add_occurrence(construct->entity, *construct, shard.occurrences, shard.occurrences_first_faulty);
                auto cost = scaled_dropoff_cost(
                    config.ruleset.economy.dropoff_cost,
                    player.dropoffs.size(),
                    config.ruleset.economy.dropoff_cost_growth);
                const auto location = player.get_entity_location(construct->entity);
                const auto &cell = map.at(location);
                const auto &entity = store.get_entity(construct->entity);
                if (cell.energy + entity.energy >= cost) {
                    cost = 0;
                } else {
                    cost -= cell.energy + entity.energy;
                }
                add_expense(player, *construct, cost, shard.expenses, shard.expenses_first_faulty);
                shard.batch.constructs[player.id].emplace_back(*construct);
            }
            continue;
        }
        if (auto spawn = dynamic_cast<const SpawnCommand *>(&command)) {
            const auto spawn_cost = scaled_spawn_cost(
                config.ruleset.economy.new_entity_energy_cost,
                player.entities.size(),
                config.ruleset.economy.spawn_cost_growth,
                config.ruleset.economy.spawn_quad_threshold,
                config.ruleset.economy.spawn_quad_growth);
            add_expense(player, *spawn, spawn_cost, shard.expenses, shard.expenses_first_faulty);
            shard.batch.spawns[player.id].emplace_back(*spawn);
            continue;
        }
        if (auto attack = dynamic_cast<const AttackCommand *>(&command)) {
            if (config.ruleset.combat.enable_combat_commands && player.has_entity(attack->entity)) {
                add_occurrence(attack->entity, *attack, shard.occurrences, shard.occurrences_first_faulty);
                shard.batch.attacks[player.id].emplace_back(*attack);
            }
            continue;
        }
        if (auto defend = dynamic_cast<const DefendCommand *>(&command)) {
            if (config.ruleset.combat.enable_combat_commands) {
                if (player.has_entity(defend->entity)) {
                    add_occurrence(defend->entity, *defend, shard.occurrences, shard.occurrences_first_faulty);
                    shard.batch.defends[player.id].emplace_back(*defend);
                } else {
                    shard.local_result.non_fatal_errors.push_back(
                        EntityNotFoundError<DefendCommand>(player.id, *defend).log_message());
                }
            }
            continue;
        }
        if (auto heal = dynamic_cast<const HealCommand *>(&command)) {
            if (config.ruleset.combat.enable_combat_commands) {
                if (player.has_entity(heal->entity)) {
                    add_occurrence(heal->entity, *heal, shard.occurrences, shard.occurrences_first_faulty);
                    shard.batch.heals[player.id].emplace_back(*heal);
                } else {
                    shard.local_result.non_fatal_errors.push_back(
                        EntityNotFoundError<HealCommand>(player.id, *heal).log_message());
                }
            }
        }
    }

    return shard;
}

void merge_batch(CommandBatch &target, CommandBatch &&source) {
    for (auto &[player_id, commands] : source.constructs) {
        auto &dest = target.constructs[player_id];
        append_range(dest, commands);
    }
    for (auto &[player_id, commands] : source.defends) {
        auto &dest = target.defends[player_id];
        append_range(dest, commands);
    }
    for (auto &[player_id, commands] : source.moves) {
        auto &dest = target.moves[player_id];
        append_range(dest, commands);
    }
    for (auto &[player_id, commands] : source.attacks) {
        auto &dest = target.attacks[player_id];
        append_range(dest, commands);
    }
    for (auto &[player_id, commands] : source.heals) {
        auto &dest = target.heals[player_id];
        append_range(dest, commands);
    }
    for (auto &[player_id, commands] : source.spawns) {
        auto &dest = target.spawns[player_id];
        append_range(dest, commands);
    }
}

void merge_local_result(LocalStepResult &target, LocalStepResult &&source) {
    target.non_fatal_errors.insert(target.non_fatal_errors.end(),
                                   std::make_move_iterator(source.non_fatal_errors.begin()),
                                   std::make_move_iterator(source.non_fatal_errors.end()));
    target.eliminated_players.insert(target.eliminated_players.end(),
                                     std::make_move_iterator(source.eliminated_players.begin()),
                                     std::make_move_iterator(source.eliminated_players.end()));
    target.changed_entities.insert(source.changed_entities.begin(), source.changed_entities.end());
    target.changed_cells.insert(source.changed_cells.begin(), source.changed_cells.end());
}

void merge_shard_state(ValidationShardResult &global, ValidationShardResult &&shard) {
    merge_batch(global.batch, std::move(shard.batch));
    merge_local_result(global.local_result, std::move(shard.local_result));

    for (auto &[entity_id, occurrence] : shard.occurrences) {
        auto &target = global.occurrences[entity_id];
        target.first += occurrence.first;
        append_range(target.second, occurrence.second);
    }
    for (auto &[entity_id, command] : shard.occurrences_first_faulty) {
        global.occurrences_first_faulty.emplace(entity_id, command);
    }
    for (auto &[player_id, expense] : shard.expenses) {
        auto &target = global.expenses[player_id];
        target.first += expense.first;
        append_range(target.second, expense.second);
    }
    for (auto &[player_id, command] : shard.expenses_first_faulty) {
        global.expenses_first_faulty.emplace(player_id, command);
    }
    for (auto &[player_id, commands] : shard.move_ownership_faulty) {
        auto &target = global.move_ownership_faulty[player_id];
        append_range(target, commands);
    }
    for (auto &[player_id, commands] : shard.construct_ownership_faulty) {
        auto &target = global.construct_ownership_faulty[player_id];
        append_range(target, commands);
    }
}

} // namespace

void ValidationPhase::execute(RuleContext &context) const {
    auto &state = context.state();
    auto &store = state.store_ref();
    auto &actions = context.actions();
    auto &result = context.result();
    ValidationShardResult aggregate;

    std::vector<Player::id_type> player_ids;
    player_ids.reserve(actions.size());
    for (const auto &[player_id, command_list] : actions) {
        (void)command_list;
        if (store.players_ref().find(player_id) != store.players_ref().end()) {
            player_ids.push_back(player_id);
        }
    }

    std::vector<ValidationShardResult> shards(player_ids.size());
    InlineExecutor fallback_executor;
    auto *executor = context.executor() != nullptr ? context.executor() : &fallback_executor;
    executor->parallel_for(0, player_ids.size(), [&](std::size_t index) {
        const auto player_id = player_ids[index];
        auto &player = store.players_ref().at(player_id);
        const auto &command_list = actions.at(player_id);
        shards[index] = validate_player_commands(state, context.config(), player, command_list);
    });

    for (auto &shard : shards) {
        merge_shard_state(aggregate, std::move(shard));
    }

    merge_into(result, std::move(aggregate.local_result));

    bool success = true;
    for (const auto &[player_id, misowned] : aggregate.move_ownership_faulty) {
        for (const auto &faulty : misowned) {
            append_error(result, std::make_unique<EntityNotFoundError<MoveCommand>>(player_id, faulty));
        }
        success = false;
    }
    for (const auto &[player_id, misowned] : aggregate.construct_ownership_faulty) {
        for (const auto &faulty : misowned) {
            append_error(result, std::make_unique<EntityNotFoundError<ConstructCommand>>(player_id, faulty));
        }
        success = false;
    }

    for (auto &[player_id, faulty] : aggregate.expenses_first_faulty) {
        const auto &player = store.get_player(player_id);
        auto &[energy, context_commands] = aggregate.expenses[player_id];
        append_error(result, std::make_unique<PlayerInsufficientEnergyError>(player_id, faulty, context_commands, player.energy, energy));
        result.eliminated_players.push_back(player_id);
        success = false;
    }
    for (auto &[entity_id, faulty] : aggregate.occurrences_first_faulty) {
        const auto owner = store.get_entity(entity_id).owner;
        auto &[count, context_commands] = aggregate.occurrences[entity_id];
        (void)count;
        append_error(result, std::make_unique<ExcessiveCommandsError>(owner, faulty, context_commands, entity_id));
        result.eliminated_players.push_back(owner);
        success = false;
    }

    if (!success) {
        for (auto player_id : result.eliminated_players) {
            actions.erase(player_id);
        }
        return;
    }

    result.validated_commands = std::move(aggregate.batch);
}

} // namespace hlt::rules::phases


