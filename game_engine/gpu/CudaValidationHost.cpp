#include "CudaValidation.hpp"

#include "CommandError.hpp"
#include "GameState.hpp"
#include "PhaseHelpers.hpp"

namespace hlt::gpu {
namespace {

static constexpr auto MAX_COMMANDS_PER_ENTITY = 1;

void append_validated_command(CommandBatch &batch, const ValidationCommandRef &ref) {
    if (ref.move != nullptr) {
        batch.moves[ref.player].emplace_back(*ref.move);
    } else if (ref.construct != nullptr) {
        batch.constructs[ref.player].emplace_back(*ref.construct);
    } else if (ref.spawn != nullptr) {
        batch.spawns[ref.player].emplace_back(*ref.spawn);
    } else if (ref.attack != nullptr) {
        batch.attacks[ref.player].emplace_back(*ref.attack);
    } else if (ref.defend != nullptr) {
        batch.defends[ref.player].emplace_back(*ref.defend);
    } else if (ref.heal != nullptr) {
        batch.heals[ref.player].emplace_back(*ref.heal);
    }
}

} // namespace

std::vector<ValidationCommandRef> flatten_validation_actions(const ActionBatch &actions, const Store &store) {
    std::vector<ValidationCommandRef> refs;
    for (const auto &[player_id, command_list] : actions) {
        if (store.players_ref().find(player_id) == store.players_ref().end()) {
            continue;
        }
        refs.reserve(refs.size() + command_list.size());
        for (const auto &command_ptr : command_list) {
            const Command &command = *command_ptr;
            ValidationCommandRef ref;
            ref.player = player_id;
            ref.command = &command;
            if (const auto *move = dynamic_cast<const MoveCommand *>(&command)) {
                ref.move = move;
                ref.flat = FlatCommand{player_id,
                                       move->entity,
                                       Entity::None,
                                       Player::None,
                                       Location{0, 0},
                                       move->direction,
                                       FlatCommandType::Move};
            } else if (const auto *construct = dynamic_cast<const ConstructCommand *>(&command)) {
                ref.construct = construct;
                ref.flat = FlatCommand{player_id,
                                       construct->entity,
                                       Entity::None,
                                       Player::None,
                                       Location{0, 0},
                                       Direction::Still,
                                       FlatCommandType::Construct};
            } else if (const auto *spawn = dynamic_cast<const SpawnCommand *>(&command)) {
                ref.spawn = spawn;
                ref.flat = FlatCommand{player_id,
                                       Entity::None,
                                       Entity::None,
                                       Player::None,
                                       Location{0, 0},
                                       Direction::Still,
                                       FlatCommandType::Spawn};
            } else if (const auto *attack = dynamic_cast<const AttackCommand *>(&command)) {
                ref.attack = attack;
                const auto type = attack->is_structure_target ? FlatCommandType::AttackStructure : FlatCommandType::AttackShip;
                ref.flat = FlatCommand{player_id,
                                       attack->entity,
                                       attack->target,
                                       attack->target_structure_owner,
                                       attack->target_structure_location,
                                       Direction::Still,
                                       type};
            } else if (const auto *defend = dynamic_cast<const DefendCommand *>(&command)) {
                ref.defend = defend;
                ref.flat = FlatCommand{player_id,
                                       defend->entity,
                                       Entity::None,
                                       Player::None,
                                       Location{0, 0},
                                       Direction::Still,
                                       FlatCommandType::Defend};
            } else if (const auto *heal = dynamic_cast<const HealCommand *>(&command)) {
                ref.heal = heal;
                ref.flat = FlatCommand{player_id,
                                       heal->entity,
                                       Entity::None,
                                       Player::None,
                                       Location{0, 0},
                                       Direction::Still,
                                       FlatCommandType::Heal};
            } else {
                continue;
            }
            refs.push_back(ref);
        }
    }
    return refs;
}

CommandFrame command_frame_from_validation_refs(const std::vector<ValidationCommandRef> &refs) {
    CommandFrame frame;
    frame.commands.reserve(refs.size());
    for (const auto &ref : refs) {
        frame.commands.push_back(ref.flat);
    }
    return frame;
}

void apply_validation_decisions(GameState &state,
                                ActionBatch &actions,
                                StepResult &result,
                                const std::vector<ValidationCommandRef> &refs,
                                const std::vector<ValidationFrameDecision> &decisions) {
    CommandBatch batch;
    id_map<Entity, std::pair<int, ErrorContext>> occurrences;
    id_map<Entity, std::reference_wrapper<const Command>> occurrences_first_faulty;
    id_map<Player, std::pair<energy_type, ErrorContext>> expenses;
    id_map<Player, std::reference_wrapper<const Command>> expenses_first_faulty;
    id_map<Player, std::vector<std::reference_wrapper<const MoveCommand>>> move_ownership_faulty;
    id_map<Player, std::vector<std::reference_wrapper<const ConstructCommand>>> construct_ownership_faulty;

    for (std::size_t index = 0; index < refs.size() && index < decisions.size(); ++index) {
        const auto &ref = refs[index];
        const auto &decision = decisions[index];
        if (!decision.command || !decision.player_exists) {
            continue;
        }

        if (ref.move != nullptr && !decision.ownership_ok) {
            move_ownership_faulty[ref.player].emplace_back(*ref.move);
            continue;
        }
        if (ref.construct != nullptr && !decision.ownership_ok) {
            construct_ownership_faulty[ref.player].emplace_back(*ref.construct);
            continue;
        }
        if ((ref.defend != nullptr || ref.heal != nullptr) && decision.combat_enabled && !decision.ownership_ok) {
            if (ref.defend != nullptr) {
                result.non_fatal_errors.push_back(EntityNotFoundError<DefendCommand>(ref.player, *ref.defend).log_message());
            } else {
                result.non_fatal_errors.push_back(EntityNotFoundError<HealCommand>(ref.player, *ref.heal).log_message());
            }
            continue;
        }

        if (decision.occurrence_command) {
            auto &entry = occurrences[decision.entity];
            if (entry.first++ == MAX_COMMANDS_PER_ENTITY) {
                occurrences_first_faulty.emplace(decision.entity, *ref.command);
            } else {
                entry.second.emplace_back(*ref.command);
            }
        }

        if (decision.expense_command) {
            const auto &player = state.store_ref().get_player(ref.player);
            auto &entry = expenses[ref.player];
            if ((entry.first += decision.expense) > player.energy) {
                if (auto [_, inserted] = expenses_first_faulty.emplace(ref.player, *ref.command); !inserted) {
                    entry.second.emplace_back(*ref.command);
                }
            } else {
                entry.second.emplace_back(*ref.command);
            }
        }

        if (decision.include_in_batch) {
            append_validated_command(batch, ref);
        }
    }

    bool success = true;
    for (const auto &[player_id, misowned] : move_ownership_faulty) {
        for (const auto &faulty : misowned) {
            rules::phases::append_error(result, std::make_unique<EntityNotFoundError<MoveCommand>>(player_id, faulty));
        }
        success = false;
    }
    for (const auto &[player_id, misowned] : construct_ownership_faulty) {
        for (const auto &faulty : misowned) {
            rules::phases::append_error(result, std::make_unique<EntityNotFoundError<ConstructCommand>>(player_id, faulty));
        }
        success = false;
    }
    for (auto &[player_id, faulty] : expenses_first_faulty) {
        const auto &player = state.store_ref().get_player(player_id);
        auto &[energy, context_commands] = expenses[player_id];
        rules::phases::append_error(result, std::make_unique<PlayerInsufficientEnergyError>(player_id, faulty, context_commands, player.energy, energy));
        result.eliminated_players.push_back(player_id);
        success = false;
    }
    for (auto &[entity_id, faulty] : occurrences_first_faulty) {
        const auto owner = state.store_ref().get_entity(entity_id).owner;
        auto &[count, context_commands] = occurrences[entity_id];
        (void)count;
        rules::phases::append_error(result, std::make_unique<ExcessiveCommandsError>(owner, faulty, context_commands, entity_id));
        result.eliminated_players.push_back(owner);
        success = false;
    }

    if (!success) {
        for (auto player_id : result.eliminated_players) {
            actions.erase(player_id);
        }
        return;
    }

    result.validated_commands = std::move(batch);
}

} // namespace hlt::gpu
