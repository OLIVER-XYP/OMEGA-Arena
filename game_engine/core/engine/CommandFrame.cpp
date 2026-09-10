#include "CommandFrame.hpp"

namespace hlt {

namespace {

void append_command(CommandFrame &frame, Player::id_type player, const MoveCommand &command) {
    frame.commands.push_back(FlatCommand{player,
                                         command.entity,
                                         Entity::None,
                                         Player::None,
                                         Location{0, 0},
                                         command.direction,
                                         FlatCommandType::Move});
}

void append_command(CommandFrame &frame, Player::id_type player, const SpawnCommand &command) {
    (void)command;
    frame.commands.push_back(FlatCommand{player,
                                         Entity::None,
                                         Entity::None,
                                         Player::None,
                                         Location{0, 0},
                                         Direction::Still,
                                         FlatCommandType::Spawn});
}

void append_command(CommandFrame &frame, Player::id_type player, const ConstructCommand &command) {
    frame.commands.push_back(FlatCommand{player,
                                         command.entity,
                                         Entity::None,
                                         Player::None,
                                         Location{0, 0},
                                         Direction::Still,
                                         FlatCommandType::Construct});
}

void append_command(CommandFrame &frame, Player::id_type player, const AttackCommand &command) {
    const auto type = command.is_structure_target ? FlatCommandType::AttackStructure : FlatCommandType::AttackShip;
    frame.commands.push_back(FlatCommand{player,
                                         command.entity,
                                         command.target,
                                         command.target_structure_owner,
                                         command.target_structure_location,
                                         Direction::Still,
                                         type});
}

void append_command(CommandFrame &frame, Player::id_type player, const DefendCommand &command) {
    frame.commands.push_back(FlatCommand{player,
                                         command.entity,
                                         Entity::None,
                                         Player::None,
                                         Location{0, 0},
                                         Direction::Still,
                                         FlatCommandType::Defend});
}

void append_command(CommandFrame &frame, Player::id_type player, const HealCommand &command) {
    frame.commands.push_back(FlatCommand{player,
                                         command.entity,
                                         Entity::None,
                                         Player::None,
                                         Location{0, 0},
                                         Direction::Still,
                                         FlatCommandType::Heal});
}

} // namespace

CommandFrame flatten_action_batch(const ActionBatch &actions) {
    CommandFrame frame;
    for (const auto &[player_id, command_list] : actions) {
        frame.commands.reserve(frame.commands.size() + command_list.size());
        for (const auto &command_ptr : command_list) {
            const Command &command = *command_ptr;
            if (const auto *move = dynamic_cast<const MoveCommand *>(&command)) {
                append_command(frame, player_id, *move);
            } else if (const auto *spawn = dynamic_cast<const SpawnCommand *>(&command)) {
                append_command(frame, player_id, *spawn);
            } else if (const auto *construct = dynamic_cast<const ConstructCommand *>(&command)) {
                append_command(frame, player_id, *construct);
            } else if (const auto *attack = dynamic_cast<const AttackCommand *>(&command)) {
                append_command(frame, player_id, *attack);
            } else if (const auto *defend = dynamic_cast<const DefendCommand *>(&command)) {
                append_command(frame, player_id, *defend);
            } else if (const auto *heal = dynamic_cast<const HealCommand *>(&command)) {
                append_command(frame, player_id, *heal);
            }
        }
    }
    return frame;
}

CommandFrame flatten_command_batch(const CommandBatch &commands) {
    CommandFrame frame;
    for (const auto &[player_id, list] : commands.moves) {
        for (const MoveCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    for (const auto &[player_id, list] : commands.constructs) {
        for (const ConstructCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    for (const auto &[player_id, list] : commands.defends) {
        for (const DefendCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    for (const auto &[player_id, list] : commands.attacks) {
        for (const AttackCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    for (const auto &[player_id, list] : commands.heals) {
        for (const HealCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    for (const auto &[player_id, list] : commands.spawns) {
        for (const SpawnCommand &command : list) {
            append_command(frame, player_id, command);
        }
    }
    return frame;
}

} // namespace hlt
