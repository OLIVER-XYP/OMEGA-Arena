#ifndef COMMANDFRAME_HPP
#define COMMANDFRAME_HPP

#include <vector>

#include "ActionBatch.hpp"
#include "CommandBatch.hpp"
#include "Entity.hpp"
#include "Location.hpp"
#include "Player.hpp"

namespace hlt {

enum class FlatCommandType : unsigned char {
    Move,
    Spawn,
    Construct,
    AttackShip,
    AttackStructure,
    Defend,
    Heal
};

struct FlatCommand {
    Player::id_type player{Player::None};
    Entity::id_type entity{Entity::None};
    Entity::id_type target{Entity::None};
    Player::id_type target_player{Player::None};
    Location target_location{0, 0};
    Direction direction{Direction::Still};
    FlatCommandType type{FlatCommandType::Move};
};

struct CommandFrame {
    std::vector<FlatCommand> commands;
};

CommandFrame flatten_action_batch(const ActionBatch &actions);
CommandFrame flatten_command_batch(const CommandBatch &commands);

} // namespace hlt

#endif // COMMANDFRAME_HPP
