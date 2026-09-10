#include <memory>

#include "BotCommunicationError.hpp"
#include "Constants.hpp"
#include "Command.hpp"
#include "CommandError.hpp"

constexpr auto JSON_TYPE_KEY = "type";
constexpr auto JSON_ENTITY_KEY = "id";
constexpr auto JSON_DIRECTION_KEY = "direction";

namespace hlt {

void to_json(nlohmann::json &json, const Command &command) { command.to_json(json); }
void to_json(nlohmann::json &json, const std::unique_ptr<Command> &command) { command->to_json(json); }

std::istream &operator>>(std::istream &istream, std::unique_ptr<Command> &command) {
    char command_type;
    if (istream >> command_type) {
        switch (command_type) {
        case Command::Name::Move: {
            Entity::id_type entity;
            Direction direction;
            istream >> entity >> direction;
            command = std::make_unique<MoveCommand>(entity, direction);
            break;
        }
        case Command::Name::Stay: {
            Entity::id_type entity;
            istream >> entity;
            command = std::make_unique<MoveCommand>(entity, Direction::Still);
            break;
        }
        case Command::Name::Spawn: {
            command = std::make_unique<SpawnCommand>();
            break;
        }
        case Command::Name::Construct: {
            Entity::id_type entity;
            istream >> entity;
            command = std::make_unique<ConstructCommand>(entity);
            break;
        }
        case Command::Name::Attack: {
            if (!Constants::get().ENABLE_COMBAT_COMMANDS) {
                throw BotCommunicationError("a", istream.tellg(),
                    "combat commands are disabled (ENABLE_COMBAT_COMMANDS=false)");
            }
            Entity::id_type entity, target;
            istream >> entity >> target;
            command = std::make_unique<AttackCommand>(entity, target);
            break;
        }
        case Command::Name::AttackStructure: {
            if (!Constants::get().ENABLE_COMBAT_COMMANDS) {
                throw BotCommunicationError("x", istream.tellg(),
                    "combat commands are disabled (ENABLE_COMBAT_COMMANDS=false)");
            }
            Entity::id_type entity;
            Player::id_type owner;
            dimension_type x, y;
            istream >> entity >> owner >> x >> y;
            command = std::make_unique<AttackCommand>(entity, owner, Location{x, y});
            break;
        }
        case Command::Name::Defend: {
            if (!Constants::get().ENABLE_COMBAT_COMMANDS) {
                throw BotCommunicationError("d", istream.tellg(),
                    "combat commands are disabled (ENABLE_COMBAT_COMMANDS=false)");
            }
            Entity::id_type entity;
            istream >> entity;
            command = std::make_unique<DefendCommand>(entity);
            break;
        }
        case Command::Name::Heal: {
            if (!Constants::get().ENABLE_COMBAT_COMMANDS) {
                throw BotCommunicationError("h", istream.tellg(),
                    "combat commands are disabled (ENABLE_COMBAT_COMMANDS=false)");
            }
            Entity::id_type entity;
            istream >> entity;
            command = std::make_unique<HealCommand>(entity);
            break;
        }
        default:
            throw BotCommunicationError(to_string(command_type), istream.tellg());
        }
    }
    return istream;
}

std::ostream &operator<<(std::ostream &ostream, std::unique_ptr<Command> &command) {
    return ostream << command->to_bot_serial();
}

void MoveCommand::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, to_string(Name::Move)},
            {JSON_ENTITY_KEY, entity},
            {JSON_DIRECTION_KEY, direction}};
}

std::string MoveCommand::to_bot_serial() const {
    return to_string(Name::Move) + " " + to_string(entity) + " " + to_string(static_cast<char>(direction));
}

void SpawnCommand::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, to_string(Name::Spawn)}};
}

std::string SpawnCommand::to_bot_serial() const {
    return to_string(Name::Spawn);
}

void ConstructCommand::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, to_string(Name::Construct)},
            {JSON_ENTITY_KEY, entity}};
}

std::string ConstructCommand::to_bot_serial() const {
    return to_string(Name::Construct) + " " + to_string(entity);
}

void AttackCommand::to_json(nlohmann::json &json) const {
    if (is_structure_target) {
        json = {{JSON_TYPE_KEY, to_string(Name::AttackStructure)},
                {JSON_ENTITY_KEY, entity},
                {"target_owner", target_structure_owner},
                {"target_location", target_structure_location}};
    } else {
        json = {{JSON_TYPE_KEY, to_string(Name::Attack)},
                {JSON_ENTITY_KEY, entity},
                {"target", target}};
    }
}

std::string AttackCommand::to_bot_serial() const {
    if (is_structure_target) {
        return to_string(Name::AttackStructure) + " " + to_string(entity) + " " +
               to_string(target_structure_owner) + " " +
               std::to_string(target_structure_location.x) + " " +
               std::to_string(target_structure_location.y);
    }
    return to_string(Name::Attack) + " " + to_string(entity) + " " + to_string(target);
}

void DefendCommand::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, to_string(Name::Defend)},
            {JSON_ENTITY_KEY, entity}};
}

std::string DefendCommand::to_bot_serial() const {
    return to_string(Name::Defend) + " " + to_string(entity);
}

void HealCommand::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, to_string(Name::Heal)},
            {JSON_ENTITY_KEY, entity}};
}

std::string HealCommand::to_bot_serial() const {
    return to_string(Name::Heal) + " " + to_string(entity);
}

}