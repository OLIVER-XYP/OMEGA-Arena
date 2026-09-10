#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "Entity.hpp"
#include "Map.hpp"
#include "Player.hpp"

#include "nlohmann/json_fwd.hpp"

namespace hlt {

/** Abstract type of commands issued by the user. */
class Command {
public:
    /** The command names. */
    enum Name : char {
        Move = 'm',
        Stay = 's',
        Spawn = 'g',
        Construct = 'c',
        Attack = 'a',
        Defend = 'd',
        Heal = 'h',
        AttackStructure = 'x'
    };

    virtual void to_json(nlohmann::json &json) const = 0;
    virtual std::string to_bot_serial() const = 0;
    virtual ~Command() = default;
};

void to_json(nlohmann::json &json, const Command &command);
void to_json(nlohmann::json &json, const std::unique_ptr<Command> &command);
std::istream &operator>>(std::istream &istream, std::unique_ptr<Command> &command);
std::ostream &operator<<(std::ostream &ostream, std::unique_ptr<Command> &command);

class MoveCommand final : public Command {
public:
    const Entity::id_type entity;
    const Direction direction;

    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;
    MoveCommand(const Entity::id_type &entity, Direction direction) : entity(entity), direction(direction) {}
};

class SpawnCommand final : public Command {
public:
    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;
    explicit SpawnCommand() {}
};

class ConstructCommand final : public Command {
public:
    const Entity::id_type entity;

    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;
    explicit ConstructCommand(const Entity::id_type &entity) : entity(entity) {}
};

class AttackCommand final : public Command {
public:
    const Entity::id_type entity;
    const Entity::id_type target;

    bool is_structure_target;
    Player::id_type target_structure_owner;
    Location target_structure_location;

    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;

    AttackCommand(const Entity::id_type &entity, const Entity::id_type &target)
        : entity(entity), target(target), is_structure_target(false),
          target_structure_owner(Player::None), target_structure_location(0, 0) {}

    AttackCommand(const Entity::id_type &entity,
                  const Player::id_type &owner,
                  Location location)
        : entity(entity), target(Entity::None), is_structure_target(true),
          target_structure_owner(owner), target_structure_location(location) {}
};

class DefendCommand final : public Command {
public:
    const Entity::id_type entity;

    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;
    explicit DefendCommand(const Entity::id_type &entity) : entity(entity) {}
};

class HealCommand final : public Command {
public:
    const Entity::id_type entity;

    void to_json(nlohmann::json &json) const override;
    std::string to_bot_serial() const override;
    explicit HealCommand(const Entity::id_type &entity) : entity(entity) {}
};

}

#endif // COMMAND_HPP
