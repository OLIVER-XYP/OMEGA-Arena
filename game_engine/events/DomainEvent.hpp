#ifndef DOMAINEVENT_HPP
#define DOMAINEVENT_HPP

#include <string>
#include <variant>
#include <vector>

#include "Entity.hpp"
#include "Location.hpp"
#include "Player.hpp"

namespace hlt::events {

struct ShipMovedEvent {
    hlt::Entity::id_type entity = hlt::Entity::None;
    hlt::Location from{0, 0};
    hlt::Location to{0, 0};
};

struct MinedEvent {
    hlt::Entity::id_type entity = hlt::Entity::None;
    hlt::Player::id_type owner = hlt::Player::None;
    hlt::Location location{0, 0};
    hlt::energy_type extracted{};
    hlt::energy_type gained{};
    bool was_captured{};
};

struct DepositedEvent {
    hlt::Player::id_type player = hlt::Player::None;
    hlt::Location location{0, 0};
    hlt::energy_type amount{};
};

struct CombatResolvedEvent {
    hlt::Location attacker_location{0, 0};
    hlt::Entity::id_type attacker = hlt::Entity::None;
    hlt::Location target_location{0, 0};
    hlt::Entity::id_type target = hlt::Entity::None;
    bool hit{};
};

struct CollisionResolvedEvent {
    hlt::Location location{0, 0};
    std::vector<hlt::Entity::id_type> ships;
};

struct EntityDestroyedEvent {
    hlt::Entity::id_type entity = hlt::Entity::None;
    hlt::Player::id_type owner = hlt::Player::None;
    hlt::Location location{0, 0};
};

struct SpawnedEvent {
    hlt::Player::id_type player = hlt::Player::None;
    hlt::Entity::id_type entity = hlt::Entity::None;
    hlt::Location location{0, 0};
    hlt::energy_type energy{};
};

struct ConstructionResolvedEvent {
    hlt::Location location{0, 0};
    hlt::Player::id_type player = hlt::Player::None;
    hlt::Entity::id_type entity = hlt::Entity::None;
};

struct CapturedEvent {
    hlt::Location location{0, 0};
    hlt::Player::id_type old_owner = hlt::Player::None;
    hlt::Entity::id_type old_id = hlt::Entity::None;
    hlt::Player::id_type new_owner = hlt::Player::None;
    hlt::Entity::id_type new_id = hlt::Entity::None;
};

struct PlayerEliminatedEvent {
    hlt::Player::id_type player = hlt::Player::None;
    std::string reason;
};

using DomainEvent = std::variant<ShipMovedEvent,
                                 MinedEvent,
                                 DepositedEvent,
                                 CombatResolvedEvent,
                                 CollisionResolvedEvent,
                                 EntityDestroyedEvent,
                                 SpawnedEvent,
                                 ConstructionResolvedEvent,
                                 CapturedEvent,
                                 PlayerEliminatedEvent>;

} // namespace hlt::events

#endif // DOMAINEVENT_HPP
