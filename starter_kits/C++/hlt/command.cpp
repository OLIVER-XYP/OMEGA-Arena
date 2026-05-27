#include "command.hpp"

#include <iostream>

constexpr char GENERATE= 'g';
constexpr char CONSTRUCT[] = "c ";
constexpr char MOVE[] = "m ";
constexpr char ATTACK[] = "a ";
constexpr char ATTACK_STRUCTURE[] = "x ";
constexpr char STAY[] = "s ";
constexpr char DEFEND[] = "d ";

hlt::Command hlt::command::spawn_ship() {
    return std::string(1, GENERATE);
}

hlt::Command hlt::command::transform_ship_into_dropoff_site(EntityId id) {
    return CONSTRUCT + std::to_string(id);
}

hlt::Command hlt::command::move(EntityId id, hlt::Direction direction) {
    return MOVE + std::to_string(id) + ' ' + static_cast<char>(direction);
}

hlt::Command hlt::command::stay(EntityId id) {
    return STAY + std::to_string(id);
}

hlt::Command hlt::command::attack(EntityId attacker_id, EntityId target_id) {
    return ATTACK + std::to_string(attacker_id) + ' ' + std::to_string(target_id);
}

hlt::Command hlt::command::attack_structure(EntityId attacker_id, PlayerId owner_id, const Position &target_pos) {
    return ATTACK_STRUCTURE + std::to_string(attacker_id) + ' ' + std::to_string(owner_id) + ' ' +
           std::to_string(target_pos.x) + ' ' + std::to_string(target_pos.y);
}

hlt::Command hlt::command::defend(EntityId id) {
    return DEFEND + std::to_string(id);
}
