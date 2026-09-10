#pragma once

#include "direction.hpp"
#include "types.hpp"
#include "position.hpp"

#include <string>

namespace hlt {
    typedef std::string Command;

    namespace command {
        Command spawn_ship();
        Command transform_ship_into_dropoff_site(EntityId id);
        Command move(EntityId id, Direction direction);
        Command stay(EntityId id);
        Command attack(EntityId attacker_id, EntityId target_id);
        Command attack_structure(EntityId attacker_id, PlayerId owner_id, const Position &target_pos);
        Command defend(EntityId id);
    }
}
