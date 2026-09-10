#pragma once

#include "entity.hpp"
#include "constants.hpp"
#include "command.hpp"

#include <memory>

namespace hlt {
    struct Ship : Entity {
        Halite halite;
        int hp;

        Ship(PlayerId player_id, EntityId ship_id, int x, int y, Halite halite, int hp = 100) :
            Entity(player_id, ship_id, x, y),
            halite(halite),
            hp(hp)
        {}

        bool is_full() const {
            return halite >= constants::MAX_HALITE;
        }

        Command make_dropoff() const {
            return hlt::command::transform_ship_into_dropoff_site(id);
        }

        Command move(Direction direction) const {
            return hlt::command::move(id, direction);
        }

        Command stay_still() const {
            return hlt::command::stay(id);
        }

        Command attack(const std::shared_ptr<Ship> &target) const {
            return hlt::command::attack(id, target->id);
        }

        Command attack_structure(PlayerId owner_id, const Position &target_pos) const {
            return hlt::command::attack_structure(id, owner_id, target_pos);
        }

        Command defend() const {
            return hlt::command::defend(id);
        }

        static std::shared_ptr<Ship> _generate(PlayerId player_id);
    };
}
