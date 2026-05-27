#include "ship.hpp"
#include "input.hpp"

std::shared_ptr<hlt::Ship> hlt::Ship::_generate(hlt::PlayerId player_id) {
    hlt::EntityId ship_id;
    int x, y, hp;
    hlt::Halite halite;
    hlt::get_sstream() >> ship_id >> x >> y >> halite >> hp;

    return std::make_shared<hlt::Ship>(player_id, ship_id, x, y, halite, hp);
}
