#ifndef OBSERVATION_HPP
#define OBSERVATION_HPP

#include <string>
#include <vector>

#include "Location.hpp"
#include "Player.hpp"
#include "GameConfig.hpp"

namespace hlt::protocol {

struct PlayerInitInfo {
    hlt::Player::id_type id = hlt::Player::None;
    hlt::Location factory{0, 0};
};

struct InitObservation {
    hlt::GameConfig config;
    hlt::Player::id_type player_id = hlt::Player::None;
    std::vector<PlayerInitInfo> players;
    std::string serialized_map;
};

struct ChangedCellInfo {
    hlt::Location location{0, 0};
    hlt::energy_type energy{};
};

struct TurnObservation {
    unsigned long turn_number{};
    hlt::Player::id_type player_id = hlt::Player::None;
    std::string serialized_players;
    std::vector<ChangedCellInfo> changed_cells;
};

} // namespace hlt::protocol

#endif // OBSERVATION_HPP
