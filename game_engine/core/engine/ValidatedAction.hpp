#ifndef VALIDATEDACTION_HPP
#define VALIDATEDACTION_HPP

#include "Command.hpp"
#include "Player.hpp"

namespace hlt {

struct ValidatedAction {
    Player::id_type player_id = Player::None;
    const Command *command{};
};

} // namespace hlt

#endif // VALIDATEDACTION_HPP
