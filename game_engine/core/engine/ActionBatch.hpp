#ifndef ACTIONBATCH_HPP
#define ACTIONBATCH_HPP

#include <memory>
#include <vector>

#include "Command.hpp"
#include "Player.hpp"

namespace hlt {

using PlayerCommandList = std::vector<std::unique_ptr<Command>>;
using ActionBatch = ordered_id_map<Player, PlayerCommandList>;

} // namespace hlt

#endif // ACTIONBATCH_HPP
