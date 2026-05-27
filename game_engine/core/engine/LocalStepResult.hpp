#ifndef LOCALSTEPRESULT_HPP
#define LOCALSTEPRESULT_HPP

#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

#include "Entity.hpp"
#include "Location.hpp"
#include "Player.hpp"
#include "StepResult.hpp"

namespace hlt {

template <typename TargetRange, typename SourceRange>
inline void append_move_range(TargetRange &target, SourceRange &&source) {
    target.insert(target.end(),
                  std::make_move_iterator(source.begin()),
                  std::make_move_iterator(source.end()));
}

struct LocalStepResult {
    std::vector<std::string> non_fatal_errors;
    std::vector<Player::id_type> eliminated_players;
    std::unordered_set<Entity::id_type> changed_entities;
    std::unordered_set<Location> changed_cells;
};

inline void merge_into(StepResult &global, LocalStepResult &&local) {
    append_move_range(global.non_fatal_errors, std::move(local.non_fatal_errors));
    append_move_range(global.eliminated_players, std::move(local.eliminated_players));
    global.changed_entities.insert(local.changed_entities.begin(), local.changed_entities.end());
    global.changed_cells.insert(local.changed_cells.begin(), local.changed_cells.end());
}

} // namespace hlt

#endif // LOCALSTEPRESULT_HPP
