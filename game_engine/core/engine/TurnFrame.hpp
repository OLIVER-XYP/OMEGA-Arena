#ifndef TURNFRAME_HPP
#define TURNFRAME_HPP

#include <vector>

#include "StepResult.hpp"

namespace hlt {

struct TurnFrame {
    std::vector<Entity::id_type> changed_entities;
};

TurnFrame capture_turn_frame(const StepResult &result);
bool turn_frame_contains_changed_entity(const TurnFrame &frame, Entity::id_type entity_id);

} // namespace hlt

#endif // TURNFRAME_HPP
