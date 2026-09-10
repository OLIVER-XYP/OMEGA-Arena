#include "TurnFrame.hpp"

#include <algorithm>

namespace hlt {

TurnFrame capture_turn_frame(const StepResult &result) {
    TurnFrame frame;
    frame.changed_entities.reserve(result.changed_entities.size());
    for (const auto entity_id : result.changed_entities) {
        frame.changed_entities.push_back(entity_id);
    }
    std::sort(frame.changed_entities.begin(), frame.changed_entities.end());
    return frame;
}

bool turn_frame_contains_changed_entity(const TurnFrame &frame, Entity::id_type entity_id) {
    return std::binary_search(frame.changed_entities.begin(), frame.changed_entities.end(), entity_id);
}

} // namespace hlt
