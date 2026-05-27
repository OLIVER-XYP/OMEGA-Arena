#include "DefendPhase.hpp"

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

void DefendPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    for (auto &[entity_id, entity] : store.all_entities()) {
        (void)entity_id;
        // Emergency-spawned ships carry multi-turn protection: decrement the
        // counter each turn and keep is_defending=true while it lasts.
        if (entity.protection_turns > 0) {
            entity.protection_turns--;
            entity.is_defending = true;
        } else {
            entity.is_defending = false;
        }
    }

    for (const auto &[player_id, defends] : result.validated_commands->defends) {
        const auto &player = store.get_player(player_id);
        for (const DefendCommand &command : defends) {
            if (!player.has_entity(command.entity)) {
                append_error(result, std::make_unique<EntityNotFoundError<DefendCommand>>(player_id, command));
                continue;
            }
            auto &entity = store.get_entity(command.entity);
            entity.is_defending = true;
            result.changed_entities.emplace(command.entity);
        }
    }
}

} // namespace hlt::rules::phases
