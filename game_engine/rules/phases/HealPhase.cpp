#include "HealPhase.hpp"

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

void HealPhase::execute(RuleContext &context) const {
    auto &result = context.result();
    if (!result.validated_commands.has_value()) {
        return;
    }

    auto &store = context.state().store_ref();
    for (const auto &[player_id, heals] : result.validated_commands->heals) {
        const auto &player = store.get_player(player_id);
        for (const HealCommand &command : heals) {
            if (!player.has_entity(command.entity)) {
                append_error(result, std::make_unique<EntityNotFoundError<HealCommand>>(player_id, command));
            }
        }
    }
}

} // namespace hlt::rules::phases
