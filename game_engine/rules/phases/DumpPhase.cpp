#include "DumpPhase.hpp"

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

void DumpPhase::execute(RuleContext &context) const {
    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    const int max_hp = context.config().ruleset.combat.initial_hp;

    for (auto &[entity_id, entity] : store.all_entities()) {
        (void)entity_id;
        const auto &player = store.get_player(entity.owner);
        const auto &location = player.get_entity_location(entity.id);
        auto &cell = map.at(location);
        if (cell.owner == entity.owner) {
            const auto deposited = entity.energy;
            dump_energy(store, entity, location, cell, deposited);
            entity.hp = max_hp;
            context.result().changed_cells.emplace(location);
            context.result().changed_entities.emplace(entity.id);
            context.event_sink().emit(events::DepositedEvent{entity.owner, location, deposited});
        }
    }
}

} // namespace hlt::rules::phases
