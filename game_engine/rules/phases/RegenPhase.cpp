#include "RegenPhase.hpp"

#include <algorithm>
#include <cmath>

namespace hlt::rules::phases {

// Halite regeneration ("territory control").  Runs after MiningPhase: each
// non-structure cell regrows toward CELL_REGEN_CAP_FRACTION × initial_energy by
// ceil(CELL_REGEN_RATE × initial_energy) per turn.  Regrowth is relative to each
// cell's original (post-generation) value, so rich zones replenish and barren
// cells stay barren.  Holding a rich zone becomes a durable economic advantage,
// and a trailing player's map also recovers, dampening snowball.
void RegenPhase::execute(RuleContext &context) const {
    const auto &economy = context.config().ruleset.economy;
    if (!economy.cell_regen_enabled
            || economy.cell_regen_rate <= 0.0
            || economy.cell_regen_cap_fraction <= 0.0) {
        return;
    }

    auto &store = context.state().store_ref();
    auto &map = context.state().map_ref();
    auto &result = context.result();

    for (dimension_type y = 0; y < map.height; ++y) {
        for (dimension_type x = 0; x < map.width; ++x) {
            auto &cell = map.at(x, y);
            // Structure tiles (factory/dropoff) never hold map halite.
            if (cell.owner != Player::None) {
                continue;
            }
            if (cell.initial_energy <= 0) {
                continue;
            }

            const auto cap = static_cast<energy_type>(
                economy.cell_regen_cap_fraction * static_cast<double>(cell.initial_energy));
            if (cell.energy >= cap) {
                continue;
            }

            const auto regen = static_cast<energy_type>(
                std::ceil(economy.cell_regen_rate * static_cast<double>(cell.initial_energy)));
            if (regen <= 0) {
                continue;
            }

            const auto new_energy = std::min(cap, cell.energy + regen);
            const auto delta = new_energy - cell.energy;
            if (delta <= 0) {
                continue;
            }

            cell.energy = new_energy;
            store.map_total_energy += static_cast<unsigned long long>(delta);
            result.changed_cells.emplace(Location{x, y});
        }
    }
}

} // namespace hlt::rules::phases
