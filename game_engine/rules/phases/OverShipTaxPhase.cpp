#include "OverShipTaxPhase.hpp"

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

// Per-turn fleet-economy adjustment:
//   - SHIP_INCOME_PER_TURN  : +K credited per ship to factory_halite (score).
//   - OVER_SHIP_TAX_PER_TURN: -T per ship above OVER_SHIP_TAX_THRESHOLD,
//                              deducted from player.energy AND factory_halite.
// With T > K, OVER_SHIP_TAX_THRESHOLD becomes the per-player ship-count
// optimum: every ship up to the threshold contributes net +K, every ship
// above contributes net (K - T) < 0.
void OverShipTaxPhase::execute(RuleContext &context) const {
    const auto &economy = context.config().ruleset.economy;
    const bool tax_on     = economy.over_ship_tax_per_turn       > 0;
    const bool income_on  = economy.ship_income_per_turn         > 0;
    const bool quad_on    = economy.ship_count_deviation_penalty > 0;
    if (!tax_on && !income_on && !quad_on) return;

    auto &store = context.state().store_ref();
    for (auto &[player_id, player] : store.players_ref()) {
        (void)player_id;
        const auto n = player.entities.size();

        // Per-ship income: credit factory_halite (the score pool) only;
        // leave player.energy untouched so spawn affordability is unchanged.
        if (income_on && n > 0 && !player.factory_destroyed) {
            const energy_type income =
                static_cast<energy_type>(n) * economy.ship_income_per_turn;
            player.factory_halite += income;
        }

        // Over-ship tax on excess ships (linear, asymmetric).
        if (tax_on && n > economy.over_ship_tax_threshold) {
            const auto excess = n - economy.over_ship_tax_threshold;
            const energy_type tax =
                static_cast<energy_type>(excess) * economy.over_ship_tax_per_turn;
            player.energy = std::max<energy_type>(0, player.energy - tax);
            if (!player.factory_destroyed) {
                // Clamp at 1 so we don't drive structure_score below the
                // game_ended threshold and terminate the game early.
                player.factory_halite = std::max<energy_type>(1, player.factory_halite - tax);
            }
        }

        // Quadratic over-target penalty: penalty = K × max(0, n - target)² per
        // turn, deducted from both energy and factory_halite.  Asymmetric: no
        // penalty when ramping up below target (n < target → 0), but cost
        // grows quadratically in excess once above target.  At equal ship-count
        // plateau, marginal cost of the (n+1)-th ship is K × (2(n-target)+1).
        if (quad_on) {
            const long long target = static_cast<long long>(economy.ship_count_target);
            const long long actual = static_cast<long long>(n);
            if (actual > target) {
                const long long excess = actual - target;
                const long long penalty_ll =
                    static_cast<long long>(economy.ship_count_deviation_penalty) * excess * excess;
                const energy_type penalty = static_cast<energy_type>(penalty_ll);
                if (penalty > 0) {
                    player.energy = std::max<energy_type>(0, player.energy - penalty);
                    if (!player.factory_destroyed) {
                        player.factory_halite = std::max<energy_type>(1, player.factory_halite - penalty);
                    }
                }
            }
        }
    }
}

} // namespace hlt::rules::phases
