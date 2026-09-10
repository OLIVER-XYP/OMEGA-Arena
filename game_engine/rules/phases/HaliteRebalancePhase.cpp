#include "HaliteRebalancePhase.hpp"

#include "PhaseHelpers.hpp"

namespace hlt::rules::phases {

// Rubber-band rebalancing.  Every PERIOD turns, find the players with the
// highest and lowest halite banks and transfer FRACTION × (max - min) / 2
// halite from the leader to the trailer.  This caps the score gap and keeps
// self-play tightly competitive at the cost of suppressing pure economic skill.
void HaliteRebalancePhase::execute(RuleContext &context) const {
    const auto &economy = context.config().ruleset.economy;
    if (!economy.halite_rebalance_enabled
            || economy.halite_rebalance_period == 0
            || economy.halite_rebalance_fraction <= 0.0) {
        return;
    }
    const auto turn_no = context.state().turn.number;
    if (turn_no == 0 || turn_no % economy.halite_rebalance_period != 0) {
        return;
    }

    auto &store = context.state().store_ref();

    // Compute each player's score-relevant total: factory_halite + dropoff pools.
    // This is what `structure_score` in TurnStatsCollector reports — i.e., the
    // halite that determines the final published score.  Rebalancing this is the
    // only way to actually move the observed lo/hi ratio.
    auto structure_total = [](const Player &p) {
        energy_type t = p.factory_halite;
        for (const auto &d : p.dropoffs) t += d.halite_pool;
        return t;
    };

    Player::id_type leader_id{};
    Player::id_type trailer_id{};
    energy_type max_score = std::numeric_limits<energy_type>::min();
    energy_type min_score = std::numeric_limits<energy_type>::max();
    bool seen_any = false;

    for (auto &[player_id, player] : store.players_ref()) {
        const auto s = structure_total(player);
        if (!seen_any || s > max_score) { max_score = s; leader_id = player_id; }
        if (!seen_any || s < min_score) { min_score = s; trailer_id = player_id; }
        seen_any = true;
    }

    if (!seen_any || leader_id == trailer_id) return;
    if (max_score <= min_score) return;

    // Skip naturally-tight games to avoid overshooting into near-ties.
    if (economy.halite_rebalance_min_gap_frac > 0.0
            && max_score + min_score > 0) {
        const double gap_frac = static_cast<double>(max_score - min_score)
                              / static_cast<double>(max_score + min_score);
        if (gap_frac < economy.halite_rebalance_min_gap_frac) return;
    }

    const auto gap = max_score - min_score;
    auto transfer = static_cast<energy_type>(
        static_cast<double>(gap) * economy.halite_rebalance_fraction * 0.5);
    if (transfer <= 0) return;

    auto &leader = store.get_player(leader_id);
    auto &trailer = store.get_player(trailer_id);

    // Drain from leader's structures (factory first, then dropoffs).  Skip any
    // that have been destroyed in combat.
    auto drain = [&](energy_type &pool, energy_type &take) {
        const energy_type d = std::min(pool, take);
        pool -= d;
        take -= d;
    };
    energy_type remaining = transfer;
    if (!leader.factory_destroyed) drain(leader.factory_halite, remaining);
    for (auto &d : leader.dropoffs) {
        if (remaining <= 0) break;
        if (d.destroyed) continue;
        drain(d.halite_pool, remaining);
    }
    const energy_type actually_drained = transfer - remaining;
    if (actually_drained <= 0) return;

    // Credit to trailer's factory (cheapest landing spot — always present).
    if (!trailer.factory_destroyed) {
        trailer.factory_halite += actually_drained;
    } else if (!trailer.dropoffs.empty()) {
        // Fallback: trailer's factory destroyed, credit to first live dropoff.
        for (auto &d : trailer.dropoffs) {
            if (!d.destroyed) { d.halite_pool += actually_drained; break; }
        }
    }

    // Keep player.energy bank consistent with the structure score so the bot's
    // spawn/dropoff affordability sees the same picture the score does.
    leader.energy = std::max<energy_type>(0, leader.energy - actually_drained);
    trailer.energy += actually_drained;
}

} // namespace hlt::rules::phases
