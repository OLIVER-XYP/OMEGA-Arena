#ifndef GAMECONFIG_HPP
#define GAMECONFIG_HPP

#include "Constants.hpp"

namespace hlt {

struct EconomyConfig {
    energy_type max_cell_production{};
    energy_type min_cell_production{};
    energy_type max_energy{};
    energy_type new_entity_energy_cost{};
    energy_type initial_energy{};
    energy_type dropoff_cost{};
    energy_type initial_dropoff_halite{};
    energy_type initial_factory_halite{};
    unsigned long move_cost_ratio{};
    unsigned long dropoff_penalty_ratio{};
    unsigned long extract_ratio{};
    // Snowball-dampening growth factors: effective cost = base × (1 + growth × N_existing).
    double spawn_cost_growth{};
    double dropoff_cost_growth{};
    // Comeback "rescue package" for eliminated players: every period turns
    // spawn N protected ships and credit a halite bonus to the player.
    bool emergency_spawn_enabled{};
    unsigned long emergency_spawn_period{};
    unsigned long emergency_spawn_count{};
    energy_type   emergency_halite_bonus{};
    unsigned long emergency_protection_turns{};
    // Rubber-band halite rebalance: caps the score gap.
    bool halite_rebalance_enabled{};
    unsigned long halite_rebalance_period{};
    double halite_rebalance_fraction{};
    double halite_rebalance_min_gap_frac{};
    // Over-ship tax + per-ship income: jointly create a per-player ship-count optimum.
    unsigned long over_ship_tax_threshold{};
    energy_type   over_ship_tax_per_turn{};
    energy_type   ship_income_per_turn{};
    // Quadratic deviation penalty around ship_count_target: symmetric optimum at target.
    unsigned long ship_count_target{};
    energy_type   ship_count_deviation_penalty{};
    // Quadratic over-threshold spawn-cost component.
    unsigned long spawn_quad_threshold{};
    double        spawn_quad_growth{};
    // Mining interference: enemy ships within range reduce mining yield.
    int    mining_interference_range{};
    double mining_interference_ratio{};
    double mining_interference_cargo_loss_ratio{};
    // Plunder: ships near enemy structures gain passive halite income.
    int    plunder_range{};
    energy_type plunder_halite_per_turn{};
    // Halite regeneration: cells regrow toward a fraction of their initial energy.
    bool   cell_regen_enabled{};
    double cell_regen_rate{};
    double cell_regen_cap_fraction{};
};

struct CombatConfig {
    int initial_hp{};
    int attack_hp_damage{};
    int attack_hp_self_damage{};
    double attack_halite_steal_ratio{};
    int heal_amount{};
    int defend_retaliation_damage{};
    bool defend_allows_mining{};
    bool enable_combat_commands{};
    int collision_hp_damage{};
    bool enable_attacker_self_damage{};
    bool kill_credit_to_attacker{};
    double kill_halite_bonus_ratio{};
    int attack_range{};
};

struct InspirationConfig {
    bool enabled{};
    unsigned long extract_ratio{};
    double bonus_multiplier{};
    unsigned long move_cost_ratio{};
    dimension_type radius{};
    unsigned long ship_count{};
};

struct CaptureConfig {
    bool enabled{};
    dimension_type radius{};
    unsigned long ships_above_for_capture{};
};

struct MapGenConfig {
    dimension_type default_width{};
    dimension_type default_height{};
    double persistence{};
    double factor_exp_1{};
    double factor_exp_2{};
};

struct MatchConfig {
    bool strict_errors{};
    unsigned long max_players{};
    unsigned long min_turns{};
    unsigned long min_turn_threshold{};
    unsigned long max_turns{};
    unsigned long max_turn_threshold{};
};

struct RulesetConfig {
    std::string version;
    EconomyConfig economy;
    CombatConfig combat;
    InspirationConfig inspiration;
    CaptureConfig capture;
};

struct GameConfig {
    MatchConfig match;
    MapGenConfig mapgen;
    RulesetConfig ruleset;

    static GameConfig from_constants(const Constants &constants = Constants::get());
    void apply_to_global_constants() const;
};

} // namespace hlt

#endif // GAMECONFIG_HPP
