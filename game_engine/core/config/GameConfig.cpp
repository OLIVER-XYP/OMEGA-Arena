#include "GameConfig.hpp"

namespace hlt {

GameConfig GameConfig::from_constants(const Constants &constants) {
    GameConfig config{};
    config.match.strict_errors = constants.STRICT_ERRORS;
    config.match.max_players = constants.MAX_PLAYERS;
    config.match.min_turns = constants.MIN_TURNS;
    config.match.min_turn_threshold = constants.MIN_TURN_THRESHOLD;
    config.match.max_turns = constants.MAX_TURNS;
    config.match.max_turn_threshold = constants.MAX_TURN_THRESHOLD;

    config.mapgen.default_width = constants.DEFAULT_MAP_WIDTH;
    config.mapgen.default_height = constants.DEFAULT_MAP_HEIGHT;
    config.mapgen.persistence = constants.PERSISTENCE;
    config.mapgen.factor_exp_1 = constants.FACTOR_EXP_1;
    config.mapgen.factor_exp_2 = constants.FACTOR_EXP_2;

    config.ruleset.version = constants.RULESET_VERSION;
    config.ruleset.economy.max_cell_production = constants.MAX_CELL_PRODUCTION;
    config.ruleset.economy.min_cell_production = constants.MIN_CELL_PRODUCTION;
    config.ruleset.economy.max_energy = constants.MAX_ENERGY;
    config.ruleset.economy.new_entity_energy_cost = constants.NEW_ENTITY_ENERGY_COST;
    config.ruleset.economy.initial_energy = constants.INITIAL_ENERGY;
    config.ruleset.economy.dropoff_cost = constants.DROPOFF_COST;
    config.ruleset.economy.initial_dropoff_halite = constants.INITIAL_DROPOFF_HALITE;
    config.ruleset.economy.initial_factory_halite = constants.INITIAL_FACTORY_HALITE;
    config.ruleset.economy.move_cost_ratio = constants.MOVE_COST_RATIO;
    config.ruleset.economy.dropoff_penalty_ratio = constants.DROPOFF_PENALTY_RATIO;
    config.ruleset.economy.extract_ratio = constants.EXTRACT_RATIO;
    config.ruleset.economy.spawn_cost_growth = constants.SPAWN_COST_GROWTH;
    config.ruleset.economy.dropoff_cost_growth = constants.DROPOFF_COST_GROWTH;
    config.ruleset.economy.emergency_spawn_enabled = constants.EMERGENCY_SPAWN_ENABLED;
    config.ruleset.economy.emergency_spawn_period = constants.EMERGENCY_SPAWN_PERIOD;
    config.ruleset.economy.emergency_spawn_count = constants.EMERGENCY_SPAWN_COUNT;
    config.ruleset.economy.emergency_halite_bonus = constants.EMERGENCY_HALITE_BONUS;
    config.ruleset.economy.emergency_protection_turns = constants.EMERGENCY_PROTECTION_TURNS;
    config.ruleset.economy.halite_rebalance_enabled = constants.HALITE_REBALANCE_ENABLED;
    config.ruleset.economy.halite_rebalance_period = constants.HALITE_REBALANCE_PERIOD;
    config.ruleset.economy.halite_rebalance_fraction = constants.HALITE_REBALANCE_FRACTION;
    config.ruleset.economy.halite_rebalance_min_gap_frac = constants.HALITE_REBALANCE_MIN_GAP_FRAC;
    config.ruleset.economy.over_ship_tax_threshold = constants.OVER_SHIP_TAX_THRESHOLD;
    config.ruleset.economy.over_ship_tax_per_turn = constants.OVER_SHIP_TAX_PER_TURN;
    config.ruleset.economy.ship_income_per_turn = constants.SHIP_INCOME_PER_TURN;
    config.ruleset.economy.ship_count_target = constants.SHIP_COUNT_TARGET;
    config.ruleset.economy.ship_count_deviation_penalty = constants.SHIP_COUNT_DEVIATION_PENALTY;
    config.ruleset.economy.spawn_quad_threshold = constants.SPAWN_QUAD_THRESHOLD;
    config.ruleset.economy.spawn_quad_growth   = constants.SPAWN_QUAD_GROWTH;
    config.ruleset.economy.mining_interference_range = constants.MINING_INTERFERENCE_RANGE;
    config.ruleset.economy.mining_interference_ratio = constants.MINING_INTERFERENCE_RATIO;
    config.ruleset.economy.mining_interference_cargo_loss_ratio = constants.MINING_INTERFERENCE_CARGO_LOSS_RATIO;
    config.ruleset.economy.plunder_range = constants.PLUNDER_RANGE;
    config.ruleset.economy.plunder_halite_per_turn = constants.PLUNDER_HALITE_PER_TURN;
    config.ruleset.economy.cell_regen_enabled = constants.CELL_REGEN_ENABLED;
    config.ruleset.economy.cell_regen_rate = constants.CELL_REGEN_RATE;
    config.ruleset.economy.cell_regen_cap_fraction = constants.CELL_REGEN_CAP_FRACTION;

    config.ruleset.combat.initial_hp = constants.INITIAL_HP;
    config.ruleset.combat.attack_hp_damage = constants.ATTACK_HP_DAMAGE;
    config.ruleset.combat.attack_hp_self_damage = constants.ATTACK_HP_SELF_DAMAGE;
    config.ruleset.combat.attack_halite_steal_ratio = constants.ATTACK_HALITE_STEAL_RATIO;
    config.ruleset.combat.heal_amount = constants.HEAL_AMOUNT;
    config.ruleset.combat.defend_retaliation_damage = constants.DEFEND_RETALIATION_DAMAGE;
    config.ruleset.combat.defend_allows_mining = constants.DEFEND_ALLOWS_MINING;
    config.ruleset.combat.enable_combat_commands = constants.ENABLE_COMBAT_COMMANDS;
    config.ruleset.combat.collision_hp_damage = constants.COLLISION_HP_DAMAGE;
    config.ruleset.combat.enable_attacker_self_damage = constants.ENABLE_ATTACKER_SELF_DAMAGE;
    config.ruleset.combat.kill_credit_to_attacker = constants.KILL_CREDIT_TO_ATTACKER;
    config.ruleset.combat.kill_halite_bonus_ratio = constants.KILL_HALITE_BONUS_RATIO;
    config.ruleset.combat.attack_range = constants.ATTACK_RANGE;

    config.ruleset.inspiration.enabled = constants.INSPIRATION_ENABLED;
    config.ruleset.inspiration.extract_ratio = constants.INSPIRED_EXTRACT_RATIO;
    config.ruleset.inspiration.bonus_multiplier = constants.INSPIRED_BONUS_MULTIPLIER;
    config.ruleset.inspiration.move_cost_ratio = constants.INSPIRED_MOVE_COST_RATIO;
    config.ruleset.inspiration.radius = constants.INSPIRATION_RADIUS;
    config.ruleset.inspiration.ship_count = constants.INSPIRATION_SHIP_COUNT;

    config.ruleset.capture.enabled = constants.CAPTURE_ENABLED;
    config.ruleset.capture.radius = constants.CAPTURE_RADIUS;
    config.ruleset.capture.ships_above_for_capture = constants.SHIPS_ABOVE_FOR_CAPTURE;
    return config;
}

void GameConfig::apply_to_global_constants() const {
    auto &constants = Constants::get_mut();
    constants.STRICT_ERRORS = match.strict_errors;
    constants.MAX_PLAYERS = match.max_players;
    constants.MIN_TURNS = match.min_turns;
    constants.MIN_TURN_THRESHOLD = match.min_turn_threshold;
    constants.MAX_TURNS = match.max_turns;
    constants.MAX_TURN_THRESHOLD = match.max_turn_threshold;

    constants.DEFAULT_MAP_WIDTH = mapgen.default_width;
    constants.DEFAULT_MAP_HEIGHT = mapgen.default_height;
    constants.PERSISTENCE = mapgen.persistence;
    constants.FACTOR_EXP_1 = mapgen.factor_exp_1;
    constants.FACTOR_EXP_2 = mapgen.factor_exp_2;

    constants.RULESET_VERSION = ruleset.version;
    constants.MAX_CELL_PRODUCTION = ruleset.economy.max_cell_production;
    constants.MIN_CELL_PRODUCTION = ruleset.economy.min_cell_production;
    constants.MAX_ENERGY = ruleset.economy.max_energy;
    constants.NEW_ENTITY_ENERGY_COST = ruleset.economy.new_entity_energy_cost;
    constants.INITIAL_ENERGY = ruleset.economy.initial_energy;
    constants.DROPOFF_COST = ruleset.economy.dropoff_cost;
    constants.INITIAL_DROPOFF_HALITE = ruleset.economy.initial_dropoff_halite;
    constants.INITIAL_FACTORY_HALITE = ruleset.economy.initial_factory_halite;
    constants.MOVE_COST_RATIO = ruleset.economy.move_cost_ratio;
    constants.DROPOFF_PENALTY_RATIO = ruleset.economy.dropoff_penalty_ratio;
    constants.EXTRACT_RATIO = ruleset.economy.extract_ratio;
    constants.SPAWN_COST_GROWTH = ruleset.economy.spawn_cost_growth;
    constants.DROPOFF_COST_GROWTH = ruleset.economy.dropoff_cost_growth;
    constants.EMERGENCY_SPAWN_ENABLED = ruleset.economy.emergency_spawn_enabled;
    constants.EMERGENCY_SPAWN_PERIOD = ruleset.economy.emergency_spawn_period;
    constants.EMERGENCY_SPAWN_COUNT = ruleset.economy.emergency_spawn_count;
    constants.EMERGENCY_HALITE_BONUS = ruleset.economy.emergency_halite_bonus;
    constants.EMERGENCY_PROTECTION_TURNS = ruleset.economy.emergency_protection_turns;
    constants.HALITE_REBALANCE_ENABLED = ruleset.economy.halite_rebalance_enabled;
    constants.HALITE_REBALANCE_PERIOD = ruleset.economy.halite_rebalance_period;
    constants.HALITE_REBALANCE_FRACTION = ruleset.economy.halite_rebalance_fraction;
    constants.HALITE_REBALANCE_MIN_GAP_FRAC = ruleset.economy.halite_rebalance_min_gap_frac;
    constants.OVER_SHIP_TAX_THRESHOLD = ruleset.economy.over_ship_tax_threshold;
    constants.OVER_SHIP_TAX_PER_TURN = ruleset.economy.over_ship_tax_per_turn;
    constants.SHIP_INCOME_PER_TURN = ruleset.economy.ship_income_per_turn;
    constants.SHIP_COUNT_TARGET = ruleset.economy.ship_count_target;
    constants.SHIP_COUNT_DEVIATION_PENALTY = ruleset.economy.ship_count_deviation_penalty;
    constants.SPAWN_QUAD_THRESHOLD = ruleset.economy.spawn_quad_threshold;
    constants.SPAWN_QUAD_GROWTH   = ruleset.economy.spawn_quad_growth;
    constants.MINING_INTERFERENCE_RANGE = ruleset.economy.mining_interference_range;
    constants.MINING_INTERFERENCE_RATIO = ruleset.economy.mining_interference_ratio;
    constants.MINING_INTERFERENCE_CARGO_LOSS_RATIO = ruleset.economy.mining_interference_cargo_loss_ratio;
    constants.PLUNDER_RANGE = ruleset.economy.plunder_range;
    constants.PLUNDER_HALITE_PER_TURN = ruleset.economy.plunder_halite_per_turn;
    constants.CELL_REGEN_ENABLED = ruleset.economy.cell_regen_enabled;
    constants.CELL_REGEN_RATE = ruleset.economy.cell_regen_rate;
    constants.CELL_REGEN_CAP_FRACTION = ruleset.economy.cell_regen_cap_fraction;

    constants.INITIAL_HP = ruleset.combat.initial_hp;
    constants.ATTACK_HP_DAMAGE = ruleset.combat.attack_hp_damage;
    constants.ATTACK_HP_SELF_DAMAGE = ruleset.combat.attack_hp_self_damage;
    constants.ATTACK_HALITE_STEAL_RATIO = ruleset.combat.attack_halite_steal_ratio;
    constants.HEAL_AMOUNT = ruleset.combat.heal_amount;
    constants.DEFEND_RETALIATION_DAMAGE = ruleset.combat.defend_retaliation_damage;
    constants.DEFEND_ALLOWS_MINING = ruleset.combat.defend_allows_mining;
    constants.ENABLE_COMBAT_COMMANDS = ruleset.combat.enable_combat_commands;
    constants.COLLISION_HP_DAMAGE = ruleset.combat.collision_hp_damage;
    constants.ENABLE_ATTACKER_SELF_DAMAGE = ruleset.combat.enable_attacker_self_damage;
    constants.KILL_CREDIT_TO_ATTACKER = ruleset.combat.kill_credit_to_attacker;
    constants.KILL_HALITE_BONUS_RATIO = ruleset.combat.kill_halite_bonus_ratio;
    constants.ATTACK_RANGE = ruleset.combat.attack_range;

    constants.INSPIRATION_ENABLED = ruleset.inspiration.enabled;
    constants.INSPIRED_EXTRACT_RATIO = ruleset.inspiration.extract_ratio;
    constants.INSPIRED_BONUS_MULTIPLIER = ruleset.inspiration.bonus_multiplier;
    constants.INSPIRED_MOVE_COST_RATIO = ruleset.inspiration.move_cost_ratio;
    constants.INSPIRATION_RADIUS = ruleset.inspiration.radius;
    constants.INSPIRATION_SHIP_COUNT = ruleset.inspiration.ship_count;

    constants.CAPTURE_ENABLED = ruleset.capture.enabled;
    constants.CAPTURE_RADIUS = ruleset.capture.radius;
    constants.SHIPS_ABOVE_FOR_CAPTURE = ruleset.capture.ships_above_for_capture;
}

} // namespace hlt
