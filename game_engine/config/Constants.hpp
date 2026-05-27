#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include "Units.hpp"

#include <string>

#include "nlohmann/json_fwd.hpp"

int main(int argc, char *argv[]);

namespace hlt {

/**
 * Gameplay constants that may be tweaked, though they should be at their
 * default values in a tournament setting.
 *
 * Supports serialization to and deserialization from JSON.
 */
struct Constants {
    friend int ::main(int argc, char *argv[]);

    bool STRICT_ERRORS = false;                 /**< Whether strict error checking mode is enabled. */

    unsigned long MAX_PLAYERS = 16;             /**< The maximum number of players. */
    dimension_type DEFAULT_MAP_WIDTH = 32;      /**< The default width of generated maps. */
    dimension_type DEFAULT_MAP_HEIGHT = 32;     /**< The default height of generated maps. */

    energy_type MAX_CELL_PRODUCTION = 1000;     /**< The maximum maximum amount of production on a cell. */
    energy_type MIN_CELL_PRODUCTION = 900;      /**< The minimum maximum amount of production on a cell. */
    energy_type MAX_ENERGY = 1000;              /**< The maximum amount of energy per entity. */
    energy_type NEW_ENTITY_ENERGY_COST = 1000;  /**< The base cost of a new entity. */
    energy_type INITIAL_ENERGY = 5000;          /**< The initial amount of energy for a player. */

    energy_type DROPOFF_COST = 4000;            /**< The cost of a dropoff construction. */
    energy_type INITIAL_DROPOFF_HALITE = 4000;  /**< Initial halite pool for each dropoff. */
    energy_type INITIAL_FACTORY_HALITE = 5000;  /**< Initial halite pool for each shipyard. */
    unsigned long MOVE_COST_RATIO = 10;         /**< The cost of a move is the source's energy divided by this. */
    unsigned long DROPOFF_PENALTY_RATIO = 4;    /**< The cost ratio of using another player's dropoff. */
    unsigned long EXTRACT_RATIO = 4;            /**< The ratio of energy extracted from a cell per turn. */

    double PERSISTENCE = 0.7; // Determines relative weight of local vs global features.
    double FACTOR_EXP_1 = 2; // Determines initial spikiness of map. Higher values weight towards 0.
    double FACTOR_EXP_2 = 2; // Determines final spikiness of map. Higher values weight towards 0.

    unsigned long MIN_TURNS = 1000;
    unsigned long MIN_TURN_THRESHOLD = 32;
    unsigned long MAX_TURNS = 1000;
    unsigned long MAX_TURN_THRESHOLD = 64;

    /** Combat **/
    int INITIAL_HP = 100;                       /**< Starting HP for each ship. */
    int ATTACK_HP_DAMAGE = 70;                  /**< HP damage dealt to target on hit. */
    int ATTACK_HP_SELF_DAMAGE = 20;             /**< HP damage to attacker on hit (applied only if ENABLE_ATTACKER_SELF_DAMAGE). */
    double ATTACK_HALITE_STEAL_RATIO = 0.60;    /**< Fraction of target halite stolen on hit. */
    int HEAL_AMOUNT = 0;                        /**< Unused: ships now heal only at base. */
    int DEFEND_RETALIATION_DAMAGE = 0;          /**< HP damage reflected back to an attacker who hits a defending (is_defending) ship. Gives defense a structural counter to aggression. Default 0 = pure immunity. */
    bool DEFEND_ALLOWS_MINING = false;          /**< If true, a defending (is_defending) ship still mines its cell that turn, so defending has no economy cost. Removes the "pin/freeze" exploit that otherwise cripples a defensive strategy. */

    /** Feature toggles / ruleset metadata */
    bool ENABLE_COMBAT_COMMANDS = true;         /**< Whether a/x/d/h combat commands are accepted by parser/runtime. */
    int COLLISION_HP_DAMAGE = 50;               /**< HP damage each ship takes when colliding into the same destination cell. */
    std::string RULESET_VERSION = "halite3-hp-collision-v4"; /**< Unified ruleset identifier shown in constants/logs. */

    /** Attack positive-feedback switches */
    bool ENABLE_ATTACKER_SELF_DAMAGE = true;    /**< Apply ATTACK_HP_SELF_DAMAGE to attacker each hit. */
    bool KILL_CREDIT_TO_ATTACKER    = true;     /**< Remaining cargo of killed ship goes to killer's bank instead of the map. */
    double KILL_HALITE_BONUS_RATIO  = 0.0;      /**< Extra fraction of killed cargo added on top (e.g. 0.5 = 50% bonus). */

    /** Economic diminishing returns (dampens snowball positive feedback).
     *  Tuned 2026-05-11 on 30-seed self-play: stable% 6.7→30, meanRatio 0.30→0.56. */
    double SPAWN_COST_GROWTH        = 0.03;     /**< Per-ship spawn cost growth: cost = base × (1 + growth × N_ships). */
    double DROPOFF_COST_GROWTH      = 0.25;     /**< Per-dropoff cost growth: cost = base × (1 + growth × N_dropoffs). */

    /** Comeback mechanic: eliminated players get a "rescue package" every PERIOD turns.
     *  Each event spawns COUNT free ships (each with PROTECTION_TURNS turns of
     *  is_defending immunity) and credits HALITE_BONUS halite to the player's bank. */
    bool EMERGENCY_SPAWN_ENABLED    = true;
    unsigned long EMERGENCY_SPAWN_PERIOD     = 20;
    unsigned long EMERGENCY_SPAWN_COUNT      = 3;
    energy_type   EMERGENCY_HALITE_BONUS     = 2000;
    unsigned long EMERGENCY_PROTECTION_TURNS = 3;

    /** Halite-rebalance ("rubber band") mechanic: every PERIOD turns transfer
     *  FRACTION × (max-min) / 2 halite from leader's structure-score pool to
     *  trailer's.  MIN_GAP_FRAC suppresses rebalance when gap is already small,
     *  preventing overshoot into near-ties.  Tuned 2026-05-11 on 90-seed self-
     *  play: 100% of games land lo/hi ∈ [0.87, 0.99] (Wilson 95% LB = 95.9%). */
    bool HALITE_REBALANCE_ENABLED  = true;
    unsigned long HALITE_REBALANCE_PERIOD = 25;
    double HALITE_REBALANCE_FRACTION = 0.5;
    double HALITE_REBALANCE_MIN_GAP_FRAC = 0.05;

    /** Fleet economy: per turn, each player receives
     *      ship_count × SHIP_INCOME_PER_TURN
     *  halite credited to factory_halite, and pays
     *      max(0, ship_count - OVER_SHIP_TAX_THRESHOLD) × OVER_SHIP_TAX_PER_TURN
     *  halite, deducted from BOTH player.energy and player.factory_halite.
     *  With SHIP_INCOME > 0 and TAX > SHIP_INCOME, OVER_SHIP_TAX_THRESHOLD is the
     *  per-player ship-count optimum.  Default disabled (all three 0). */
    unsigned long OVER_SHIP_TAX_THRESHOLD = 0;
    energy_type   OVER_SHIP_TAX_PER_TURN  = 0;
    energy_type   SHIP_INCOME_PER_TURN    = 0;

    /** Quadratic deviation penalty: per turn each player pays
     *      SHIP_COUNT_DEVIATION_PENALTY × max(0, ship_count - SHIP_COUNT_TARGET)²
     *  halite, deducted from both player.energy and player.factory_halite.
     *  Asymmetric (no penalty below target).  Default disabled. */
    unsigned long SHIP_COUNT_TARGET           = 12;
    energy_type   SHIP_COUNT_DEVIATION_PENALTY = 0;

    /** Halite regeneration ("territory control").  Every turn each non-structure
     *  cell with initial energy > 0 regrows by ceil(CELL_REGEN_RATE × initial_energy),
     *  capped at CELL_REGEN_CAP_FRACTION × initial_energy.  Converts the extraction
     *  race into a hold-the-rich-zone game and dampens snowball (a trailing player's
     *  map also replenishes).  Regrowth is relative to each cell's original value so
     *  rich zones stay rich and barren cells stay barren.  Default disabled. */
    bool   CELL_REGEN_ENABLED      = false;
    double CELL_REGEN_RATE         = 0.0;   /**< Per-turn regrowth as a fraction of the cell's initial energy. */
    double CELL_REGEN_CAP_FRACTION = 0.5;   /**< Regrowth ceiling as a fraction of the cell's initial energy. */

    /** Quadratic over-threshold spawn-cost component.  Effective spawn cost is
     *      base × (1 + SPAWN_COST_GROWTH × N + SPAWN_QUAD_GROWTH × max(0, N + 1 - SPAWN_QUAD_THRESHOLD)²)
     *  where N is the player's current ship count.  Leaves v4 spawn behaviour
     *  unchanged when N + 1 ≤ threshold (i.e., spawn does not push count
     *  strictly above threshold).  Enables sharp penalisation of spawns past
     *  the threshold without changing the cost of the threshold-th ship.
     *  Default disabled (growth = 0). */
    unsigned long SPAWN_QUAD_THRESHOLD = 12;
    double        SPAWN_QUAD_GROWTH    = 0.0;

    /** Capture */
    bool CAPTURE_ENABLED = false; /**< whether to use capture */
    dimension_type CAPTURE_RADIUS = 3; /**< The distance in which a ship is considered for the capture calculation */
    unsigned long SHIPS_ABOVE_FOR_CAPTURE = 3; /**< If enemyships - friendlyships is above or equal to this threshold,
                                                        the ship is captured*/

    /** Inspiration **/
    bool INSPIRATION_ENABLED = true; /**< whether to use inspiration **/
    unsigned long INSPIRED_EXTRACT_RATIO = EXTRACT_RATIO; /**< alternative mining ratio for inspired ships */
    double INSPIRED_BONUS_MULTIPLIER = 2; /**< The benefit ratio of mining when inspired. (Removing Y halite from a cell gives you X*Y additional halite.) */
    unsigned long INSPIRED_MOVE_COST_RATIO = MOVE_COST_RATIO; /**< Alternative move cost ratio for inspired ships. */
    dimension_type INSPIRATION_RADIUS = 4; /** Maximum distance away for ships to count towards inspiration. */
    unsigned long INSPIRATION_SHIP_COUNT = 2; /**< If there are at least X enemy ships, then you are inspired. */

    /*
    The two FACTOR_EXP constants do related things but they are not the same.
    FACTOR_EXP_1 exponentiates the distribution used to seed the randomness.
    FACTOR_EXP_2 exponentiates the final distribution just prior to normalization.
    Broadly, both will give spikier maps. However (and perhaps counterintuitively), using
    FACTOR_EXP_1 will give maps that have more individual, small-scale spikes. Conversely,
    FACTOR_EXP_2 gives maps that moreso utilize the global structure, and have less noise.
    FACTOR_EXP_2 is also more sensitive than FACTOR_EXP_1.
    */

    static constexpr double BLUR_FACTOR = 0.75; // Not part of canon, needed to compile

    /**
     * Get the singleton constants.
     * @return The singleton constants.
     */
    static const Constants &get() { return get_mut(); }

    /**
     * Get a mutable reference to the singleton constants.
     * @return Mutable reference to the singleton constants.
     */
    static Constants &get_mut() {
        // Guaranteed initialized only once by C++11
        static Constants instance;
        return instance;
    }

    /**
     * Encode the constants to JSON.
     * @param[out] json The JSON output.
     * @param constants The constants.
     */
    friend void to_json(nlohmann::json &json, const Constants &constants);

    /**
     * Decode the constants from JSON.
     * @param json The JSON input.
     * @param[out] constants The decoded constants.
     */
    friend void from_json(const nlohmann::json &json, Constants &constants);

    /** Delete the copy constructor. */
    Constants(const Constants &) = delete;

private:
    /** Hide the default constructor. */
    Constants() = default;
};

}

#endif // CONSTANTS_HPP
