#pragma once

#include <string>

namespace hlt {
    /**
     * The constants representing the game variation being played.
     * They come from game engine and changing them has no effect.
     * They are strictly informational.
     */
    namespace constants {
        void populate_constants(const std::string& string_from_engine);

        /** The maximum amount of halite a ship can carry. */
        extern int MAX_HALITE;
        /** The cost to build a single ship. */
        extern int SHIP_COST;
        /** The cost to build a dropoff. */
        extern int DROPOFF_COST;
        /** The maximum number of turns a game can last. */
        extern int MAX_TURNS;
        /** 1/EXTRACT_RATIO halite (rounded) is collected from a square per turn. */
        extern int EXTRACT_RATIO;
        /** 1/MOVE_COST_RATIO halite (rounded) is needed to move off a cell. */
        extern int MOVE_COST_RATIO;
        /** Whether inspiration is enabled. */
        extern bool INSPIRATION_ENABLED;
        /** A ship is inspired if at least INSPIRATION_SHIP_COUNT opponent ships are within this Manhattan distance. */
        extern int INSPIRATION_RADIUS;
        /** A ship is inspired if at least this many opponent ships are within INSPIRATION_RADIUS distance. */
        extern int INSPIRATION_SHIP_COUNT;
        /** An inspired ship mines 1/X halite from a cell per turn instead. */
        extern int INSPIRED_EXTRACT_RATIO;
        /** An inspired ship that removes Y halite from a cell collects X*Y additional halite. */
        extern double INSPIRED_BONUS_MULTIPLIER;
        /** An inspired ship instead spends 1/X% halite to move. */
        extern int INSPIRED_MOVE_COST_RATIO;

        /** Initial HP of a newly spawned ship. */
        extern int INITIAL_HP;
        /** HP damage dealt to the target of an attack. */
        extern int ATTACK_HP_DAMAGE;
        /** HP self-damage taken by the attacker. */
        extern int ATTACK_HP_SELF_DAMAGE;
        /** Fraction of target's halite stolen on a successful attack. */
        extern double ATTACK_HALITE_STEAL_RATIO;
        /** Whether combat attack/defend commands are enabled. */
        extern bool ENABLE_COMBAT_COMMANDS;

        /** Per-ship spawn cost growth: effective cost = SHIP_COST × (1 + growth × num_ships). */
        extern double SPAWN_COST_GROWTH;
        /** Spawn count threshold before quadratic over-threshold growth applies. */
        extern int SPAWN_QUAD_THRESHOLD;
        /** Quadratic over-threshold spawn growth. */
        extern double SPAWN_QUAD_GROWTH;
        /** Per-dropoff cost growth: effective cost = DROPOFF_COST × (1 + growth × num_dropoffs). */
        extern double DROPOFF_COST_GROWTH;
    }
}
