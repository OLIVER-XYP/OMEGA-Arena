#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

struct BotParams {
    // Core economy
    double stop_ratio = 0.04;
    double spawn_end_ratio = 0.42;
    int safe_margin = 1;
    int end_rush_turns = 30;
    int return_buffer = 12;
    int min_return_cargo = 220;
    int strategic_dist = 5;
    int strategic_rich = 550;
    double strategic_eff_thresh = 20.0;
    double return_close = 0.55;
    double return_far = 0.83;
    int return_dist = 15;

    // Mining / target search
    int search_radius_early = 10;
    int search_radius_mid = 8;
    int search_radius_late = 6;
    int min_cell_halite_consider = 30;
    int target_ttl = 8;
    double stay_bonus = 1.08;
    double long_distance_penalty = 0.02;

    // Exploration / adaptive territorial filter.
    // When we have more ships than the enemy, we relax the "stay on our half"
    // restriction and let ships venture into contested or enemy-side cells.
    // The filter is also a SOFT penalty (not a hard reject), so a rich enemy-side
    // cell can still beat a poor own-side cell even when dominance is parity.
    double invasion_penalty = 0.4;          // score penalty per cell of margin violation
    double dominance_relax_ratio = 1.3;     // at this my/enemy ship ratio, margin -> 0
    double dominance_disable_ratio = 1.8;   // at this ratio, filter is disabled entirely
    int expand_radius_bonus = 4;            // extra search radius when dominant

    // Navigation / congestion
    bool enable_secondary_nav = true;
    int congestion_radius = 2;
    int congestion_penalty = 12;
    bool allow_swap_like_escape = true;

    // Fixed benchmark cap: small-map max ships is intentionally hard-coded in MyBot.cpp.
    // These remaining spawn knobs are safe independent controls around timing/ROI only.
    bool enable_spawn_roi = true;
    int spawn_min_turns_left = 90;




    double spawn_roi_min = 1.10;
    int spawn_payback_turns = 70;
    double spawn_cutoff_turn_ratio = 0.12;
    double shipyard_target_penalty = 1.20;
    int stuck_reset_turns = 3;

    // Dropoff construction
    bool enable_dropoff = true;
    int max_dropoffs = 2;
    int dropoff_min_turn = 80;
    int dropoff_max_turn = 780;
    int dropoff_min_dist_from_own = 10;
    int dropoff_site_radius = 4;
    int dropoff_min_local_halite = 6000;
    int dropoff_min_nearby_ships = 3;
    int dropoff_min_player_halite_buffer = 1200;
    int dropoff_emergency_return_turns = 80;

    // Combat
    bool enable_attack = true;
    bool attack_only_when_enabled_by_rules = true;
    double attack_ratio = 1.55;
    int attack_min_target_halite = 120;
    int attack_max_self_halite_for_risk = 700;
    int attack_structure_value = 260;
    int attack_ship_value = 180;
    int attack_range = 1;              // max distance to attack (engine ATTACK_RANGE)

    // Hunt
    int hunt_min_enemy_halite = 320;   // minimum enemy cargo to consider hunting
    int hunt_max_range = 8;            // max distance to assign a hunter
    int hunt_max_hunters = 2;          // max simultaneous hunters per bot
    int hunt_min_turn = 80;            // don't hunt until this turn
    int low_hp_return = 30;            // return to heal if HP falls below this
    int attack_min_self_hp = 50;       // don't attack if own HP is below this
    // Pack hunting: how many of our hunters may converge on the SAME enemy. A
    // ship has 100 HP and a hit does 70, so it takes two simultaneous hits to
    // KILL (not just bruise). With 1 (default) hunters spread out, land one hit
    // and the target flees/banks -> the enemy fleet never shrinks. With 2, a
    // wolfpack corners a (preferably stationary, laden) miner and kills it in one
    // turn, banking its cargo via kill-credit AND removing it from the economy.
    int hunters_per_target = 1;

    // Camp mode: instead of chasing, hunters ambush near enemy structures
    // (shipyard + dropoffs), killing ships as they spawn or return to bank.
    // Zero travel time — the geometric bottleneck is eliminated.
    bool camp_enabled = false;
    int campers_per_structure = 4;     // hunters per enemy structure
    int camp_assign_turn = 30;         // first turn to start camping

    // Defensive immunity: a ship carrying >= this cargo issues a 'defend' command
    // (full attack-immunity for the turn) when an enemy ship is adjacent, instead
    // of mining/moving. Protects loaded miners from raiders. Default off (huge).
    int defend_min_cargo = 1000000;
    // How far away an enemy triggers a pre-emptive defend (Manhattan distance).
    // 1 = react only to adjacent (a turn behind a moving attacker); 2 = defend
    // before the raider closes, so it can never land a free hit.
    int defend_trigger_range = 1;
    // Value-gating for defend: also defend a (non-full) ship if its current cell
    // is worth >= this much halite -- i.e. staying to mine is already the plan, so
    // defending is free. Empty ships on mined-out cells keep moving (no freeze).
    int defend_min_cell_halite = 1000000;

    // Focus-fire: if at least this many of our ships are adjacent to the same
    // enemy ship, they ALL attack it the same turn (concentrated fire kills it
    // before it can trade efficiently). The coordinated counter to a roaming
    // raider. Default 0 = off. Set to 2+ on a defensive profile.
    int focus_fire_min_ships = 0;

    // Threat-aware navigation: when routing home (Return mode), penalize each
    // candidate cell by this weight times its threat (nearby enemy combat ships),
    // so loaded ships detour AROUND raiders instead of feeding them. The cost of
    // a move is distance-to-home + threat_avoid_weight*threat. Default 0 = off
    // (plain shortest-path nav). Tune up on profiles that must protect cargo.
    double threat_avoid_weight = 0.0;

    // Anti death-spiral: suppress spawning when the enemy outnumbers us by at
    // least this many ships (we're losing the military, so new ships will likely
    // die before banking -- preserve the treasury instead of feeding the rush).
    // Default large = off. Set to ~2 on a defensive profile.
    int spawn_stop_if_behind_by = 999;

    // Logging / diagnostics
    bool enable_periodic_log = true;
    int log_period = 20;
};

inline std::string trim_copy(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

inline bool parse_bool(const std::string &v) {
    return v == "1" || v == "true" || v == "True" || v == "TRUE";
}

template <typename T>
inline T clamp_value(T value, T lo, T hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

inline BotParams load_bot_params(const std::string &file_path) {
    BotParams p;
    std::ifstream in(file_path);
    if (!in.good()) return p;

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto k = trim_copy(line.substr(0, eq));
        auto v = trim_copy(line.substr(eq + 1));
        kv[k] = v;
    }

    auto getd = [&](const std::string &k, double cur) {
        auto it = kv.find(k);
        if (it == kv.end()) return cur;
        try { return std::stod(it->second); } catch (...) { return cur; }
    };
    auto geti = [&](const std::string &k, int cur) {
        auto it = kv.find(k);
        if (it == kv.end()) return cur;
        try { return std::stoi(it->second); } catch (...) { return cur; }
    };
    auto getb = [&](const std::string &k, bool cur) {
        auto it = kv.find(k);
        if (it == kv.end()) return cur;
        return parse_bool(it->second);
    };

    p.stop_ratio = getd("STOP_RATIO", p.stop_ratio);
    p.spawn_end_ratio = getd("SPAWN_END_RATIO", p.spawn_end_ratio);
    p.safe_margin = geti("SAFE_MARGIN", p.safe_margin);
    p.end_rush_turns = geti("END_RUSH_TURNS", p.end_rush_turns);
    p.return_buffer = geti("RETURN_BUFFER", p.return_buffer);
    p.min_return_cargo = geti("MIN_RETURN_CARGO", p.min_return_cargo);
    p.strategic_dist = geti("STRATEGIC_DIST", p.strategic_dist);
    p.strategic_rich = geti("STRATEGIC_RICH", p.strategic_rich);
    p.strategic_eff_thresh = getd("STRATEGIC_EFF_THRESH", p.strategic_eff_thresh);
    p.return_close = getd("RETURN_CLOSE", p.return_close);
    p.return_far = getd("RETURN_FAR", p.return_far);
    p.return_dist = geti("RETURN_DIST", p.return_dist);

    p.search_radius_early = geti("SEARCH_RADIUS_EARLY", p.search_radius_early);
    p.search_radius_mid = geti("SEARCH_RADIUS_MID", p.search_radius_mid);
    p.search_radius_late = geti("SEARCH_RADIUS_LATE", p.search_radius_late);
    p.min_cell_halite_consider = geti("MIN_CELL_HALITE_CONSIDER", p.min_cell_halite_consider);
    p.target_ttl = geti("TARGET_TTL", p.target_ttl);
    p.stay_bonus = getd("STAY_BONUS", p.stay_bonus);
    p.long_distance_penalty = getd("LONG_DISTANCE_PENALTY", p.long_distance_penalty);

    p.invasion_penalty = getd("INVASION_PENALTY", p.invasion_penalty);
    p.dominance_relax_ratio = getd("DOMINANCE_RELAX_RATIO", p.dominance_relax_ratio);
    p.dominance_disable_ratio = getd("DOMINANCE_DISABLE_RATIO", p.dominance_disable_ratio);
    p.expand_radius_bonus = geti("EXPAND_RADIUS_BONUS", p.expand_radius_bonus);

    p.enable_secondary_nav = getb("ENABLE_SECONDARY_NAV", p.enable_secondary_nav);
    p.congestion_radius = geti("CONGESTION_RADIUS", p.congestion_radius);
    p.congestion_penalty = geti("CONGESTION_PENALTY", p.congestion_penalty);
    p.allow_swap_like_escape = getb("ALLOW_SWAP_LIKE_ESCAPE", p.allow_swap_like_escape);

    p.enable_spawn_roi = getb("ENABLE_SPAWN_ROI", p.enable_spawn_roi);
    p.spawn_min_turns_left = geti("SPAWN_MIN_TURNS_LEFT", p.spawn_min_turns_left);




    p.spawn_roi_min = getd("SPAWN_ROI_MIN", p.spawn_roi_min);
    p.spawn_payback_turns = geti("SPAWN_PAYBACK_TURNS", p.spawn_payback_turns);
    p.spawn_cutoff_turn_ratio = getd("SPAWN_CUTOFF_TURN_RATIO", p.spawn_cutoff_turn_ratio);
    p.shipyard_target_penalty = getd("SHIPYARD_TARGET_PENALTY", p.shipyard_target_penalty);
    p.stuck_reset_turns = geti("STUCK_RESET_TURNS", p.stuck_reset_turns);

    p.enable_dropoff = getb("ENABLE_DROPOFF", p.enable_dropoff);
    p.max_dropoffs = geti("MAX_DROPOFFS", p.max_dropoffs);
    p.dropoff_min_turn = geti("DROPOFF_MIN_TURN", p.dropoff_min_turn);
    p.dropoff_max_turn = geti("DROPOFF_MAX_TURN", p.dropoff_max_turn);
    p.dropoff_min_dist_from_own = geti("DROPOFF_MIN_DIST_FROM_OWN", p.dropoff_min_dist_from_own);
    p.dropoff_site_radius = geti("DROPOFF_SITE_RADIUS", p.dropoff_site_radius);
    p.dropoff_min_local_halite = geti("DROPOFF_MIN_LOCAL_HALITE", p.dropoff_min_local_halite);
    p.dropoff_min_nearby_ships = geti("DROPOFF_MIN_NEARBY_SHIPS", p.dropoff_min_nearby_ships);
    p.dropoff_min_player_halite_buffer = geti("DROPOFF_MIN_PLAYER_HALITE_BUFFER", p.dropoff_min_player_halite_buffer);
    p.dropoff_emergency_return_turns = geti("DROPOFF_EMERGENCY_RETURN_TURNS", p.dropoff_emergency_return_turns);

    p.enable_attack = getb("ENABLE_ATTACK", p.enable_attack);
    p.attack_only_when_enabled_by_rules = getb("ATTACK_ONLY_WHEN_ENABLED_BY_RULES", p.attack_only_when_enabled_by_rules);
    p.attack_ratio = getd("ATTACK_RATIO", p.attack_ratio);
    p.attack_min_target_halite = geti("ATTACK_MIN_TARGET_HALITE", p.attack_min_target_halite);
    p.attack_max_self_halite_for_risk = geti("ATTACK_MAX_SELF_HALITE_FOR_RISK", p.attack_max_self_halite_for_risk);
    p.attack_structure_value = geti("ATTACK_STRUCTURE_VALUE", p.attack_structure_value);
    p.attack_ship_value = geti("ATTACK_SHIP_VALUE", p.attack_ship_value);
    p.attack_range = geti("ATTACK_RANGE", p.attack_range);

    p.hunt_min_enemy_halite = geti("HUNT_MIN_ENEMY_HALITE", p.hunt_min_enemy_halite);
    p.hunt_max_range = geti("HUNT_MAX_RANGE", p.hunt_max_range);
    p.hunt_max_hunters = geti("HUNT_MAX_HUNTERS", p.hunt_max_hunters);
    p.hunt_min_turn = geti("HUNT_MIN_TURN", p.hunt_min_turn);
    p.low_hp_return = geti("LOW_HP_RETURN", p.low_hp_return);
    p.attack_min_self_hp = geti("ATTACK_MIN_SELF_HP", p.attack_min_self_hp);
    p.hunters_per_target = geti("HUNTERS_PER_TARGET", p.hunters_per_target);
    p.camp_enabled = getb("CAMP_ENABLED", p.camp_enabled);
    p.campers_per_structure = geti("CAMPERS_PER_STRUCTURE", p.campers_per_structure);
    p.camp_assign_turn = geti("CAMP_ASSIGN_TURN", p.camp_assign_turn);
    p.defend_min_cargo = geti("DEFEND_MIN_CARGO", p.defend_min_cargo);
    p.defend_trigger_range = geti("DEFEND_TRIGGER_RANGE", p.defend_trigger_range);
    p.defend_min_cell_halite = geti("DEFEND_MIN_CELL_HALITE", p.defend_min_cell_halite);
    p.focus_fire_min_ships = geti("FOCUS_FIRE_MIN_SHIPS", p.focus_fire_min_ships);
    p.threat_avoid_weight = getd("THREAT_AVOID_WEIGHT", p.threat_avoid_weight);
    p.spawn_stop_if_behind_by = geti("SPAWN_STOP_IF_BEHIND_BY", p.spawn_stop_if_behind_by);

    p.enable_periodic_log = getb("ENABLE_PERIODIC_LOG", p.enable_periodic_log);
    p.log_period = geti("LOG_PERIOD", p.log_period);

    p.stop_ratio = clamp_value(p.stop_ratio, 0.01, 0.12);
    p.spawn_end_ratio = clamp_value(p.spawn_end_ratio, 0.20, 0.70);
    p.safe_margin = clamp_value(p.safe_margin, 0, 4);
    p.end_rush_turns = clamp_value(p.end_rush_turns, 10, 80);
    p.return_buffer = clamp_value(p.return_buffer, 2, 30);
    p.min_return_cargo = clamp_value(p.min_return_cargo, 80, 700);
    p.strategic_dist = clamp_value(p.strategic_dist, 2, 12);
    p.strategic_rich = clamp_value(p.strategic_rich, 250, 900);
    p.strategic_eff_thresh = clamp_value(p.strategic_eff_thresh, 5.0, 60.0);
    p.return_close = clamp_value(p.return_close, 0.30, 0.85);
    p.return_far = clamp_value(p.return_far, 0.45, 0.98);
    if (p.return_far < p.return_close) p.return_far = p.return_close;
    p.return_dist = clamp_value(p.return_dist, 6, 28);
    p.search_radius_early = clamp_value(p.search_radius_early, 4, 16);
    p.search_radius_mid = clamp_value(p.search_radius_mid, 4, 16);
    p.search_radius_late = clamp_value(p.search_radius_late, 3, 14);
    p.min_cell_halite_consider = clamp_value(p.min_cell_halite_consider, 5, 150);
    p.target_ttl = clamp_value(p.target_ttl, 2, 20);
    p.stay_bonus = clamp_value(p.stay_bonus, 0.90, 1.30);
    p.long_distance_penalty = clamp_value(p.long_distance_penalty, 0.0, 0.12);
    p.invasion_penalty = clamp_value(p.invasion_penalty, 0.0, 2.0);
    p.dominance_relax_ratio = clamp_value(p.dominance_relax_ratio, 1.0, 2.5);
    p.dominance_disable_ratio = clamp_value(p.dominance_disable_ratio, p.dominance_relax_ratio, 4.0);
    p.expand_radius_bonus = clamp_value(p.expand_radius_bonus, 0, 8);
    p.congestion_radius = clamp_value(p.congestion_radius, 1, 4);
    p.congestion_penalty = clamp_value(p.congestion_penalty, 0, 50);
    p.spawn_min_turns_left = clamp_value(p.spawn_min_turns_left, 20, 220);
    p.spawn_roi_min = clamp_value(p.spawn_roi_min, 0.75, 2.00);
    p.spawn_payback_turns = clamp_value(p.spawn_payback_turns, 20, 180);
    p.spawn_cutoff_turn_ratio = clamp_value(p.spawn_cutoff_turn_ratio, 0.05, 0.25);
    p.shipyard_target_penalty = clamp_value(p.shipyard_target_penalty, 0.0, 3.0);
    p.stuck_reset_turns = clamp_value(p.stuck_reset_turns, 1, 10);
    p.max_dropoffs = clamp_value(p.max_dropoffs, 0, 4);
    p.dropoff_min_turn = clamp_value(p.dropoff_min_turn, 20, 220);
    p.dropoff_max_turn = clamp_value(p.dropoff_max_turn, p.dropoff_min_turn, 900);
    p.dropoff_min_dist_from_own = clamp_value(p.dropoff_min_dist_from_own, 5, 24);
    p.dropoff_site_radius = clamp_value(p.dropoff_site_radius, 2, 8);
    p.dropoff_min_local_halite = clamp_value(p.dropoff_min_local_halite, 2000, 16000);
    p.dropoff_min_nearby_ships = clamp_value(p.dropoff_min_nearby_ships, 1, 8);
    p.dropoff_min_player_halite_buffer = clamp_value(p.dropoff_min_player_halite_buffer, 0, 4000);
    p.dropoff_emergency_return_turns = clamp_value(p.dropoff_emergency_return_turns, 20, 160);
    p.attack_ratio = clamp_value(p.attack_ratio, 0.8, 3.0);
    p.attack_min_target_halite = clamp_value(p.attack_min_target_halite, 0, 800);
    p.attack_max_self_halite_for_risk = clamp_value(p.attack_max_self_halite_for_risk, 0, 1000);
    p.attack_structure_value = clamp_value(p.attack_structure_value, 0, 800);
    p.attack_ship_value = clamp_value(p.attack_ship_value, 0, 800);
    p.attack_range = clamp_value(p.attack_range, 1, 5);
    p.hunt_min_enemy_halite = clamp_value(p.hunt_min_enemy_halite, 0, 900);
    p.hunt_max_range = clamp_value(p.hunt_max_range, 2, 20);
    p.hunt_max_hunters = clamp_value(p.hunt_max_hunters, 0, 30);
    p.hunters_per_target = clamp_value(p.hunters_per_target, 1, 4);
    p.campers_per_structure = clamp_value(p.campers_per_structure, 1, 10);
    p.camp_assign_turn = clamp_value(p.camp_assign_turn, 0, 250);
    p.hunt_min_turn = clamp_value(p.hunt_min_turn, 0, 250);
    p.low_hp_return = clamp_value(p.low_hp_return, 0, 100);
    p.attack_min_self_hp = clamp_value(p.attack_min_self_hp, 0, 100);
    p.log_period = clamp_value(p.log_period, 1, 200);

    return p;
}
