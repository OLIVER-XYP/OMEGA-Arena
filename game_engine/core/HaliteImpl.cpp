#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "HaliteImpl.hpp"
#include "Logging.hpp"
#include "MultiEventSink.hpp"
#include "ReplayCollector.hpp"
#include "StatsCollector.hpp"
#include "TaskExecutor.hpp"
#include "TurnStatsCollector.hpp"

namespace hlt {

/**
 * Initialize the game.
 * @param player_commands The list of player commands.
 */
void HaliteImpl::initialize_game(const std::vector<std::string> &player_commands,
                                 const Snapshot &snapshot) {
    // Update max turn # by map size (300 @ 32x32 to 500 at 80x80)
    auto &mut_constants = Constants::get_mut();
    auto turns = mut_constants.MIN_TURNS;
    const unsigned long max_dimension = std::max(game.map.width, game.map.height);
    if (max_dimension > mut_constants.MIN_TURN_THRESHOLD) {
        turns += static_cast<unsigned long>(((max_dimension - mut_constants.MIN_TURN_THRESHOLD) / static_cast<double>(mut_constants.MAX_TURN_THRESHOLD - mut_constants.MIN_TURN_THRESHOLD)) * (mut_constants.MAX_TURNS - mut_constants.MIN_TURNS));
    }
    mut_constants.MAX_TURNS = turns;

    const auto &constants = Constants::get();
    auto &players = game.store.players;
    assert(game.map.factories.size() >= player_commands.size());

    // Add a 0 frame so we can record beginning-of-game state
    game.replay.full_frames.emplace_back();
    std::unordered_set<Location> changed_cells;

    auto factory_iterator = game.map.factories.begin();

    // Load the map from the snapshot
    if (!snapshot.map.empty()) {
        assert(snapshot.map.size() == static_cast<decltype(snapshot.map)::size_type>(game.map.width * game.map.height));

        for (dimension_type row = 0; row < game.map.height; row++) {
            for (dimension_type col = 0; col < game.map.width; col++) {
                game.map.at(col, row).energy = snapshot.map.at(static_cast<size_t>(row * game.map.width + col));
                changed_cells.emplace(row, col);
            }
        }
    }

    for (dimension_type row = 0; row < game.map.height; row++) {
        for (dimension_type col = 0; col < game.map.width; col++) {
            auto &cell = game.map.at(col, row);
            // Snapshot the post-generation energy as the RegenPhase regrowth ceiling.
            cell.initial_energy = cell.energy;
            game.store.map_total_energy += cell.energy;
        }
    }

    for (const auto &command : player_commands) {
        auto &factory = *factory_iterator++;
        auto player = game.store.player_factory.make(factory, command);
        player.energy = constants.INITIAL_ENERGY;
        player.factory_halite = constants.INITIAL_FACTORY_HALITE;
        player.factory_destroyed = false;
        game.game_statistics.player_statistics.emplace_back(player.id, game.rng());
        if (snapshot.players.find(player.id) != snapshot.players.end()) {
            const auto &player_snapshot = snapshot.players.at(player.id);
            player.factory = player_snapshot.factory;
            player.energy = player_snapshot.energy;

            for (const auto &[_, dropoff_location] : player_snapshot.dropoffs) {
                auto &cell = game.map.at(dropoff_location);
                cell.owner = player.id;
                auto drop = game.store.new_dropoff(dropoff_location);
                drop.halite_pool = constants.INITIAL_DROPOFF_HALITE;
                drop.destroyed = false;
                player.dropoffs.emplace_back(drop);
                game.replay.full_frames.back().events.push_back(
                        std::make_unique<ConstructionEvent>(
                                dropoff_location, player.id, Entity::id_type{0}));
                changed_cells.insert(dropoff_location);
            }

            for (const auto &entity : player_snapshot.entities) {
                auto &cell = game.map.at(entity.location);
                const auto &new_entity = game.store.new_entity(entity.energy, player.id);
                cell.entity = new_entity.id;
                player.add_entity(new_entity.id, entity.location);
                game.replay.full_frames.back().events.push_back(
                        std::make_unique<SpawnEvent>(
                                entity.location, entity.energy, player.id, cell.entity));
            }
        }
        players.emplace(player.id, player);
    }
    game.replay.game_statistics = game.game_statistics;

    for (auto &[player_id, player] : game.store.players) {
        // Zero the energy on factory and mark as owned.
        auto &factory = game.map.at(player.factory);
        game.store.map_total_energy -= factory.energy;
        factory.energy = 0;
        factory.owner = player_id;
        // Prepare the log.
        game.logs.add(player_id);
    }
    game.replay.full_frames.back().add_cells(game.map, changed_cells);
    update_player_stats();
    game.replay.full_frames.back().add_end_state(game.store);
}

/** Run the game. */
void HaliteImpl::run_game() {
    const auto &constants = Constants::get();

    const char *profiling_env = std::getenv("HALITE_TURN_PROFILE_CSV");
    const bool profiling_enabled = profiling_env != nullptr && profiling_env[0] != '\0';
    const std::string profiling_path = profiling_enabled ? std::string(profiling_env) : std::string{};
    const char *scenario_env = std::getenv("HALITE_PROFILE_SCENARIO");
    const std::string scenario_id = (scenario_env != nullptr && scenario_env[0] != '\0') ? std::string(scenario_env) : std::string("default");

    const char *exec_threads_env = std::getenv("HALITE_EXEC_THREADS");
    const bool use_thread_pool = exec_threads_env != nullptr && exec_threads_env[0] != '\0';
    std::unique_ptr<ThreadPoolExecutor> thread_pool_executor;
    InlineExecutor inline_executor;
    if (use_thread_pool) {
        const std::size_t configured_threads = std::max<std::size_t>(1, static_cast<std::size_t>(std::strtoul(exec_threads_env, nullptr, 10)));
        thread_pool_executor = std::make_unique<ThreadPoolExecutor>(configured_threads);
    }

    const char *dynamic_gate_env = std::getenv("HALITE_EXEC_DYNAMIC_MIN_SHIPS");
    const std::size_t dynamic_min_ships = (dynamic_gate_env != nullptr && dynamic_gate_env[0] != '\0')
                                              ? std::max<std::size_t>(1, static_cast<std::size_t>(std::strtoul(dynamic_gate_env, nullptr, 10)))
                                              : 40;

    std::vector<TurnExecutionProfile> turn_profiles;

    // Per-ship audit: only collect/write when HALITE_SHIP_AUDIT env var is set.
    const char *audit_path = std::getenv("HALITE_SHIP_AUDIT");
    game.store.audit_enabled = (audit_path != nullptr && audit_path[0] != '\0');

    ordered_id_map<Player, std::future<void>> results{};
    bool success = true;
    for (auto &[player_id, player] : game.store.players) {
        Logging::log("Launching with command " + player.command, Logging::Level::Info, player.id);
        try {
            game.networking.connect_player(player);
        } catch (const BotError &e) {
            success = false;
            kill_player(player_id);
            Logging::log("Player could not be launched", Logging::Level::Error, player.id);
            Logging::log(e.what(), Logging::Level::Error);
        }
    }
    if (!success) {
        game.replay.players.insert(game.store.players.begin(), game.store.players.end());
        Logging::log("Some players failed to launch, aborting", Logging::Level::Error);
        return;
    }

    for (auto &[player_id, player] : game.store.players) {
        Logging::log("Initializing player", Logging::Level::Info, player_id);
        results[player_id] = std::async(std::launch::async,
                                        [&networking = game.networking, &player = player] {
                                            networking.initialize_player(player);
                                        });
    }
    for (auto &[player_id, result] : results) {
        try {
            result.get();
            Logging::log("Initialized player " + game.store.get_player(player_id).name, Logging::Level::Info, player_id);
        } catch (const BotError &e) {
            kill_player(player_id);
        }
    }
    game.replay.players.insert(game.store.players.begin(), game.store.players.end());
    Logging::log("Player initialization complete");

    for (game.turn_number = 1; game.turn_number <= constants.MAX_TURNS; game.turn_number++) {
        game.store.current_turn = game.turn_number;
        Logging::set_turn_number(game.turn_number);
        game.logs.set_turn_number(game.turn_number);
        // Used to track the current turn number inside Event::update_stats
        game.game_statistics.turn_number = game.turn_number;
        Logging::log([turn_number = game.turn_number]() {
            return "Starting turn " + std::to_string(turn_number);
        }, Logging::Level::Debug);
        // Create new turn struct for replay file, to be filled by further turn actions
        game.replay.full_frames.emplace_back();
        game.replay.full_frames.back().add_entities(game.store);

        TurnExecutionProfile turn_profile{};

        TaskExecutor *selected_executor = &inline_executor;
        if (use_thread_pool) {
            if (dynamic_min_ships == 0) {
                selected_executor = thread_pool_executor.get();
            } else {
                std::size_t live_ship_count = 0;
                for (const auto &[player_id, player] : game.store.players) {
                    (void)player_id;
                    live_ship_count += player.entities.size();
                }
                selected_executor = live_ship_count >= dynamic_min_ships
                                        ? static_cast<TaskExecutor *>(thread_pool_executor.get())
                                        : static_cast<TaskExecutor *>(&inline_executor);
            }
        }

        process_turn(*selected_executor, profiling_enabled ? &turn_profile : nullptr);
        if (profiling_enabled) {
            turn_profiles.push_back(std::move(turn_profile));
        }

        // Add end of frame state.
        game.replay.full_frames.back().add_end_state(game.store);

        if (game_ended()) {
            game.turn_number++;
            break;
        }
    }
    game.game_statistics.number_turns = game.turn_number;

    game.replay.full_frames.emplace_back();
    game.replay.full_frames.back().add_entities(game.store);
    update_player_stats();
    game.replay.full_frames.back().add_end_state(game.store);

    rank_players();
    Logging::log("Game has ended");
    Logging::set_turn_number(Logging::ended);
    game.logs.set_turn_number(PlayerLog::ended);
    for (const auto &[player_id, player] : game.store.players) {
        game.replay.players.find(player_id)->second.terminated = player.terminated;
        if (!player.terminated) {
            game.networking.kill_player(player);
        }
    }

    if (profiling_enabled) {
        bool need_header = true;
        {
            std::ifstream probe(profiling_path);
            if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
                need_header = false;
            }
        }
        std::ofstream out(profiling_path, std::ios::app);
        if (out) {
            if (need_header) {
                out << "scenario,executor,threads,turn,total_turn_ns,parallel_for_calls,total_items_processed,phase,phase_ns\n";
            }
            for (std::size_t turn_index = 0; turn_index < turn_profiles.size(); ++turn_index) {
                const auto &profile = turn_profiles[turn_index];
                for (const auto &phase_timing : profile.phase_timings) {
                    out << scenario_id << ',' << profile.executor_type << ',' << profile.executor_thread_count
                        << ',' << (turn_index + 1) << ',' << profile.total_turn_ns << ','
                        << profile.executor_stats.parallel_for_calls << ',' << profile.executor_stats.total_items_processed << ','
                        << phase_timing.phase_name << ',' << phase_timing.duration_ns << "\n";
                }
            }
        }

        std::vector<long long> total_turn_ns_values;
        total_turn_ns_values.reserve(turn_profiles.size());
        for (const auto &profile : turn_profiles) {
            total_turn_ns_values.push_back(profile.total_turn_ns);
        }
        std::sort(total_turn_ns_values.begin(), total_turn_ns_values.end());

        auto percentile_at = [&](double p) -> long long {
            if (total_turn_ns_values.empty()) {
                return 0;
            }
            const double max_index = static_cast<double>(total_turn_ns_values.size() - 1);
            const auto idx = static_cast<std::size_t>(std::max(0.0, std::min(max_index, p * max_index)));
            return total_turn_ns_values[idx];
        };

        const long long total_ns = std::accumulate(total_turn_ns_values.begin(), total_turn_ns_values.end(), 0LL);
        const long long avg_ns = total_turn_ns_values.empty() ? 0 : total_ns / static_cast<long long>(total_turn_ns_values.size());
        const long long p50_ns = percentile_at(0.50);
        const long long p95_ns = percentile_at(0.95);
        const long long p99_ns = percentile_at(0.99);

        const std::string summary_path = profiling_path + ".summary.csv";
        bool need_summary_header = true;
        {
            std::ifstream probe(summary_path);
            if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
                need_summary_header = false;
            }
        }
        std::ofstream summary(summary_path, std::ios::app);
        if (summary) {
            if (need_summary_header) {
                summary << "scenario,executor,threads,turns,avg_turn_ns,p50_turn_ns,p95_turn_ns,p99_turn_ns\n";
            }
            const std::string executor_name = turn_profiles.empty() ? std::string("unknown") : turn_profiles.front().executor_type;
            const std::size_t thread_count = turn_profiles.empty() ? 0 : turn_profiles.front().executor_thread_count;
            summary << scenario_id << ',' << executor_name << ',' << thread_count << ','
                    << total_turn_ns_values.size() << ',' << avg_ns << ',' << p50_ns << ',' << p95_ns << ',' << p99_ns << "\n";
        }
    }

    // Per-ship audit: capture survivors with survived=true, then dump CSV.
    if (game.store.audit_enabled) {
        const auto final_turn = game.turn_number;
        for (auto &[entity_id, entity] : game.store.entities) {
            (void)entity_id;
            game.store.ship_audits.push_back(Store::ShipAudit{
                static_cast<int>(entity.owner.value),
                static_cast<long>(entity.id.value),
                entity.spawn_turn,
                final_turn,
                entity.lifetime_deposited,
                entity.enemy_halite_taken,
                entity.enemy_hp_dealt,
                true
            });
        }
        if (const char *audit_path = std::getenv("HALITE_SHIP_AUDIT")) {
            const char *seed_env = std::getenv("HALITE_AUDIT_SEED");
            const std::string seed_str = (seed_env != nullptr) ? std::string(seed_env) : std::string("");
            // Write header only if the file is new/empty (so multiple game
            // invocations appending to the same CSV don't sprinkle headers).
            bool need_header = true;
            {
                std::ifstream probe(audit_path);
                if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
                    need_header = false;
                }
            }
            std::ofstream out(audit_path, std::ios::app);
            if (out) {
                if (need_header) {
                    out << "seed,player,entity,spawn_turn,death_turn,deposited,halite_taken,hp_dealt,survived\n";
                }
                for (const auto &a : game.store.ship_audits) {
                    out << seed_str << ',' << a.player_id << ',' << a.entity_id << ',' << a.spawn_turn
                        << ',' << a.death_turn << ',' << a.lifetime_deposited
                        << ',' << a.enemy_halite_taken << ',' << a.enemy_hp_dealt
                        << ',' << (a.survived ? 1 : 0) << "\n";
                }
            }
        }
    }
}

/** Retrieve and process commands, and update the game state for the current turn. */
void HaliteImpl::process_turn(TaskExecutor &executor, TurnExecutionProfile *profile) {
    // Reset HP-death flag; checked by game_ended() after this turn completes.
    game.store.any_ship_hp_zeroed = false;
    // Retrieve all commands
    using Commands = std::vector<std::unique_ptr<Command>>;
    ordered_id_map<Player, Commands> commands{};
    id_map<Player, std::future<Commands>> results{};
    for (auto &[player_id, player] : game.store.players) {
        if (!player.terminated) {
            results[player_id] = std::async(std::launch::async,
                                            [&networking = game.networking, &player = player] {
                                                return networking.handle_frame(player);
                                            });
        }
    }
    for (auto &[player_id, result] : results) {
        try {
            commands[player_id] = result.get();
        } catch (const BotError &e) {
            (void)e;
            kill_player(player_id);
            commands.erase(player_id);
        }
    }

    GameState state{game.store, game.map, game.game_statistics};
    state.turn.number = game.turn_number;
    observers::ReplayCollector replay_collector{game.replay};
    observers::StatsCollector stats_collector{game.game_statistics, game.store, game.map};
    events::MultiEventSink sink;
    sink.add(replay_collector);
    sink.add(stats_collector);
    auto step_result = turn_engine_.step(state, commands, sink, executor, profile);
    observers::TurnStatsCollector turn_stats_collector;
    turn_stats_collector.collect(state, GameConfig::from_constants(), executor);

    for (auto player_id : step_result.eliminated_players) {
        if (game.store.players.find(player_id) != game.store.players.end()) {
            kill_player(player_id);
        }
    }

    game.store.changed_cells = step_result.changed_cells;
    game.replay.full_frames.back().moves = std::move(commands);
    game.replay.full_frames.back().add_cells(game.map, game.store.changed_cells);
}

void HaliteImpl::update_inspiration() {
    if (!Constants::get().INSPIRATION_ENABLED) {
        return;
    }

    const auto inspiration_radius = Constants::get().INSPIRATION_RADIUS;
    const auto ships_threshold = Constants::get().INSPIRATION_SHIP_COUNT;

    // Check every ship of every player
    for (const auto &[player_id, player] : game.store.players) {
        for (const auto &[entity_id, location] : player.entities) {
            // map from player ID to # of ships within the inspiration
            // radius of the current ship
            id_map<Player, long> ships_in_radius;
            for (const auto &[pid, _] : game.store.players) {
                ships_in_radius[pid] = 0;
            }

            // Explore locations around this ship
            for (auto dx = -inspiration_radius; dx <= inspiration_radius; dx++) {
                for (auto dy = -inspiration_radius; dy <= inspiration_radius; dy++) {
                    const auto cur = Location{
                        (((location.x + dx) % game.map.width) + game.map.width) % game.map.width,
                        (((location.y + dy) % game.map.height) + game.map.height) % game.map.height
                    };

                    const auto &cur_cell = game.map.at(cur);
                    if (cur_cell.entity == Entity::None ||
                        game.map.distance(location, cur) > inspiration_radius) {
                        continue;
                    }
                    const auto &other_entity = game.store.get_entity(cur_cell.entity);
                    ships_in_radius[other_entity.owner]++;
                }
            }

            // Total up ships of other player
            unsigned long opponent_entities = 0;
            for (const auto &[pid, count] : ships_in_radius) {
                if (pid != player_id) {
                    opponent_entities += count;
                }
            }

            // Mark ship as inspired or not
            auto &entity = game.store.get_entity(entity_id);
            entity.is_inspired = opponent_entities >= ships_threshold;
        }
    }
}

/**
 * Determine whether a player can still play in the future
 *
 * @param player Player to check
 * @return True if the player can play on the next turn
 */
bool HaliteImpl::player_can_play(const Player &player) const {
    return !player.entities.empty() || player.energy >= Constants::get().NEW_ENTITY_ENERGY_COST;
}
/**
 * Determine whether the game has ended.
 * @return True if the game has ended.
 */
bool HaliteImpl::game_ended() const {
    for (const auto &[player_id, player] : game.store.players) {
        (void)player_id;
        if (player.terminated) continue;

        energy_type structure_total = player.factory_halite;
        for (const auto &dropoff : player.dropoffs) {
            structure_total += dropoff.halite_pool;
        }

        if (structure_total == 0) {
            return true;
        }
    }
    return false;
}

/**
 * Update all players' statistics after a single turn.
 */
void HaliteImpl::update_player_stats() {
    for (PlayerStatistics &player_stats : game.game_statistics.player_statistics) {
        // Player with sprites is still alive, so mark as alive on this turn and add production gained
        const auto &player_id = player_stats.player_id;
        const Player &player = game.store.get_player(player_id);
        if (!player.terminated && player.can_play) {
            if (player_can_play(player)) { // Player may have died during this turn, in which case do not update final turn
                player_stats.last_turn_alive = game.turn_number;
                
                // Calculate carried_at_end
                player_stats.carried_at_end = 0;
                for (const auto &[_entity_id, location] : player.entities) {
                    player_stats.carried_at_end += game.store.get_entity(_entity_id).energy;
                }
            }
            energy_type structure_score = player.factory_halite;
            for (const auto &dropoff : player.dropoffs) {
                structure_score += dropoff.halite_pool;
            }
            player_stats.turn_productions.push_back(structure_score);
            player_stats.turn_deposited.push_back(player.total_energy_deposited);
            player_stats.number_dropoffs = player.dropoffs.size();
            player_stats.ships_peak = std::max(player_stats.ships_peak, (long)player.entities.size());
            for (const auto &[_entity_id, location] : player.entities) {
                const dimension_type entity_distance = game.map.distance(location, player.factory);
                if (entity_distance > player_stats.max_entity_distance)
                    player_stats.max_entity_distance = entity_distance;
                player_stats.total_distance += entity_distance;
                player_stats.total_entity_lifespan++;
                if (possible_interaction(player_id, location)) {
                    player_stats.interaction_opportunities++;
                }
            }

            player_stats.halite_per_dropoff[player.factory] = player.factory_halite;
            player_stats.total_production = player.factory_halite;
            for (const auto &dropoff : player.dropoffs) {
                player_stats.halite_per_dropoff[dropoff.location] = dropoff.halite_pool;
                player_stats.total_production += dropoff.halite_pool;
            }
        } else {
            player_stats.turn_productions.push_back(0);
            player_stats.turn_deposited.push_back(0);
        }
    }
}

/**
 * Determine if entity owned by given player is in range of another player (their entity, dropoff, or factory) and thus may interact
 *
 * param owner_id Id of owner of entity at given location
 * param entity_location Location of entity we are assessing for an interaction opportunity
 * return bool Indicator of whether there players are in close range for an interaction (true) or not (false)
 */
bool HaliteImpl::possible_interaction(const Player::id_type owner_id, const Location entity_location) {
    // Fetch all locations 2 cells away
    std::unordered_set<Location> close_cells{};
    const auto neighbors = game.map.get_neighbors(entity_location);
    close_cells.insert(neighbors.begin(), neighbors.end());
    for (const Location &neighbor : neighbors) {
        const auto cells_once_removed = game.map.get_neighbors(neighbor);
        close_cells.insert(cells_once_removed.begin(), cells_once_removed.end());
    }
    // Interaction possibilty implies a cell has an entity owned by another player or there is a factory or dropoff
    // of another player on the cell. Interactions between entities of a single player are ignored
    for (const Location &cell_location : close_cells) {
        const Cell &cell = game.map.at(cell_location);
        if (cell.entity != Entity::None) {
            if (game.store.get_entity(cell.entity).owner != owner_id) return true;
        }
        if (cell.owner != Player::None && cell.owner != owner_id) return true;
    }
    return false;

}

/**
 * Update players' rankings based on their final turn alive, then break ties with production totals in final turn.
 * Function is intended to be called at end of game, and will in place modify the ranking field of player statistics
 * to rank players from winner (1) to last player.
 */
void HaliteImpl::rank_players() {
    auto &statistics = game.game_statistics.player_statistics;
    std::stable_sort(statistics.begin(), statistics.end());
    // Reverse list to have best players first
    std::reverse(statistics.begin(), statistics.end());

    for (size_t index = 0; index < statistics.size(); ++index) {
        statistics[index].rank = index + 1l;
    }

    // Re-sort by player ID
    std::stable_sort(statistics.begin(), statistics.end(),
                     [](const PlayerStatistics &a, const PlayerStatistics &b) -> bool {
                         return a.player_id.value < b.player_id.value;
                     });
}

void HaliteImpl::kill_player(const Player::id_type &player_id) {
    Logging::log("Killing player", Logging::Level::Warning, player_id);
    auto &player = game.store.get_player(player_id);
    player.terminated = true;
    player.can_play = false;
    game.networking.kill_player(player);

    auto &entities = player.entities;
    for (auto entity_iterator = entities.begin();
         entity_iterator != entities.end();
         entity_iterator = entities.erase(entity_iterator)) {
        const auto &[entity_id, location] = *entity_iterator;
        auto &cell = game.map.at(location);
        cell.entity = Entity::None;
        game.store.delete_entity(entity_id);
    }
    player.energy = 0;
}

/**
 * Construct HaliteImpl from game interface.
 * @param game The game interface.
 */
HaliteImpl::HaliteImpl(Halite &game) : game(game), turn_engine_(game.get_config()) {}

/**
 * Handle a player command error.
 * @param offenders The set of players this turn who have caused errors.
 * @param commands The player command mapping.
 * @param error The error caused by the player.
 */
void HaliteImpl::handle_error(std::unordered_set<Player::id_type> &offenders,
                              ordered_id_map<Player, std::vector<std::unique_ptr<Command>>> &commands,
                              CommandError error) {
    const auto message = error->log_message();
    const auto &faulty = error->command();
    const auto player_id = error->player;

    // Log the error information.
    if (error->ignored) {
        Logging::log(message, Logging::Level::Warning, player_id);
        game.logs.log(player_id, message, PlayerLog::Level::Warning);
    } else {
        Logging::log(message, Logging::Level::Error, player_id);
        offenders.emplace(player_id);
        game.logs.log(player_id, message, PlayerLog::Level::Error);
    }

    // Find the position of a command within a player's command list.
    auto &player_commands = commands[player_id];
    const auto find_position = [&player_commands](const Command &faulty) {
        return std::find_if(player_commands.begin(), player_commands.end(), [&faulty](const auto &command) {
            return std::addressof(*command) == std::addressof(faulty);
        });
    };

    // Given a command position, log the context of that command.
    const auto log_context = [&player_commands, &game = game, &player_id](const auto position) {
        static constexpr long COMMAND_CONTEXT_LINES = 2L;
        const auto distance = static_cast<long>(std::distance(player_commands.begin(), position));
        const auto commands_begin = player_commands.begin() + std::max(0L, distance - COMMAND_CONTEXT_LINES);
        const auto commands_end = player_commands.begin() +
                                  std::min(static_cast<long>(player_commands.size()),
                                           distance + COMMAND_CONTEXT_LINES + 1);
        for (auto iterator = commands_begin; iterator != commands_end; iterator++) {
            auto number = std::distance(player_commands.begin(), iterator);
            auto marker = iterator == position ? ">>> " : "    ";
            game.logs.log(player_id, marker + std::to_string(number + 1) + "   " + (*iterator)->to_bot_serial());
        }
        game.logs.log(player_id, "");
    };

    // Log the faulty command.
    auto position = find_position(faulty);
    auto distance = std::distance(player_commands.begin(), position);
    game.logs.log(player_id, "At command " + std::to_string(distance + 1) + " of " +
                  std::to_string(player_commands.size()) + ":");
    log_context(position);

    // If there is a context, log the context.
    const auto &context = error->context();
    if (!context.empty()) {
        const auto context_message = error->context_message();
        game.logs.log(player_id, context_message);
        static constexpr size_t MAX_CONTEXT_COMMANDS = 5;
        const auto context_end = context.begin() + std::min(MAX_CONTEXT_COMMANDS, context.size());
        for (auto iterator = context.begin(); iterator != context_end; iterator++) {
            log_context(find_position(*iterator));
        }
        if (context.size() > MAX_CONTEXT_COMMANDS) {
            game.logs.log(player_id, "(suppressing " + std::to_string(context.size() - MAX_CONTEXT_COMMANDS) +
                                     " other commands)\n");
        }
    }
}

}
