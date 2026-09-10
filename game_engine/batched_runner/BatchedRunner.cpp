#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Command.hpp"
#include "Constants.hpp"
#include "CpuTurnBackend.hpp"
#include "CudaTurnBackend.hpp"
#include "CudaBatchedMap.hpp"
#include "EventSink.hpp"
#include "GameConfig.hpp"
#include "GameState.hpp"
#include "Generator.hpp"
#include "Map.hpp"
#include "MultiEventSink.hpp"
#include "StatsCollector.hpp"
#include "Statistics.hpp"
#include "Store.hpp"
#include "TaskExecutor.hpp"
#include "TurnEngine.hpp"
#include "TurnStatsCollector.hpp"
#include "RegenPhase.hpp"
#include "RuleContext.hpp"
#include "PhaseHelpers.hpp"
#include "nlohmann/json.hpp"

namespace hlt::batched {
namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    unsigned long games = 1;
    unsigned long turn_limit = 300;
    unsigned int seed_base = 900000;
    std::string backend = "gpu";
    std::string config_path = "starter_kits/C++/competitive_engine_v2.json";
    std::string provider_p0 = "random_legal";
    std::string provider_p1 = "random_legal";
    unsigned long workers = 1;
    std::string results_json;
    std::string profile_json;
    bool lockstep_runner = false;
    bool lockstep_batched_inspiration = false;
    bool lockstep_batched_validation = false;
    bool lockstep_batched_dump = false;
    bool lockstep_batched_regen = false;
    bool lockstep_batched_end_economy = false;
    bool lockstep_economy_window_benchmark = false;
    bool persistent_frame_window = false;
    bool batched_inspiration_resident_benchmark = false;
    unsigned long regen_iterations = 1;
    unsigned long dump_iterations = 1;
    unsigned long spawn_iterations = 1;
    unsigned long movement_commands = 256;
    unsigned long movement_iterations = 1;
    unsigned long validation_commands = 256;
    unsigned long validation_iterations = 1;
    energy_type dump_refill_cargo = 100;
    bool batched_regen_benchmark = false;
    bool batched_dump_benchmark = false;
    bool batched_pipeline_benchmark = false;
    bool batched_mining_benchmark = false;
    bool batched_economy_pipeline_benchmark = false;
    bool batched_spawn_benchmark = false;
    bool batched_movement_benchmark = false;
    bool batched_movement_apply_benchmark = false;
    bool batched_destination_benchmark = false;
    bool batched_collision_benchmark = false;
    bool batched_real_economy_benchmark = false;
    bool batched_validation_benchmark = false;
};

struct NullEventSink final : events::EventSink {
    void emit(const events::DomainEvent &event) override { (void)event; }
};

struct ProviderView {
    const Store &store;
    const Map &map;
    const GameStatistics &statistics;
    Player::id_type player_id;
    unsigned long turn;
    unsigned long turn_limit;
    std::mt19937 &rng;
};

class ActionProvider {
public:
    virtual ~ActionProvider() = default;
    virtual std::string name() const = 0;
    virtual void begin_game(unsigned int seed, Player::id_type player_id) {
        (void)seed;
        (void)player_id;
    }
    virtual PlayerCommandList actions_for_turn(const ProviderView &view) = 0;
};

Direction direction_towards(const Map &map, Location from, Location to) {
    const auto width = map.width;
    const auto height = map.height;
    auto dx = static_cast<int>(to.x) - static_cast<int>(from.x);
    auto dy = static_cast<int>(to.y) - static_cast<int>(from.y);
    if (std::abs(dx) > static_cast<int>(width) / 2) {
        dx = dx > 0 ? dx - static_cast<int>(width) : dx + static_cast<int>(width);
    }
    if (std::abs(dy) > static_cast<int>(height) / 2) {
        dy = dy > 0 ? dy - static_cast<int>(height) : dy + static_cast<int>(height);
    }
    if (std::abs(dx) >= std::abs(dy) && dx != 0) {
        return dx > 0 ? Direction::East : Direction::West;
    }
    if (dy != 0) {
        return dy > 0 ? Direction::South : Direction::North;
    }
    return Direction::Still;
}

Location moved_location(const Map &map, Location location, Direction direction) {
    map.move_location(location, direction);
    return location;
}

bool can_move(const Store &store, const Map &map, Entity::id_type id, Location location, Direction direction) {
    if (direction == Direction::Still) {
        return true;
    }
    const auto entity_it = store.entities_ref().find(id);
    if (entity_it == store.entities_ref().end()) {
        return false;
    }
    const auto cost = map.at(location).energy / static_cast<energy_type>(std::max<unsigned long>(1, Constants::get().MOVE_COST_RATIO));
    return entity_it->second.energy >= cost;
}

std::vector<Direction> legal_move_directions(const Store &store, const Map &map, Entity::id_type id, Location location) {
    std::vector<Direction> dirs{Direction::Still};
    for (auto dir : {Direction::North, Direction::South, Direction::East, Direction::West}) {
        if (can_move(store, map, id, location, dir)) {
            dirs.push_back(dir);
        }
    }
    return dirs;
}

const PlayerStatistics *stats_for_player(const GameStatistics &statistics, Player::id_type player_id) {
    for (const auto &stats : statistics.player_statistics) {
        if (stats.player_id == player_id) {
            return &stats;
        }
    }
    return nullptr;
}

energy_type structure_score(const Player &player) {
    energy_type score = player.factory_halite;
    for (const auto &dropoff : player.dropoffs) {
        score += dropoff.halite_pool;
    }
    return score;
}

bool shipyard_occupied(const Player &player) {
    for (const auto &[entity_id, location] : player.entities) {
        (void)entity_id;
        if (location == player.factory) {
            return true;
        }
    }
    return false;
}

bool destination_has_own_ship(const Player &player, Location destination, Location current) {
    if (destination == current) {
        return false;
    }
    for (const auto &[entity_id, location] : player.entities) {
        (void)entity_id;
        if (location == destination) {
            return true;
        }
    }
    return false;
}

bool safe_destination(const Player &player,
                      Location current,
                      Location destination,
                      const std::unordered_set<Location> &reserved) {
    return reserved.find(destination) == reserved.end() &&
           !destination_has_own_ship(player, destination, current);
}

Direction choose_safe_direction(const Store &store,
                                const Map &map,
                                const Player &player,
                                Entity::id_type entity_id,
                                Location location,
                                Direction desired,
                                const std::unordered_set<Location> &reserved,
                                std::mt19937 &rng,
                                bool randomize_fallback) {
    std::vector<Direction> candidates;
    candidates.push_back(desired);
    candidates.push_back(Direction::Still);
    for (auto dir : {Direction::North, Direction::South, Direction::East, Direction::West}) {
        if (dir != desired) {
            candidates.push_back(dir);
        }
    }
    if (randomize_fallback && candidates.size() > 2) {
        std::shuffle(candidates.begin() + 2, candidates.end(), rng);
    }
    for (auto dir : candidates) {
        if (!can_move(store, map, entity_id, location, dir)) {
            continue;
        }
        const auto destination = moved_location(map, location, dir);
        if (safe_destination(player, location, destination, reserved)) {
            return dir;
        }
    }
    return Direction::Still;
}

energy_type spawn_cost_for(const Player &player) {
    const auto &constants = Constants::get();
    const auto n = static_cast<double>(player.entities.size());
    const auto over = std::max(0.0, n + 1.0 - static_cast<double>(constants.SPAWN_QUAD_THRESHOLD));
    const auto multiplier = 1.0 + constants.SPAWN_COST_GROWTH * n + constants.SPAWN_QUAD_GROWTH * over * over;
    return static_cast<energy_type>(static_cast<double>(constants.NEW_ENTITY_ENERGY_COST) * multiplier);
}

bool can_spawn(const Player &player) {
    return !player.terminated && !shipyard_occupied(player) && player.energy >= spawn_cost_for(player);
}

class RandomLegalProvider final : public ActionProvider {
public:
    std::string name() const override { return "random_legal"; }

    PlayerCommandList actions_for_turn(const ProviderView &view) override {
        PlayerCommandList commands;
        const auto &player = view.store.players_ref().at(view.player_id);
        std::unordered_set<Location> reserved;
        for (const auto &[entity_id, location] : player.entities) {
            auto dirs = legal_move_directions(view.store, view.map, entity_id, location);
            std::shuffle(dirs.begin(), dirs.end(), view.rng);
            Direction chosen = Direction::Still;
            for (auto dir : dirs) {
                const auto destination = moved_location(view.map, location, dir);
                if (safe_destination(player, location, destination, reserved)) {
                    chosen = dir;
                    break;
                }
            }
            reserved.insert(moved_location(view.map, location, chosen));
            commands.push_back(std::make_unique<MoveCommand>(entity_id, chosen));
        }
        if (can_spawn(player) && reserved.find(player.factory) == reserved.end()) {
            std::uniform_int_distribution<int> dist(0, 99);
            if (dist(view.rng) < 25) {
                commands.push_back(std::make_unique<SpawnCommand>());
            }
        }
        return commands;
    }
};

class HeuristicProvider final : public ActionProvider {
    std::string provider_name;
    int aggression;
    int defend_rate;

public:
    HeuristicProvider(std::string name, int aggression, int defend_rate)
        : provider_name(std::move(name)), aggression(aggression), defend_rate(defend_rate) {}

    std::string name() const override { return provider_name; }

    PlayerCommandList actions_for_turn(const ProviderView &view) override {
        PlayerCommandList commands;
        const auto &player = view.store.players_ref().at(view.player_id);
        std::unordered_set<Location> reserved;
        const auto remaining = view.turn_limit > view.turn ? view.turn_limit - view.turn : 0;
        const auto *stats = stats_for_player(view.statistics, view.player_id);
        const auto current_score = stats != nullptr && !stats->turn_productions.empty()
                                       ? stats->turn_productions.back()
                                       : structure_score(player);
        for (const auto &[entity_id, location] : player.entities) {
            const auto entity_it = view.store.entities_ref().find(entity_id);
            if (entity_it == view.store.entities_ref().end()) {
                continue;
            }
            const auto &entity = entity_it->second;
            if (defend_rate > 0 && entity.energy < Constants::get().MAX_ENERGY / 5) {
                std::uniform_int_distribution<int> dist(0, 99);
                if (dist(view.rng) < defend_rate) {
                    commands.push_back(std::make_unique<DefendCommand>(entity_id));
                    continue;
                }
            }
            Location nearest_enemy_location = location;
            int nearest_enemy_distance = 1000000;
            if (aggression > 0) {
                int best_distance = 1000000;
                for (const auto &[other_id, other] : view.store.entities_ref()) {
                    if (other.owner == view.player_id) {
                        continue;
                    }
                    const auto &enemy_player = view.store.players_ref().at(other.owner);
                    auto target_location = enemy_player.get_entity_location(other_id);
                    const auto distance = static_cast<int>(view.map.distance(location, target_location));
                    if (distance < nearest_enemy_distance) {
                        nearest_enemy_location = target_location;
                        nearest_enemy_distance = distance;
                    }
                    if (distance <= Constants::get().ATTACK_RANGE && distance < best_distance) {
                        best_distance = distance;
                    }
                }
                if (best_distance <= Constants::get().ATTACK_RANGE) {
                    std::uniform_int_distribution<int> dist(0, 99);
                    if (dist(view.rng) < aggression) {
                        commands.push_back(std::make_unique<DefendCommand>(entity_id));
                        reserved.insert(location);
                        continue;
                    }
                }
            }
            Direction direction = Direction::Still;
            if (remaining < static_cast<unsigned long>(view.map.distance(location, player.factory) + 8) ||
                entity.energy >= Constants::get().MAX_ENERGY * 7 / 10) {
                direction = direction_towards(view.map, location, player.factory);
            } else if (view.map.at(location).energy < 80) {
                energy_type best_halite = view.map.at(location).energy;
                if (aggression > 40 && nearest_enemy_distance < 10) {
                    direction = direction_towards(view.map, location, nearest_enemy_location);
                    best_halite = 0;
                }
                for (auto dir : {Direction::North, Direction::South, Direction::East, Direction::West}) {
                    const auto next = moved_location(view.map, location, dir);
                    if (view.map.at(next).energy > best_halite && can_move(view.store, view.map, entity_id, location, dir)) {
                        best_halite = view.map.at(next).energy;
                        direction = dir;
                    }
                }
            }
            if (!can_move(view.store, view.map, entity_id, location, direction)) {
                direction = Direction::Still;
            }
            direction = choose_safe_direction(view.store, view.map, player, entity_id, location, direction, reserved, view.rng, false);
            reserved.insert(moved_location(view.map, location, direction));
            commands.push_back(std::make_unique<MoveCommand>(entity_id, direction));
        }
        if (can_spawn(player) && reserved.find(player.factory) == reserved.end() && view.turn < view.turn_limit * 2 / 3) {
            const auto ships = player.entities.size();
            const auto target_ships = provider_name == "eco_heuristic" ? 12U : provider_name == "control_heuristic" ? 9U : 14U;
            if (ships < target_ships && player.energy > spawn_cost_for(player) + current_score / 20) {
                commands.push_back(std::make_unique<SpawnCommand>());
            }
        }
        return commands;
    }
};

std::unique_ptr<ActionProvider> make_provider(const std::string &name) {
    if (name == "random_legal") {
        return std::make_unique<RandomLegalProvider>();
    }
    if (name == "eco_heuristic") {
        return std::make_unique<HeuristicProvider>(name, 5, 5);
    }
    if (name == "aggro_heuristic") {
        return std::make_unique<HeuristicProvider>(name, 60, 0);
    }
    if (name == "control_heuristic") {
        return std::make_unique<HeuristicProvider>(name, 20, 45);
    }
    if (name == "replay_scripted") {
        throw std::invalid_argument("replay_scripted provider is reserved for phase 1.1 and is not implemented yet");
    }
    throw std::invalid_argument("unknown provider: " + name);
}

class GameInstance {
    unsigned int seed;
    unsigned long turn_limit;
    GameConfig config;
    mapgen::MapParameters map_parameters;
    Map map;
    Store store;
    GameStatistics statistics;
    TurnEngine turn_engine;
    CpuTurnBackend cpu_backend;
    std::unique_ptr<ITurnBackend> gpu_backend;
    std::unique_ptr<ActionProvider> p0;
    std::unique_ptr<ActionProvider> p1;
    std::mt19937 rng;
    bool ended = false;
    unsigned long turns_run = 0;
    long long elapsed_ms = 0;
    std::vector<TurnExecutionProfile> profiles;
    std::vector<std::string> errors;

    ITurnBackend *selected_backend() {
        if (config.runtime.engine_backend == EngineBackendMode::Cpu) {
            return &cpu_backend;
        }
        if (gpu::cuda_turn_backend_available()) {
            if (!gpu_backend) {
                gpu_backend = gpu::make_cuda_turn_backend(turn_engine, config);
            }
            if (gpu_backend) {
                return gpu_backend.get();
            }
        }
        if (config.runtime.engine_backend == EngineBackendMode::Auto) {
            return &cpu_backend;
        }
        throw std::runtime_error(std::string("GPU backend unavailable: ") + gpu::cuda_turn_backend_unavailable_reason());
    }

    bool game_ended() const {
        for (const auto &[player_id, player] : store.players_ref()) {
            (void)player_id;
            if (player.terminated) {
                continue;
            }
            if (structure_score(player) == 0) {
                return true;
            }
        }
        return false;
    }

    void init_player(Player &player, unsigned int random_id) {
        player.energy = Constants::get().INITIAL_ENERGY;
        player.factory_halite = Constants::get().INITIAL_FACTORY_HALITE;
        player.factory_destroyed = false;
        statistics.player_statistics.emplace_back(player.id, random_id);
        auto &factory = map.at(player.factory);
        store.map_total_energy -= factory.energy;
        factory.energy = 0;
        factory.initial_energy = 0;
        factory.owner = player.id;
        auto &entity = store.new_entity(0, player.id);
        entity.spawn_turn = 0;
        factory.entity = entity.id;
        player.add_entity(entity.id, player.factory);
    }

public:
    GameInstance(unsigned int seed,
                 unsigned long turn_limit,
                 GameConfig config,
                 std::unique_ptr<ActionProvider> p0,
                 std::unique_ptr<ActionProvider> p1)
        : seed(seed),
          turn_limit(turn_limit),
          config(std::move(config)),
          map_parameters{mapgen::MapType::Fractal, seed, 32, 32, 2},
          map(32, 32),
          turn_engine(this->config),
          cpu_backend(turn_engine),
          p0(std::move(p0)),
          p1(std::move(p1)),
          rng(seed) {}

    void initialize() {
        mapgen::Generator::generate(map, map_parameters);
        for (dimension_type row = 0; row < map.height; ++row) {
            for (dimension_type col = 0; col < map.width; ++col) {
                auto &cell = map.at(col, row);
                cell.initial_energy = cell.energy;
                store.map_total_energy += cell.energy;
                statistics.map_total_halite += cell.energy;
            }
        }
        if (map.factories.size() < 2) {
            throw std::runtime_error("map generation produced fewer than two factories");
        }
        auto &player0 = store.new_player(map.factories[0], p0->name());
        auto &player1 = store.new_player(map.factories[1], p1->name());
        init_player(player0, rng());
        init_player(player1, rng());
        p0->begin_game(seed, player0.id);
        p1->begin_game(seed, player1.id);
    }

    bool run_one_turn(unsigned long turn, bool collect_profile, TaskExecutor &executor) {
        if (ended || turn > turn_limit) {
            return false;
        }
        store.current_turn = turn;
        statistics.turn_number = turn;
        ActionBatch actions;
        prepare_actions_for_turn(turn, actions);
        return run_prepared_turn(turn, actions, collect_profile, executor);
    }

    void prepare_actions_for_turn(unsigned long turn, ActionBatch &actions) {
        set_turn_number(turn);
        actions.clear();
        const Player::id_type player0{0};
        const Player::id_type player1{1};
        actions[player0] = p0->actions_for_turn(ProviderView{store, map, statistics, player0, turn, turn_limit, rng});
        actions[player1] = p1->actions_for_turn(ProviderView{store, map, statistics, player1, turn, turn_limit, rng});
    }

    bool run_prepared_turn(unsigned long turn, ActionBatch &actions, bool collect_profile, TaskExecutor &executor) {
        if (ended || turn > turn_limit) {
            return false;
        }
        set_turn_number(turn);
        GameState state{store, map, statistics};
        state.turn.number = turn;
        NullEventSink null_sink;
        observers::StatsCollector stats_collector{statistics, store, map};
        events::MultiEventSink sink;
        sink.add(null_sink);
        sink.add(stats_collector);
        TurnExecutionProfile profile;
        TurnExecutionProfile *profile_ptr = collect_profile ? &profile : nullptr;
        StepResult step_result = selected_backend()->step(state, actions, sink, executor, profile_ptr);
        finish_prepared_turn(turn, state, step_result, std::move(profile), collect_profile, executor);
        return !ended;
    }

    bool run_prevalidated_turn(unsigned long turn,
                               ActionBatch &actions,
                               StepResult prevalidated_result,
                               bool collect_profile,
                               TaskExecutor &executor) {
        if (ended || turn > turn_limit) {
            return false;
        }
        set_turn_number(turn);
        GameState state{store, map, statistics};
        state.turn.number = turn;
        NullEventSink null_sink;
        observers::StatsCollector stats_collector{statistics, store, map};
        events::MultiEventSink sink;
        sink.add(null_sink);
        sink.add(stats_collector);
        TurnExecutionProfile profile;
        TurnExecutionProfile *profile_ptr = collect_profile ? &profile : nullptr;
        auto *backend = selected_backend();
        if (!backend->supports_prevalidated_step()) {
            throw std::runtime_error("selected backend does not support prevalidated lockstep turns");
        }
        StepResult step_result = backend->step_from_validated(state,
                                                             actions,
                                                             sink,
                                                             executor,
                                                             profile_ptr,
                                                             std::move(prevalidated_result));
        finish_prepared_turn(turn, state, step_result, std::move(profile), collect_profile, executor);
        return !ended;
    }

    StepResult run_until_dump(unsigned long turn,
                              ActionBatch &actions,
                              StepResult prevalidated_result,
                              bool collect_profile,
                              TaskExecutor &executor,
                              TurnExecutionProfile &profile) {
        if (ended || turn > turn_limit) {
            return prevalidated_result;
        }
        set_turn_number(turn);
        GameState state{store, map, statistics};
        state.turn.number = turn;
        NullEventSink null_sink;
        observers::StatsCollector stats_collector{statistics, store, map};
        events::MultiEventSink sink;
        sink.add(null_sink);
        sink.add(stats_collector);
        TurnExecutionProfile *profile_ptr = collect_profile ? &profile : nullptr;
        auto *backend = selected_backend();
        if (!backend->supports_split_dump_step()) {
            throw std::runtime_error("selected backend does not support split dump turns");
        }
        return backend->step_until_dump(state,
                                        actions,
                                        sink,
                                        executor,
                                        profile_ptr,
                                        std::move(prevalidated_result));
    }

    bool run_after_dump(unsigned long turn,
                        ActionBatch &actions,
                        StepResult partial_result,
                        bool collect_profile,
                        TaskExecutor &executor,
                        TurnExecutionProfile &profile) {
        if (ended || turn > turn_limit) {
            return false;
        }
        set_turn_number(turn);
        GameState state{store, map, statistics};
        state.turn.number = turn;
        NullEventSink null_sink;
        observers::StatsCollector stats_collector{statistics, store, map};
        events::MultiEventSink sink;
        sink.add(null_sink);
        sink.add(stats_collector);
        TurnExecutionProfile *profile_ptr = collect_profile ? &profile : nullptr;
        StepResult step_result = selected_backend()->step_after_dump(state,
                                                                    actions,
                                                                    sink,
                                                                    executor,
                                                                    profile_ptr,
                                                                    std::move(partial_result));
        finish_prepared_turn(turn, state, step_result, std::move(profile), collect_profile, executor);
        return !ended;
    }

    void finish_prepared_turn(unsigned long turn,
                              GameState &state,
                              const StepResult &step_result,
                              TurnExecutionProfile profile,
                              bool collect_profile,
                              TaskExecutor &executor) {
        observers::TurnStatsCollector turn_stats_collector;
        turn_stats_collector.collect(state, config, executor);
        for (auto player_id : step_result.eliminated_players) {
            auto it = store.players_ref().find(player_id);
            if (it != store.players_ref().end()) {
                it->second.terminated = true;
                it->second.can_play = false;
            }
        }
        for (const auto &error : step_result.non_fatal_errors) {
            if (errors.size() < 32) {
                errors.push_back(error);
            } else if (errors.size() == 32) {
                errors.emplace_back("additional non-fatal errors suppressed");
            }
        }
        if (collect_profile) {
            profiles.push_back(std::move(profile));
        }
        turns_run = turn;
        if (game_ended()) {
            ended = true;
        }
    }

    void finish_run(long long run_elapsed_ms) {
        statistics.number_turns = turns_run;
        rank_players();
        elapsed_ms = run_elapsed_ms;
        statistics.execution_time = elapsed_ms;
    }

    void run(bool collect_profile) {
        const auto start = Clock::now();
        InlineExecutor inline_executor;
        TaskExecutor *executor = &inline_executor;
        for (unsigned long turn = 1; turn <= turn_limit; ++turn) {
            run_one_turn(turn, collect_profile, *executor);
            if (ended) {
                break;
            }
        }
        finish_run(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count());
    }

    void rank_players() {
        auto &stats = statistics.player_statistics;
        std::stable_sort(stats.begin(), stats.end());
        std::reverse(stats.begin(), stats.end());
        for (std::size_t index = 0; index < stats.size(); ++index) {
            stats[index].rank = static_cast<long>(index + 1);
        }
        std::stable_sort(stats.begin(), stats.end(), [](const PlayerStatistics &a, const PlayerStatistics &b) {
            return a.player_id.value < b.player_id.value;
        });
    }

    nlohmann::json summary_json() const {
        nlohmann::json players = nlohmann::json::object();
        for (const auto &stats : statistics.player_statistics) {
            const auto &player = store.players_ref().at(stats.player_id);
            players[to_string(stats.player_id)] = {
                {"rank", stats.rank},
                {"score", stats.turn_productions.empty() ? structure_score(player) : stats.turn_productions.back()},
                {"ships", player.entities.size()},
                {"terminated", player.terminated},
                {"provider", player.command}
            };
        }
        return {
            {"seed", seed},
            {"turns", turns_run},
            {"ended", ended},
            {"elapsed_ms", elapsed_ms},
            {"errors", errors.size()},
            {"error_samples", errors},
            {"map_total_halite", statistics.map_total_halite},
            {"players", players}
        };
    }

    StateFrame capture_frame() {
        GameState state{store, map, statistics};
        state.turn.number = store.current_turn;
        return capture_state_frame(state);
    }

    void apply_frame(const StateFrame &frame) {
        GameState state{store, map, statistics};
        state.turn.number = store.current_turn;
        apply_state_frame(state, frame);
    }

    void set_turn_number(unsigned long turn) {
        store.current_turn = turn;
        statistics.turn_number = turn;
    }

    const Store &store_ref() const { return store; }

    GameState state_view() {
        GameState state{store, map, statistics};
        state.turn.number = store.current_turn;
        return state;
    }

    bool is_finished() const { return ended; }

    const std::vector<TurnExecutionProfile> &turn_profiles() const { return profiles; }
};

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--games") {
            options.games = std::stoul(require_value(arg));
        } else if (arg == "--backend") {
            options.backend = require_value(arg);
        } else if (arg == "--config") {
            options.config_path = require_value(arg);
        } else if (arg == "--turn-limit") {
            options.turn_limit = std::stoul(require_value(arg));
        } else if (arg == "--seed-base") {
            options.seed_base = static_cast<unsigned int>(std::stoul(require_value(arg)));
        } else if (arg == "--provider-p0") {
            options.provider_p0 = require_value(arg);
        } else if (arg == "--provider-p1") {
            options.provider_p1 = require_value(arg);
        } else if (arg == "--workers") {
            options.workers = std::stoul(require_value(arg));
        } else if (arg == "--results-json") {
            options.results_json = require_value(arg);
        } else if (arg == "--profile-json") {
            options.profile_json = require_value(arg);
        } else if (arg == "--lockstep-runner") {
            options.lockstep_runner = true;
        } else if (arg == "--lockstep-batched-inspiration") {
            options.lockstep_batched_inspiration = true;
        } else if (arg == "--lockstep-batched-validation") {
            options.lockstep_batched_validation = true;
        } else if (arg == "--lockstep-batched-dump") {
            options.lockstep_batched_dump = true;
        } else if (arg == "--lockstep-batched-regen") {
            options.lockstep_batched_regen = true;
        } else if (arg == "--lockstep-batched-end-economy") {
            options.lockstep_batched_end_economy = true;
        } else if (arg == "--lockstep-economy-window-benchmark") {
            options.lockstep_economy_window_benchmark = true;
        } else if (arg == "--persistent-frame-window") {
            options.persistent_frame_window = true;
        } else if (arg == "--batched-inspiration-resident-benchmark") {
            options.batched_inspiration_resident_benchmark = true;
        } else if (arg == "--regen-iterations") {
            options.regen_iterations = std::stoul(require_value(arg));
        } else if (arg == "--dump-iterations") {
            options.dump_iterations = std::stoul(require_value(arg));
        } else if (arg == "--spawn-iterations") {
            options.spawn_iterations = std::stoul(require_value(arg));
        } else if (arg == "--movement-commands") {
            options.movement_commands = std::stoul(require_value(arg));
        } else if (arg == "--movement-iterations") {
            options.movement_iterations = std::stoul(require_value(arg));
        } else if (arg == "--validation-commands") {
            options.validation_commands = std::stoul(require_value(arg));
        } else if (arg == "--validation-iterations") {
            options.validation_iterations = std::stoul(require_value(arg));
        } else if (arg == "--dump-refill-cargo") {
            options.dump_refill_cargo = static_cast<energy_type>(std::stol(require_value(arg)));
        } else if (arg == "--batched-regen-benchmark") {
            options.batched_regen_benchmark = true;
        } else if (arg == "--batched-dump-benchmark") {
            options.batched_dump_benchmark = true;
        } else if (arg == "--batched-pipeline-benchmark") {
            options.batched_pipeline_benchmark = true;
        } else if (arg == "--batched-mining-benchmark") {
            options.batched_mining_benchmark = true;
        } else if (arg == "--batched-economy-pipeline-benchmark") {
            options.batched_economy_pipeline_benchmark = true;
        } else if (arg == "--batched-spawn-benchmark") {
            options.batched_spawn_benchmark = true;
        } else if (arg == "--batched-movement-benchmark") {
            options.batched_movement_benchmark = true;
        } else if (arg == "--batched-movement-apply-benchmark") {
            options.batched_movement_apply_benchmark = true;
        } else if (arg == "--batched-destination-benchmark") {
            options.batched_destination_benchmark = true;
        } else if (arg == "--batched-collision-benchmark") {
            options.batched_collision_benchmark = true;
        } else if (arg == "--batched-real-economy-benchmark") {
            options.batched_real_economy_benchmark = true;
        } else if (arg == "--batched-validation-benchmark") {
            options.batched_validation_benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "halite_batched_runner --games N --backend gpu|cpu|auto --config PATH --turn-limit N "
                         "--provider-p0 NAME --provider-p1 NAME --workers N --results-json PATH --profile-json PATH "
                         "--lockstep-runner --lockstep-batched-inspiration --lockstep-batched-validation "
                         "--lockstep-batched-dump --lockstep-batched-regen --lockstep-batched-end-economy "
                         "--lockstep-economy-window-benchmark --persistent-frame-window "
                         "--batched-inspiration-resident-benchmark "
                         "--batched-regen-benchmark --regen-iterations N --batched-dump-benchmark "
                         "--dump-iterations N --dump-refill-cargo N --batched-pipeline-benchmark "
                         "--batched-mining-benchmark --batched-economy-pipeline-benchmark "
                         "--batched-spawn-benchmark --spawn-iterations N "
                         "--batched-movement-benchmark --movement-commands N --movement-iterations N "
                         "--batched-movement-apply-benchmark --batched-destination-benchmark "
                         "--batched-collision-benchmark --batched-real-economy-benchmark "
                         "--batched-validation-benchmark --validation-commands N --validation-iterations N\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.games == 0) {
        throw std::invalid_argument("--games must be positive");
    }
    if (options.turn_limit == 0) {
        throw std::invalid_argument("--turn-limit must be positive");
    }
    if (options.workers == 0) {
        throw std::invalid_argument("--workers must be positive");
    }
    if (options.regen_iterations == 0) {
        throw std::invalid_argument("--regen-iterations must be positive");
    }
    if (options.dump_iterations == 0) {
        throw std::invalid_argument("--dump-iterations must be positive");
    }
    if (options.spawn_iterations == 0) {
        throw std::invalid_argument("--spawn-iterations must be positive");
    }
    if (options.movement_commands == 0) {
        throw std::invalid_argument("--movement-commands must be positive");
    }
    if (options.movement_iterations == 0) {
        throw std::invalid_argument("--movement-iterations must be positive");
    }
    if (options.validation_commands == 0) {
        throw std::invalid_argument("--validation-commands must be positive");
    }
    if (options.validation_iterations == 0) {
        throw std::invalid_argument("--validation-iterations must be positive");
    }
    if (options.dump_refill_cargo <= 0) {
        throw std::invalid_argument("--dump-refill-cargo must be positive");
    }
    const bool any_lockstep_batch_flag = options.lockstep_batched_inspiration ||
                                         options.lockstep_batched_validation ||
                                         options.lockstep_batched_dump ||
                                         options.lockstep_batched_regen ||
                                         options.lockstep_batched_end_economy;
    if (options.lockstep_runner && options.backend == "gpu" && !any_lockstep_batch_flag) {
        options.lockstep_batched_inspiration = true;
        options.lockstep_batched_validation = true;
        options.lockstep_batched_dump = true;
        options.lockstep_batched_regen = true;
        options.lockstep_batched_end_economy = true;
    }
    return options;
}

GameConfig load_config(const Options &options) {
    if (!options.config_path.empty()) {
        std::ifstream constants_file(options.config_path);
        if (!constants_file.is_open()) {
            throw std::runtime_error("could not open config file: " + options.config_path);
        }
        nlohmann::json constants_json;
        constants_file >> constants_json;
        from_json(constants_json, Constants::get_mut());
    }
    auto config = GameConfig::from_constants();
    config.match.max_turns = options.turn_limit;
    config.match.min_turns = options.turn_limit;
    config.runtime.engine_backend = parse_engine_backend_mode(options.backend);
    config.runtime.gpu_batch_size = 1;
    config.apply_to_global_constants();
    return config;
}

std::unique_ptr<GameInstance> make_initialized_game(const Options &options,
                                                    const GameConfig &config,
                                                    unsigned long index) {
    auto game = std::make_unique<GameInstance>(
        options.seed_base + static_cast<unsigned int>(index),
        options.turn_limit,
        config,
        make_provider(options.provider_p0),
        make_provider(options.provider_p1));
    game->initialize();
    return game;
}

StateFrame synthetic_regen_frame(unsigned int seed, const GameConfig &config) {
    std::mt19937 rng(seed);
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.resize(cells);
    frame.cell_initial_energy.resize(cells);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    std::uniform_int_distribution<int> initial_dist(200, 1000);
    std::uniform_int_distribution<int> depleted_percent(0, 85);
    for (std::size_t index = 0; index < cells; ++index) {
        const auto initial = static_cast<energy_type>(initial_dist(rng));
        const auto depleted = static_cast<energy_type>(initial * depleted_percent(rng) / 100);
        frame.cell_initial_energy[index] = initial;
        frame.cell_energy[index] = depleted;
        frame.map_total_energy += depleted;
    }
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);
    frame.cell_owner[0] = Player::id_type{0};
    frame.cell_owner[16 * 32 + 16] = Player::id_type{1};
    frame.cell_energy[0] = 0;
    frame.cell_energy[16 * 32 + 16] = 0;
    frame.cell_initial_energy[0] = 0;
    frame.cell_initial_energy[16 * 32 + 16] = 0;
    (void)config;
    return frame;
}

void cpu_regen_frame(StateFrame &frame, const GameConfig &config) {
    const auto &economy = config.ruleset.economy;
    if (!economy.cell_regen_enabled || economy.cell_regen_rate <= 0.0 || economy.cell_regen_cap_fraction <= 0.0) {
        return;
    }
    for (std::size_t index = 0; index < frame.cell_energy.size(); ++index) {
        if (frame.cell_owner[index] != Player::None || frame.cell_initial_energy[index] <= 0) {
            continue;
        }
        const auto cap = static_cast<energy_type>(
            economy.cell_regen_cap_fraction * static_cast<double>(frame.cell_initial_energy[index]));
        if (frame.cell_energy[index] >= cap) {
            continue;
        }
        const auto regen = static_cast<energy_type>(
            std::ceil(economy.cell_regen_rate * static_cast<double>(frame.cell_initial_energy[index])));
        if (regen <= 0) {
            continue;
        }
        const auto new_energy = std::min(cap, frame.cell_energy[index] + regen);
        const auto delta = new_energy - frame.cell_energy[index];
        if (delta <= 0) {
            continue;
        }
        frame.cell_energy[index] = new_energy;
        frame.map_total_energy += static_cast<unsigned long long>(delta);
    }
}

StateFrame synthetic_dump_frame(unsigned int seed, const GameConfig &config, std::size_t ships_per_game = 256) {
    std::mt19937 rng(seed);
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);

    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.players[0].dropoff_begin = 0;
    frame.players[0].dropoff_count = 1;
    frame.players[1].dropoff_begin = 1;
    frame.players[1].dropoff_count = 1;
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);
    frame.dropoffs.push_back(DropoffFrameEntry{Player::id_type{0}, Location{4, 4}});
    frame.dropoffs.push_back(DropoffFrameEntry{Player::id_type{1}, Location{20, 20}});

    const auto own_cell = [&](Location location, Player::id_type owner) {
        const auto index = static_cast<std::size_t>(location.y * frame.width + location.x);
        frame.cell_owner[index] = owner;
    };
    own_cell(frame.players[0].factory, Player::id_type{0});
    own_cell(frame.players[1].factory, Player::id_type{1});
    own_cell(frame.dropoffs[0].location, Player::id_type{0});
    own_cell(frame.dropoffs[1].location, Player::id_type{1});

    std::uniform_int_distribution<int> cargo_dist(1, std::max<energy_type>(1, config.ruleset.economy.max_energy));
    std::uniform_int_distribution<int> coordinate_dist(0, 31);
    frame.entities.reserve(ships_per_game);
    frame.player_entity_ids.reserve(ships_per_game);
    for (int owner_value = 0; owner_value < 2; ++owner_value) {
        const auto owner = Player::id_type{owner_value};
        const auto player_slot = static_cast<std::size_t>(owner_value);
        frame.players[player_slot].ship_begin = frame.player_entity_ids.size();
        const auto player_ships = ships_per_game / 2 + (static_cast<std::size_t>(owner_value) < ships_per_game % 2 ? 1 : 0);
        for (std::size_t local_index = 0; local_index < player_ships; ++local_index) {
            const auto index = frame.entities.size();
            Location location{0, 0};
            if (index % 4 == 0) {
                location = frame.players[player_slot].factory;
            } else if (index % 4 == 1) {
                location = frame.dropoffs[player_slot].location;
            } else {
                location = Location{static_cast<dimension_type>(coordinate_dist(rng)),
                                    static_cast<dimension_type>(coordinate_dist(rng))};
            }
            EntityFrameEntry entity;
            entity.id = Entity::id_type{static_cast<int>(index + 1)};
            entity.owner = owner;
            entity.location = location;
            entity.energy = static_cast<energy_type>(cargo_dist(rng));
            entity.hp = std::max(1, config.ruleset.combat.initial_hp - static_cast<int>(index % 7));
            entity.alive = true;
            frame.entity_slot_by_id.emplace(entity.id, frame.entities.size());
            frame.player_entity_ids.push_back(entity.id);
            frame.entities.push_back(entity);
            frame.players[player_slot].ship_count += 1;
            if (index % 4 <= 1) {
                const auto cell = static_cast<std::size_t>(location.y * frame.width + location.x);
                frame.cell_entity[cell] = entity.id;
            }
        }
    }
    return frame;
}

StateFrame synthetic_pipeline_frame(unsigned int seed, const GameConfig &config, std::size_t ships_per_game = 256) {
    auto frame = synthetic_dump_frame(seed, config, ships_per_game);
    std::mt19937 rng(seed ^ 0x9e3779b9U);
    std::uniform_int_distribution<int> initial_dist(200, 1000);
    std::uniform_int_distribution<int> depleted_percent(0, 85);
    frame.map_total_energy = 0;
    for (std::size_t index = 0; index < frame.cell_energy.size(); ++index) {
        if (frame.cell_owner[index] != Player::None) {
            frame.cell_energy[index] = 0;
            frame.cell_initial_energy[index] = 0;
            continue;
        }
        const auto initial = static_cast<energy_type>(initial_dist(rng));
        const auto depleted = static_cast<energy_type>(initial * depleted_percent(rng) / 100);
        frame.cell_initial_energy[index] = initial;
        frame.cell_energy[index] = depleted;
        frame.map_total_energy += depleted;
    }
    return frame;
}

StateFrame synthetic_mining_frame(unsigned int seed, const GameConfig &config, std::size_t ships_per_game = 256) {
    std::mt19937 rng(seed);
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);

    std::uniform_int_distribution<int> cell_energy_dist(400, 1000);
    std::uniform_int_distribution<int> cargo_dist(0, std::max<energy_type>(0, config.ruleset.economy.max_energy / 2));
    frame.entities.reserve(ships_per_game);
    frame.player_entity_ids.reserve(ships_per_game);
    for (int owner_value = 0; owner_value < 2; ++owner_value) {
        const auto owner = Player::id_type{owner_value};
        const auto player_slot = static_cast<std::size_t>(owner_value);
        frame.players[player_slot].ship_begin = frame.player_entity_ids.size();
        const auto player_ships = ships_per_game / 2 + (static_cast<std::size_t>(owner_value) < ships_per_game % 2 ? 1 : 0);
        for (std::size_t local_index = 0; local_index < player_ships; ++local_index) {
            const auto index = frame.entities.size();
            const auto cell_index_value = (index * 3 + 37) % cells;
            Location location{static_cast<dimension_type>(cell_index_value % frame.width),
                              static_cast<dimension_type>(cell_index_value / frame.width)};
            EntityFrameEntry entity;
            entity.id = Entity::id_type{static_cast<int>(index + 1)};
            entity.owner = owner;
            entity.location = location;
            entity.energy = static_cast<energy_type>(cargo_dist(rng));
            entity.hp = config.ruleset.combat.initial_hp;
            entity.alive = true;
            entity.is_inspired = (index % 5 == 0);
            frame.entity_slot_by_id.emplace(entity.id, frame.entities.size());
            frame.player_entity_ids.push_back(entity.id);
            frame.entities.push_back(entity);
            frame.players[player_slot].ship_count += 1;
            frame.cell_entity[cell_index_value] = entity.id;
            frame.cell_energy[cell_index_value] = static_cast<energy_type>(cell_energy_dist(rng));
            frame.cell_initial_energy[cell_index_value] = frame.cell_energy[cell_index_value];
            frame.map_total_energy += frame.cell_energy[cell_index_value];
        }
    }
    return frame;
}

StateFrame synthetic_economy_frame(unsigned int seed, const GameConfig &config, std::size_t ships_per_game = 256) {
    auto frame = synthetic_pipeline_frame(seed, config, ships_per_game);
    std::mt19937 rng(seed ^ 0x85ebca6bU);
    std::uniform_int_distribution<int> cargo_dist(1, std::max<energy_type>(1, config.ruleset.economy.max_energy / 2));
    std::uniform_int_distribution<int> mine_dist(500, 1000);
    std::size_t neutral_cursor = 0;
    for (std::size_t index = 0; index < frame.entities.size(); ++index) {
        auto &entity = frame.entities[index];
        if (index % 2 == 0) {
            const auto player_slot = static_cast<std::size_t>(entity.owner.value);
            entity.location = (index % 4 == 0) ? frame.players[player_slot].factory : frame.dropoffs[player_slot].location;
            entity.energy = static_cast<energy_type>(cargo_dist(rng));
        } else {
            while (neutral_cursor < frame.cell_owner.size() && frame.cell_owner[neutral_cursor] != Player::None) {
                ++neutral_cursor;
            }
            const auto cell = neutral_cursor++;
            entity.location = Location{static_cast<dimension_type>(cell % frame.width),
                                       static_cast<dimension_type>(cell / frame.width)};
            entity.energy = 0;
            entity.is_inspired = (index % 5 == 0);
            frame.cell_energy[cell] = static_cast<energy_type>(mine_dist(rng));
            frame.cell_initial_energy[cell] = frame.cell_energy[cell];
        }
    }
    frame.cell_entity.assign(frame.cell_entity.size(), Entity::None);
    for (const auto &entity : frame.entities) {
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        frame.cell_entity[cell] = entity.id;
    }
    return frame;
}

StateFrame synthetic_spawn_frame(const GameConfig &config, unsigned long iterations) {
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);
    const auto capacity_per_player = iterations;
    const auto entity_capacity = frame.players.size() * capacity_per_player;
    frame.entities.resize(entity_capacity);
    for (std::size_t player_index = 0; player_index < frame.players.size(); ++player_index) {
        auto &player = frame.players[player_index];
        player.ship_begin = player_index * capacity_per_player;
        player.ship_count = 0;
        player.energy = static_cast<energy_type>(config.ruleset.economy.new_entity_energy_cost * iterations * 4);
        const auto cell = static_cast<std::size_t>(player.factory.y * frame.width + player.factory.x);
        frame.cell_owner[cell] = player.id;
    }
    return frame;
}

StateFrame synthetic_movement_frame(unsigned int seed, const GameConfig &config, std::size_t command_count) {
    std::mt19937 rng(seed);
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);
    std::uniform_int_distribution<int> energy_dist(0, 1000);
    frame.entities.reserve(command_count);
    frame.player_entity_ids.reserve(command_count);
    for (int owner_value = 0; owner_value < 2; ++owner_value) {
        const auto owner = Player::id_type{owner_value};
        const auto player_slot = static_cast<std::size_t>(owner_value);
        frame.players[player_slot].ship_begin = frame.player_entity_ids.size();
        const auto player_ships = command_count / 2 + (static_cast<std::size_t>(owner_value) < command_count % 2 ? 1 : 0);
        for (std::size_t local = 0; local < player_ships; ++local) {
            const auto index = frame.entities.size();
            const auto cell = (index * 5 + 11) % cells;
            EntityFrameEntry entity;
            entity.id = Entity::id_type{static_cast<int>(index + 1)};
            entity.owner = owner;
            entity.location = Location{static_cast<dimension_type>(cell % frame.width),
                                       static_cast<dimension_type>(cell / frame.width)};
            entity.energy = config.ruleset.economy.max_energy;
            entity.hp = config.ruleset.combat.initial_hp;
            entity.alive = true;
            entity.is_inspired = (index % 7 == 0);
            frame.entities.push_back(entity);
            frame.entity_slot_by_id.emplace(entity.id, index);
            frame.player_entity_ids.push_back(entity.id);
            ++frame.players[player_slot].ship_count;
            frame.cell_entity[cell] = entity.id;
            frame.cell_energy[cell] = static_cast<energy_type>(energy_dist(rng));
        }
    }
    return frame;
}

StateFrame synthetic_movement_apply_frame(unsigned int seed, const GameConfig &config, std::size_t command_count) {
    (void)seed;
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);
    const auto capped_commands = std::min<std::size_t>(command_count, 32);
    frame.entities.reserve(capped_commands);
    frame.player_entity_ids.reserve(capped_commands);
    for (std::size_t index = 0; index < capped_commands; ++index) {
        const auto owner = Player::id_type{static_cast<int>(index % 2)};
        const auto player_slot = static_cast<std::size_t>(owner.value);
        if (frame.players[player_slot].ship_count == 0) {
            frame.players[player_slot].ship_begin = frame.player_entity_ids.size();
        }
        const auto y = static_cast<dimension_type>(index);
        const auto x = static_cast<dimension_type>((index % 2) * 16);
        const auto cell = static_cast<std::size_t>(y * frame.width + x);
        EntityFrameEntry entity;
        entity.id = Entity::id_type{static_cast<int>(index + 1)};
        entity.owner = owner;
        entity.location = Location{x, y};
        entity.energy = config.ruleset.economy.max_energy;
        entity.hp = config.ruleset.combat.initial_hp;
        entity.alive = true;
        frame.entities.push_back(entity);
        frame.entity_slot_by_id.emplace(entity.id, index);
        frame.player_entity_ids.push_back(entity.id);
        ++frame.players[player_slot].ship_count;
        frame.cell_entity[cell] = entity.id;
    }
    return frame;
}

CommandFrame synthetic_movement_commands(const StateFrame &frame) {
    CommandFrame commands;
    commands.commands.reserve(frame.entities.size());
    for (std::size_t index = 0; index < frame.entities.size(); ++index) {
        const auto &entity = frame.entities[index];
        FlatCommand command;
        command.player = entity.owner;
        command.entity = entity.id;
        command.type = FlatCommandType::Move;
        switch (index % 4) {
        case 0:
            command.direction = Direction::North;
            break;
        case 1:
            command.direction = Direction::South;
            break;
        case 2:
            command.direction = Direction::East;
            break;
        default:
            command.direction = Direction::West;
            break;
        }
        commands.commands.push_back(command);
    }
    return commands;
}

CommandFrame synthetic_movement_apply_commands(const StateFrame &frame) {
    CommandFrame commands;
    commands.commands.reserve(frame.entities.size());
    for (const auto &entity : frame.entities) {
        FlatCommand command;
        command.player = entity.owner;
        command.entity = entity.id;
        command.type = FlatCommandType::Move;
        command.direction = Direction::East;
        commands.commands.push_back(command);
    }
    return commands;
}

StateFrame synthetic_collision_frame(unsigned int seed, const GameConfig &config, std::size_t command_count) {
    (void)seed;
    StateFrame frame;
    frame.width = 32;
    frame.height = 32;
    frame.turn_number = 1;
    const auto cells = static_cast<std::size_t>(frame.width * frame.height);
    frame.cell_energy.assign(cells, 0);
    frame.cell_initial_energy.assign(cells, 0);
    frame.cell_entity.assign(cells, Entity::None);
    frame.cell_owner.assign(cells, Player::None);
    frame.players.push_back(PlayerFrameEntry{Player::id_type{0}, Location{0, 0}});
    frame.players.push_back(PlayerFrameEntry{Player::id_type{1}, Location{16, 16}});
    frame.player_slot_by_id.emplace(Player::id_type{0}, 0);
    frame.player_slot_by_id.emplace(Player::id_type{1}, 1);

    const auto pair_count = std::max<std::size_t>(1, std::min<std::size_t>(command_count / 2, 256));
    frame.entities.reserve(pair_count * 2);
    frame.player_entity_ids.reserve(pair_count * 2);
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
        const auto target_cell = (pair * 3 + 64) % cells;
        const auto target_x = static_cast<dimension_type>(target_cell % frame.width);
        const auto target_y = static_cast<dimension_type>(target_cell / frame.width);
        const Location starts[2] = {
            Location{(target_x + frame.width - 1) % frame.width, target_y},
            Location{(target_x + 1) % frame.width, target_y}
        };
        for (std::size_t side = 0; side < 2; ++side) {
            const auto index = frame.entities.size();
            const auto owner = Player::id_type{static_cast<int>(side)};
            const auto player_slot = static_cast<std::size_t>(owner.value);
            if (frame.players[player_slot].ship_count == 0) {
                frame.players[player_slot].ship_begin = frame.player_entity_ids.size();
            }
            EntityFrameEntry entity;
            entity.id = Entity::id_type{static_cast<int>(index + 1)};
            entity.owner = owner;
            entity.location = starts[side];
            entity.energy = static_cast<energy_type>(10 + (pair % 13));
            entity.hp = config.ruleset.combat.collision_hp_damage;
            entity.alive = true;
            frame.entities.push_back(entity);
            frame.entity_slot_by_id.emplace(entity.id, index);
            frame.player_entity_ids.push_back(entity.id);
            ++frame.players[player_slot].ship_count;
            const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
            frame.cell_entity[cell] = entity.id;
        }
    }
    return frame;
}

CommandFrame synthetic_collision_commands(const StateFrame &frame) {
    CommandFrame commands;
    commands.commands.reserve(frame.entities.size());
    for (std::size_t index = 0; index < frame.entities.size(); ++index) {
        const auto &entity = frame.entities[index];
        FlatCommand command;
        command.player = entity.owner;
        command.entity = entity.id;
        command.type = FlatCommandType::Move;
        command.direction = (index % 2 == 0) ? Direction::East : Direction::West;
        commands.commands.push_back(command);
    }
    return commands;
}

CommandFrame synthetic_validation_commands(const StateFrame &frame, std::size_t command_count) {
    CommandFrame commands;
    commands.commands.reserve(command_count);
    for (std::size_t index = 0; index < command_count; ++index) {
        FlatCommand command;
        const auto &entity = frame.entities[index % frame.entities.size()];
        command.player = entity.owner;
        command.entity = entity.id;
        switch (index % 6) {
        case 0:
            command.type = FlatCommandType::Move;
            command.direction = Direction::North;
            break;
        case 1:
            command.type = FlatCommandType::Construct;
            break;
        case 2:
            command.type = FlatCommandType::Spawn;
            command.entity = Entity::None;
            break;
        case 3:
            command.type = FlatCommandType::AttackShip;
            command.target = Entity::id_type{9999};
            break;
        case 4:
            command.type = FlatCommandType::Defend;
            break;
        default:
            command.type = FlatCommandType::Heal;
            break;
        }
        commands.commands.push_back(command);
    }
    return commands;
}

void cpu_dump_frame(StateFrame &frame, const GameConfig &config) {
    const int max_hp = config.ruleset.combat.initial_hp;
    for (auto &entity : frame.entities) {
        if (!entity.alive) {
            continue;
        }
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        if (frame.cell_owner[cell] != entity.owner) {
            continue;
        }
        const auto player_slot_it = frame.player_slot_by_id.find(entity.owner);
        if (player_slot_it == frame.player_slot_by_id.end()) {
            continue;
        }
        auto &player = frame.players[player_slot_it->second];
        const auto deposited = entity.energy;
        entity.energy = 0;
        entity.lifetime_deposited += deposited;
        entity.hp = max_hp;
        player.energy += deposited;
        player.total_energy_deposited += deposited;
        if (entity.location == player.factory) {
            player.factory_energy_deposited += deposited;
            if (!player.factory_destroyed) {
                player.factory_halite += deposited;
            }
            continue;
        }
        const auto dropoff_end = player.dropoff_begin + player.dropoff_count;
        for (std::size_t dropoff_index = player.dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
            auto &dropoff = frame.dropoffs[dropoff_index];
            if (!(dropoff.location == entity.location)) {
                continue;
            }
            dropoff.deposited_halite += deposited;
            if (!dropoff.destroyed) {
                dropoff.halite_pool += deposited;
            }
            break;
        }
    }
}

void refill_dump_cargo(StateFrame &frame, energy_type cargo) {
    for (auto &entity : frame.entities) {
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        if (entity.alive && frame.cell_owner[cell] == entity.owner) {
            entity.energy = cargo;
        }
    }
}

void cpu_mining_frame(StateFrame &frame, const GameConfig &config) {
    const auto &economy = config.ruleset.economy;
    for (auto &entity : frame.entities) {
        if (!entity.alive || entity.energy >= economy.max_energy) {
            continue;
        }
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        const auto ratio = entity.is_inspired ? config.ruleset.inspiration.extract_ratio : economy.extract_ratio;
        auto extracted = static_cast<energy_type>(
            std::ceil(static_cast<double>(frame.cell_energy[cell]) / static_cast<double>(ratio)));
        auto gained = extracted;
        if (extracted == 0 && frame.cell_energy[cell] > 0) {
            extracted = frame.cell_energy[cell];
            gained = extracted;
        }
        if (extracted + entity.energy > economy.max_energy) {
            extracted = economy.max_energy - entity.energy;
        }
        if (entity.is_inspired && config.ruleset.inspiration.bonus_multiplier > 0.0) {
            gained += static_cast<energy_type>(config.ruleset.inspiration.bonus_multiplier * static_cast<double>(gained));
        }
        if (economy.max_energy - entity.energy < gained) {
            gained = economy.max_energy - entity.energy;
        }
        if (extracted <= 0 && gained <= 0) {
            continue;
        }
        entity.energy += gained;
        frame.cell_energy[cell] -= extracted;
        frame.map_total_energy -= static_cast<unsigned long long>(extracted);
    }
}

int wrapped_distance_host(const StateFrame &frame, Location a, Location b) {
    int dx = std::abs(static_cast<int>(a.x) - static_cast<int>(b.x));
    int dy = std::abs(static_cast<int>(a.y) - static_cast<int>(b.y));
    dx = std::min(dx, static_cast<int>(frame.width) - dx);
    dy = std::min(dy, static_cast<int>(frame.height) - dy);
    return dx + dy;
}

void cpu_inspiration_frame(StateFrame &frame, const GameConfig &config) {
    const auto &inspiration = config.ruleset.inspiration;
    if (!inspiration.enabled) {
        for (auto &entity : frame.entities) {
            entity.is_inspired = false;
        }
        return;
    }
    for (auto &entity : frame.entities) {
        if (!entity.alive) {
            entity.is_inspired = false;
            continue;
        }
        unsigned long opponents = 0;
        for (int dx = -inspiration.radius; dx <= inspiration.radius; ++dx) {
            for (int dy = -inspiration.radius; dy <= inspiration.radius; ++dy) {
                const int cur_x = (static_cast<int>(entity.location.x) + dx + frame.width) % frame.width;
                const int cur_y = (static_cast<int>(entity.location.y) + dy + frame.height) % frame.height;
                Location current{static_cast<dimension_type>(cur_x), static_cast<dimension_type>(cur_y)};
                if (wrapped_distance_host(frame, entity.location, current) > inspiration.radius) {
                    continue;
                }
                const auto cell = static_cast<std::size_t>(current.y * frame.width + current.x);
                const auto other_id = frame.cell_entity[cell];
                if (other_id == Entity::None) {
                    continue;
                }
                const auto slot = frame.entity_slot_by_id.find(other_id);
                if (slot == frame.entity_slot_by_id.end()) {
                    continue;
                }
                const auto &other = frame.entities[slot->second];
                if (other.alive && other.owner != entity.owner) {
                    ++opponents;
                }
            }
        }
        entity.is_inspired = opponents >= inspiration.ship_count;
    }
}

void refill_mining_inputs(StateFrame &frame, energy_type cell_energy) {
    for (auto &entity : frame.entities) {
        if (!entity.alive) {
            continue;
        }
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        if (frame.cell_owner[cell] != Player::None) {
            continue;
        }
        entity.energy = 0;
        frame.cell_energy[cell] = cell_energy;
    }
}

unsigned long long cpu_spawn_apply_frame(StateFrame &frame,
                                         const GameConfig &config,
                                         unsigned long iterations,
                                         unsigned long game_index) {
    unsigned long long spawned = 0;
    for (auto &player : frame.players) {
        const auto factory_cell = static_cast<std::size_t>(player.factory.y * frame.width + player.factory.x);
        if (frame.cell_entity[factory_cell] != Entity::None) {
            continue;
        }
        for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
            double factor = 1.0;
            if (config.ruleset.economy.spawn_cost_growth > 0.0) {
                factor += config.ruleset.economy.spawn_cost_growth * static_cast<double>(player.ship_count);
            }
            if (config.ruleset.economy.spawn_quad_growth > 0.0 &&
                (player.ship_count + 1) > config.ruleset.economy.spawn_quad_threshold) {
                const auto excess = static_cast<double>((player.ship_count + 1) - config.ruleset.economy.spawn_quad_threshold);
                factor += config.ruleset.economy.spawn_quad_growth * excess * excess;
            }
            const auto cost = static_cast<energy_type>(
                static_cast<double>(config.ruleset.economy.new_entity_energy_cost) * factor + 1.0e-9);
            if (player.energy < cost) {
                break;
            }
            const auto spawn_slot = player.ship_begin + player.ship_count;
            if (spawn_slot >= frame.entities.size() || frame.entities[spawn_slot].alive) {
                break;
            }
            auto &entity = frame.entities[spawn_slot];
            player.energy -= cost;
            entity.id = Entity::id_type{static_cast<int>(game_index * frame.entities.size() + spawn_slot + 1)};
            entity.owner = player.id;
            entity.location = player.factory;
            entity.energy = 0;
            entity.lifetime_deposited = 0;
            entity.enemy_halite_taken = 0;
            entity.enemy_hp_dealt = 0;
            entity.hp = config.ruleset.combat.initial_hp;
            entity.alive = true;
            entity.was_captured = false;
            entity.is_inspired = false;
            entity.is_defending = false;
            entity.protection_turns = 0;
            frame.cell_entity[factory_cell] = entity.id;
            frame.cell_entity[factory_cell] = Entity::None;
            ++player.ship_count;
            ++spawned;
        }
    }
    return spawned;
}

Location moved_location_frame(const StateFrame &frame, Location location, Direction direction) {
    switch (direction) {
    case Direction::North:
        location.y = (location.y + frame.height - 1) % frame.height;
        break;
    case Direction::South:
        location.y = (location.y + 1) % frame.height;
        break;
    case Direction::East:
        location.x = (location.x + 1) % frame.width;
        break;
    case Direction::West:
        location.x = (location.x + frame.width - 1) % frame.width;
        break;
    case Direction::Still:
        break;
    }
    return location;
}

void rebuild_cell_entities_from_entities(StateFrame &frame) {
    frame.cell_entity.assign(frame.cell_entity.size(), Entity::None);
    for (const auto &entity : frame.entities) {
        if (!entity.alive || entity.id == Entity::None) {
            continue;
        }
        const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        if (cell < frame.cell_entity.size()) {
            frame.cell_entity[cell] = entity.id;
        }
    }
}

void pad_frame_layout(StateFrame &frame, std::size_t entity_capacity, std::size_t dropoff_capacity) {
    while (frame.entities.size() < entity_capacity) {
        EntityFrameEntry entity;
        entity.id = Entity::None;
        entity.owner = Player::None;
        entity.alive = false;
        frame.entities.push_back(entity);
    }
    while (frame.dropoffs.size() < dropoff_capacity) {
        DropoffFrameEntry dropoff;
        dropoff.owner = Player::None;
        dropoff.destroyed = true;
        frame.dropoffs.push_back(dropoff);
    }
}

std::vector<gpu::MovementFrameDecision> cpu_movement_decisions(const StateFrame &frame,
                                                               const GameConfig &config,
                                                               const CommandFrame &commands) {
    std::vector<gpu::MovementFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (const auto &command : commands.commands) {
        gpu::MovementFrameDecision decision{command.player,
                                            command.entity,
                                            command.target_location,
                                            command.target_location,
                                            0,
                                            0,
                                            command.type == FlatCommandType::Move && command.direction != Direction::Still,
                                            false,
                                            false};
        if (!decision.command) {
            decisions.push_back(decision);
            continue;
        }
        auto slot = frame.entity_slot_by_id.find(command.entity);
        if (slot == frame.entity_slot_by_id.end()) {
            decision.entity_missing = true;
            decisions.push_back(decision);
            continue;
        }
        const auto &entity = frame.entities[slot->second];
        if (!entity.alive || entity.owner != command.player) {
            decision.entity_missing = true;
            decisions.push_back(decision);
            continue;
        }
        decision.from = entity.location;
        decision.to = moved_location_frame(frame, entity.location, command.direction);
        const auto source = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
        const auto cost_ratio = entity.is_inspired ? config.ruleset.inspiration.move_cost_ratio
                                                   : config.ruleset.economy.move_cost_ratio;
        decision.required = cost_ratio == 0 ? 0 : frame.cell_energy[source] / cost_ratio;
        decision.current_energy = entity.energy;
        decision.insufficient_energy = entity.energy < decision.required;
        decisions.push_back(decision);
    }
    return decisions;
}

unsigned long long cpu_movement_apply_no_collision(StateFrame &frame,
                                                   const GameConfig &config,
                                                   const CommandFrame &commands,
                                                   unsigned long iterations) {
    unsigned long long applied = 0;
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        for (const auto &command : commands.commands) {
            if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
                continue;
            }
            auto slot = frame.entity_slot_by_id.find(command.entity);
            if (slot == frame.entity_slot_by_id.end()) {
                continue;
            }
            auto &entity = frame.entities[slot->second];
            if (!entity.alive || entity.owner != command.player) {
                continue;
            }
            const auto source = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
            const auto cost_ratio = entity.is_inspired ? config.ruleset.inspiration.move_cost_ratio
                                                       : config.ruleset.economy.move_cost_ratio;
            const auto required = static_cast<energy_type>(cost_ratio == 0 ? 0 : frame.cell_energy[source] / cost_ratio);
            if (entity.energy < required) {
                continue;
            }
            const auto destination = moved_location_frame(frame, entity.location, command.direction);
            const auto destination_index = static_cast<std::size_t>(destination.y * frame.width + destination.x);
            entity.energy -= required;
            frame.cell_entity[source] = Entity::None;
            entity.location = destination;
            frame.cell_entity[destination_index] = entity.id;
            ++applied;
        }
    }
    rebuild_cell_entities_from_entities(frame);
    return applied;
}

gpu::BatchedDestinationStats cpu_destination_counts(const std::vector<StateFrame> &frames,
                                                    const GameConfig &config,
                                                    const std::vector<CommandFrame> &commands,
                                                    unsigned long iterations) {
    gpu::BatchedDestinationStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    stats.commands_per_game = commands.front().commands.size();
    for (std::size_t game = 0; game < frames.size(); ++game) {
        const auto &frame = frames[game];
        std::vector<unsigned int> destination_counts(frame.cell_energy.size(), 0U);
        unsigned long long moved = 0;
        for (const auto &command : commands[game].commands) {
            if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
                continue;
            }
            auto slot = frame.entity_slot_by_id.find(command.entity);
            if (slot == frame.entity_slot_by_id.end()) {
                continue;
            }
            const auto &entity = frame.entities[slot->second];
            if (!entity.alive || entity.owner != command.player) {
                continue;
            }
            const auto source = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
            const auto cost_ratio = entity.is_inspired ? config.ruleset.inspiration.move_cost_ratio
                                                       : config.ruleset.economy.move_cost_ratio;
            const auto required = static_cast<energy_type>(cost_ratio == 0 ? 0 : frame.cell_energy[source] / cost_ratio);
            if (entity.energy < required) {
                continue;
            }
            const auto destination = moved_location_frame(frame, entity.location, command.direction);
            const auto destination_index = static_cast<std::size_t>(destination.y * frame.width + destination.x);
            destination_counts[destination_index] += static_cast<unsigned int>(iterations);
            moved += iterations;
        }
        unsigned long long conflict_cells = 0;
        unsigned long long max_arrivals = 0;
        unsigned long long total_arrivals = 0;
        for (const auto count : destination_counts) {
            total_arrivals += count;
            max_arrivals = std::max(max_arrivals, static_cast<unsigned long long>(count));
            if (count > 1U) {
                ++conflict_cells;
            }
        }
        stats.moved_commands += moved;
        stats.conflict_cells += conflict_cells;
        stats.total_arrivals += total_arrivals;
        stats.max_arrivals = std::max(stats.max_arrivals, max_arrivals);
    }
    return stats;
}

gpu::BatchedMovementCollisionStats cpu_movement_collision_apply(std::vector<StateFrame> &frames,
                                                                const GameConfig &config,
                                                                const std::vector<CommandFrame> &commands,
                                                                unsigned long iterations) {
    gpu::BatchedMovementCollisionStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    stats.commands_per_game = commands.front().commands.size();
    for (std::size_t game = 0; game < frames.size(); ++game) {
        auto &frame = frames[game];
        for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
            std::vector<unsigned int> destination_counts(frame.cell_energy.size(), 0U);
            for (const auto &command : commands[game].commands) {
                if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
                    continue;
                }
                auto slot = frame.entity_slot_by_id.find(command.entity);
                if (slot == frame.entity_slot_by_id.end()) {
                    continue;
                }
                auto &entity = frame.entities[slot->second];
                if (!entity.alive || entity.owner != command.player) {
                    continue;
                }
                const auto source = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
                const auto cost_ratio = entity.is_inspired ? config.ruleset.inspiration.move_cost_ratio
                                                           : config.ruleset.economy.move_cost_ratio;
                const auto required = static_cast<energy_type>(cost_ratio == 0 ? 0 : frame.cell_energy[source] / cost_ratio);
                if (entity.energy < required) {
                    continue;
                }
                const auto destination = moved_location_frame(frame, entity.location, command.direction);
                const auto destination_index = static_cast<std::size_t>(destination.y * frame.width + destination.x);
                entity.energy -= required;
                entity.location = destination;
                frame.cell_entity[source] = Entity::None;
                ++destination_counts[destination_index];
                ++stats.moved_commands;
            }
            for (const auto count : destination_counts) {
                if (count > 1U) {
                    ++stats.collision_cells;
                }
            }
            for (auto &entity : frame.entities) {
                if (!entity.alive || entity.id == Entity::None) {
                    continue;
                }
                const auto cell = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
                if (destination_counts[cell] <= 1U) {
                    continue;
                }
                ++stats.damaged_entities;
                entity.hp -= config.ruleset.combat.collision_hp_damage;
                if (entity.hp <= 0) {
                    entity.alive = false;
                    frame.cell_energy[cell] += entity.energy;
                    stats.dropped_energy += static_cast<unsigned long long>(entity.energy);
                    entity.energy = 0;
                    ++stats.deaths;
                }
            }
            rebuild_cell_entities_from_entities(frame);
        }
    }
    return stats;
}

std::vector<gpu::ValidationFrameDecision> cpu_validation_decisions(const StateFrame &frame,
                                                                   const GameConfig &config,
                                                                   const CommandFrame &commands) {
    std::vector<gpu::ValidationFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (const auto &command : commands.commands) {
        gpu::ValidationFrameDecision decision{command.player,
                                              command.entity,
                                              command.type,
                                              0,
                                              true,
                                              false,
                                              false,
                                              config.ruleset.combat.enable_combat_commands,
                                              false,
                                              false,
                                              false};
        auto player_slot = frame.player_slot_by_id.find(command.player);
        if (player_slot == frame.player_slot_by_id.end()) {
            decisions.push_back(decision);
            continue;
        }
        decision.player_exists = true;
        const auto &player = frame.players[player_slot->second];
        auto entity_slot = frame.entity_slot_by_id.find(command.entity);
        const bool owns_entity = entity_slot != frame.entity_slot_by_id.end() &&
            frame.entities[entity_slot->second].alive &&
            frame.entities[entity_slot->second].owner == command.player;

        if (command.type == FlatCommandType::Move) {
            decision.ownership_ok = owns_entity;
            decision.occurrence_command = owns_entity;
            decision.include_in_batch = owns_entity;
        } else if (command.type == FlatCommandType::Construct) {
            decision.ownership_ok = owns_entity;
            decision.occurrence_command = owns_entity;
            decision.expense_command = owns_entity;
            decision.include_in_batch = owns_entity;
            if (owns_entity) {
                const auto &entity = frame.entities[entity_slot->second];
                auto cost = hlt::rules::phases::scaled_dropoff_cost(
                    config.ruleset.economy.dropoff_cost,
                    player.dropoff_count,
                    config.ruleset.economy.dropoff_cost_growth);
                const auto location = static_cast<std::size_t>(entity.location.y * frame.width + entity.location.x);
                const auto credit = frame.cell_energy[location] + entity.energy;
                decision.expense = credit >= cost ? 0 : cost - credit;
            }
        } else if (command.type == FlatCommandType::Spawn) {
            decision.ownership_ok = true;
            decision.expense_command = true;
            decision.include_in_batch = true;
            double factor = 1.0;
            if (config.ruleset.economy.spawn_cost_growth > 0.0) {
                factor += config.ruleset.economy.spawn_cost_growth * static_cast<double>(player.ship_count);
            }
            if (config.ruleset.economy.spawn_quad_growth > 0.0 &&
                (player.ship_count + 1) > config.ruleset.economy.spawn_quad_threshold) {
                const auto excess = static_cast<double>((player.ship_count + 1) - config.ruleset.economy.spawn_quad_threshold);
                factor += config.ruleset.economy.spawn_quad_growth * excess * excess;
            }
            decision.expense = static_cast<energy_type>(
                static_cast<double>(config.ruleset.economy.new_entity_energy_cost) * factor + 1.0e-9);
        } else if (command.type == FlatCommandType::AttackShip ||
                   command.type == FlatCommandType::AttackStructure ||
                   command.type == FlatCommandType::Defend ||
                   command.type == FlatCommandType::Heal) {
            decision.ownership_ok = owns_entity;
            decision.occurrence_command = config.ruleset.combat.enable_combat_commands && owns_entity;
            decision.include_in_batch = decision.occurrence_command;
        }
        decisions.push_back(decision);
    }
    return decisions;
}

bool dump_entity_equal(const EntityFrameEntry &lhs, const EntityFrameEntry &rhs) {
    return lhs.id == rhs.id &&
           lhs.owner == rhs.owner &&
           lhs.location == rhs.location &&
           lhs.energy == rhs.energy &&
           lhs.lifetime_deposited == rhs.lifetime_deposited &&
           lhs.hp == rhs.hp &&
           lhs.alive == rhs.alive;
}

bool dump_player_equal(const PlayerFrameEntry &lhs, const PlayerFrameEntry &rhs) {
    return lhs.id == rhs.id &&
           lhs.factory == rhs.factory &&
           lhs.energy == rhs.energy &&
           lhs.factory_energy_deposited == rhs.factory_energy_deposited &&
           lhs.total_energy_deposited == rhs.total_energy_deposited &&
           lhs.factory_halite == rhs.factory_halite &&
           lhs.factory_destroyed == rhs.factory_destroyed &&
           lhs.ship_begin == rhs.ship_begin &&
           lhs.ship_count == rhs.ship_count &&
           lhs.dropoff_begin == rhs.dropoff_begin &&
           lhs.dropoff_count == rhs.dropoff_count;
}

bool dump_dropoff_equal(const DropoffFrameEntry &lhs, const DropoffFrameEntry &rhs) {
    return lhs.owner == rhs.owner &&
           lhs.location == rhs.location &&
           lhs.deposited_halite == rhs.deposited_halite &&
           lhs.halite_pool == rhs.halite_pool &&
           lhs.destroyed == rhs.destroyed;
}

bool dump_frame_equal(const StateFrame &lhs, const StateFrame &rhs) {
    if (lhs.entities.size() != rhs.entities.size() ||
        lhs.players.size() != rhs.players.size() ||
        lhs.dropoffs.size() != rhs.dropoffs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.entities.size(); ++index) {
        if (!dump_entity_equal(lhs.entities[index], rhs.entities[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.players.size(); ++index) {
        if (!dump_player_equal(lhs.players[index], rhs.players[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.dropoffs.size(); ++index) {
        if (!dump_dropoff_equal(lhs.dropoffs[index], rhs.dropoffs[index])) {
            return false;
        }
    }
    return true;
}

bool mining_frame_equal(const StateFrame &lhs, const StateFrame &rhs) {
    if (lhs.map_total_energy != rhs.map_total_energy ||
        lhs.cell_energy != rhs.cell_energy ||
        lhs.entities.size() != rhs.entities.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.entities.size(); ++index) {
        if (!dump_entity_equal(lhs.entities[index], rhs.entities[index])) {
            return false;
        }
    }
    return true;
}

bool economy_frame_equal(const StateFrame &lhs, const StateFrame &rhs) {
    if (lhs.map_total_energy != rhs.map_total_energy ||
        lhs.cell_energy != rhs.cell_energy ||
        lhs.entities.size() != rhs.entities.size() ||
        lhs.players.size() != rhs.players.size() ||
        lhs.dropoffs.size() != rhs.dropoffs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.entities.size(); ++index) {
        if (!dump_entity_equal(lhs.entities[index], rhs.entities[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.players.size(); ++index) {
        if (!dump_player_equal(lhs.players[index], rhs.players[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.dropoffs.size(); ++index) {
        if (!dump_dropoff_equal(lhs.dropoffs[index], rhs.dropoffs[index])) {
            return false;
        }
    }
    return true;
}

bool spawn_frame_equal(const StateFrame &lhs, const StateFrame &rhs) {
    if (lhs.cell_entity != rhs.cell_entity ||
        lhs.players.size() != rhs.players.size() ||
        lhs.entities.size() != rhs.entities.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.players.size(); ++index) {
        if (!dump_player_equal(lhs.players[index], rhs.players[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.entities.size(); ++index) {
        if (!dump_entity_equal(lhs.entities[index], rhs.entities[index])) {
            return false;
        }
    }
    return true;
}

bool movement_decision_equal(const gpu::MovementFrameDecision &lhs, const gpu::MovementFrameDecision &rhs) {
    return lhs.player == rhs.player &&
           lhs.entity == rhs.entity &&
           lhs.from == rhs.from &&
           lhs.to == rhs.to &&
           lhs.required == rhs.required &&
           lhs.current_energy == rhs.current_energy &&
           lhs.command == rhs.command &&
           lhs.entity_missing == rhs.entity_missing &&
           lhs.insufficient_energy == rhs.insufficient_energy;
}

bool movement_apply_frame_equal(const StateFrame &lhs, const StateFrame &rhs) {
    if (lhs.cell_entity != rhs.cell_entity || lhs.entities.size() != rhs.entities.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.entities.size(); ++index) {
        if (!dump_entity_equal(lhs.entities[index], rhs.entities[index])) {
            return false;
        }
    }
    return true;
}

bool validation_decision_equal(const gpu::ValidationFrameDecision &lhs, const gpu::ValidationFrameDecision &rhs) {
    return lhs.player == rhs.player &&
           lhs.entity == rhs.entity &&
           lhs.type == rhs.type &&
           lhs.expense == rhs.expense &&
           lhs.command == rhs.command &&
           lhs.player_exists == rhs.player_exists &&
           lhs.ownership_ok == rhs.ownership_ok &&
           lhs.combat_enabled == rhs.combat_enabled &&
           lhs.occurrence_command == rhs.occurrence_command &&
           lhs.expense_command == rhs.expense_command &&
           lhs.include_in_batch == rhs.include_in_batch;
}

int run_batched_regen_benchmark(const Options &options) {
    auto config = load_config(options);
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 0.90;
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_regen_frame(options.seed_base + static_cast<unsigned int>(index), config);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.regen_iterations; ++iteration) {
        for (auto &frame : cpu_frames) {
            cpu_regen_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_regen_iterations(gpu_ptrs, config, options.regen_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (cpu_frames[game].map_total_energy != gpu_frames[game].map_total_energy ||
            cpu_frames[game].cell_energy != gpu_frames[game].cell_energy) {
            ++mismatches;
        }
    }

    const auto total_cells = options.games * stats.cells_per_game;
    const auto total_cell_iterations = total_cells * options.regen_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_regen_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.regen_iterations},
        {"cells_per_game", stats.cells_per_game},
        {"total_cells", total_cells},
        {"total_cell_iterations", total_cell_iterations},
        {"mismatches", mismatches},
        {"total_delta", stats.total_delta},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_cells_per_second", static_cast<double>(total_cell_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_cells_per_second", static_cast<double>(total_cell_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_regen games=" << options.games
              << " iterations=" << options.regen_iterations
              << " mismatches=" << mismatches
              << " gpu_cells/s=" << report["gpu_cells_per_second"].get<double>()
              << " cpu_cells/s=" << report["cpu_cells_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_dump_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    constexpr std::size_t ships_per_game = 256;
    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_dump_frame(options.seed_base + static_cast<unsigned int>(index), config, ships_per_game);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.dump_iterations; ++iteration) {
        if (iteration > 0) {
            for (auto &frame : cpu_frames) {
                refill_dump_cargo(frame, options.dump_refill_cargo);
            }
        }
        for (auto &frame : cpu_frames) {
            cpu_dump_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_dump(gpu_ptrs, config, options.dump_iterations, options.dump_refill_cargo);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (!dump_frame_equal(cpu_frames[game], gpu_frames[game])) {
            ++mismatches;
        }
    }

    const auto total_entities = options.games * stats.entities_per_game;
    const auto total_entity_iterations = total_entities * options.dump_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_dump_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.dump_iterations},
        {"refill_cargo", options.dump_refill_cargo},
        {"entities_per_game", stats.entities_per_game},
        {"total_entities", total_entities},
        {"total_entity_iterations", total_entity_iterations},
        {"mismatches", mismatches},
        {"total_deposited", stats.total_deposited},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_entities_per_second", static_cast<double>(total_entity_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_entities_per_second", static_cast<double>(total_entity_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_dump games=" << options.games
              << " iterations=" << options.dump_iterations
              << " mismatches=" << mismatches
              << " gpu_entities/s=" << report["gpu_entities_per_second"].get<double>()
              << " cpu_entities/s=" << report["cpu_entities_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_mining_benchmark(const Options &options) {
    auto config = load_config(options);
    config.ruleset.economy.mining_interference_range = 0;
    config.apply_to_global_constants();

    constexpr std::size_t ships_per_game = 256;
    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_mining_frame(options.seed_base + static_cast<unsigned int>(index), config, ships_per_game);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.regen_iterations; ++iteration) {
        for (auto &frame : cpu_frames) {
            cpu_mining_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_mining(gpu_ptrs, config, options.regen_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (!mining_frame_equal(cpu_frames[game], gpu_frames[game])) {
            ++mismatches;
        }
    }

    const auto total_entities = options.games * stats.entities_per_game;
    const auto total_entity_iterations = total_entities * options.regen_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_mining_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.regen_iterations},
        {"cells_per_game", stats.cells_per_game},
        {"entities_per_game", stats.entities_per_game},
        {"total_entities", total_entities},
        {"total_entity_iterations", total_entity_iterations},
        {"mismatches", mismatches},
        {"total_extracted", stats.total_extracted},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_entities_per_second", static_cast<double>(total_entity_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_entities_per_second", static_cast<double>(total_entity_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_mining games=" << options.games
              << " iterations=" << options.regen_iterations
              << " mismatches=" << mismatches
              << " gpu_entities/s=" << report["gpu_entities_per_second"].get<double>()
              << " cpu_entities/s=" << report["cpu_entities_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_spawn_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        (void)index;
        auto frame = synthetic_spawn_frame(config, options.spawn_iterations);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    unsigned long long cpu_spawned = 0;
    const auto cpu_start = Clock::now();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        cpu_spawned += cpu_spawn_apply_frame(cpu_frames[game], config, options.spawn_iterations, static_cast<unsigned long>(game));
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_spawn_apply(gpu_ptrs, config, options.spawn_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (!spawn_frame_equal(cpu_frames[game], gpu_frames[game])) {
            if (first_mismatch.empty()) {
                first_mismatch["game"] = game;
                first_mismatch["cpu_players"] = nlohmann::json::array();
                first_mismatch["gpu_players"] = nlohmann::json::array();
                for (std::size_t player = 0; player < cpu_frames[game].players.size(); ++player) {
                    const auto &cpu = cpu_frames[game].players[player];
                    const auto &gpu = gpu_frames[game].players[player];
                    first_mismatch["cpu_players"].push_back({{"id", cpu.id.value}, {"energy", cpu.energy}, {"ship_count", cpu.ship_count}, {"ship_begin", cpu.ship_begin}});
                    first_mismatch["gpu_players"].push_back({{"id", gpu.id.value}, {"energy", gpu.energy}, {"ship_count", gpu.ship_count}, {"ship_begin", gpu.ship_begin}});
                }
                for (std::size_t entity = 0; entity < cpu_frames[game].entities.size(); ++entity) {
                    if (!dump_entity_equal(cpu_frames[game].entities[entity], gpu_frames[game].entities[entity])) {
                        const auto &cpu = cpu_frames[game].entities[entity];
                        const auto &gpu = gpu_frames[game].entities[entity];
                        first_mismatch["entity_index"] = entity;
                        first_mismatch["cpu_entity"] = {{"id", cpu.id.value}, {"owner", cpu.owner.value}, {"x", cpu.location.x}, {"y", cpu.location.y}, {"hp", cpu.hp}, {"alive", cpu.alive}};
                        first_mismatch["gpu_entity"] = {{"id", gpu.id.value}, {"owner", gpu.owner.value}, {"x", gpu.location.x}, {"y", gpu.location.y}, {"hp", gpu.hp}, {"alive", gpu.alive}};
                        break;
                    }
                }
            }
            ++mismatches;
        }
    }

    const auto total_player_iterations = options.games * stats.players_per_game * options.spawn_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_spawn_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.spawn_iterations},
        {"players_per_game", stats.players_per_game},
        {"entity_capacity_per_game", stats.entity_capacity_per_game},
        {"total_player_iterations", total_player_iterations},
        {"mismatches", mismatches},
        {"cpu_spawned", cpu_spawned},
        {"gpu_spawned", stats.total_spawned},
        {"first_mismatch", first_mismatch},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_player_iterations_per_second", static_cast<double>(total_player_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_player_iterations_per_second", static_cast<double>(total_player_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_spawn games=" << options.games
              << " iterations=" << options.spawn_iterations
              << " mismatches=" << mismatches
              << " gpu_player_iters/s=" << report["gpu_player_iterations_per_second"].get<double>()
              << " cpu_player_iters/s=" << report["cpu_player_iterations_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_movement_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    std::vector<CommandFrame> command_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    command_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_movement_apply_frame(options.seed_base + static_cast<unsigned int>(index), config, options.movement_commands);
        command_frames.push_back(synthetic_movement_apply_commands(frame));
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    std::vector<gpu::MovementFrameDecision> cpu_decisions;
    cpu_decisions.reserve(options.games * options.movement_commands);
    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.movement_iterations; ++iteration) {
        cpu_decisions.clear();
        for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
            auto decisions = cpu_movement_decisions(cpu_frames[game], config, command_frames[game]);
            cpu_decisions.insert(cpu_decisions.end(), decisions.begin(), decisions.end());
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    std::vector<gpu::MovementFrameDecision> gpu_decisions;
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_movement_decisions(gpu_ptrs,
                                                          config,
                                                          command_frames,
                                                          gpu_decisions,
                                                          options.movement_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    const auto count = std::min(cpu_decisions.size(), gpu_decisions.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (!movement_decision_equal(cpu_decisions[index], gpu_decisions[index])) {
            if (first_mismatch.empty()) {
                const auto &cpu = cpu_decisions[index];
                const auto &gpu = gpu_decisions[index];
                first_mismatch = {
                    {"index", index},
                    {"cpu", {{"player", cpu.player.value}, {"entity", cpu.entity.value}, {"from_x", cpu.from.x}, {"from_y", cpu.from.y}, {"to_x", cpu.to.x}, {"to_y", cpu.to.y}, {"required", cpu.required}, {"energy", cpu.current_energy}, {"missing", cpu.entity_missing}, {"insufficient", cpu.insufficient_energy}}},
                    {"gpu", {{"player", gpu.player.value}, {"entity", gpu.entity.value}, {"from_x", gpu.from.x}, {"from_y", gpu.from.y}, {"to_x", gpu.to.x}, {"to_y", gpu.to.y}, {"required", gpu.required}, {"energy", gpu.current_energy}, {"missing", gpu.entity_missing}, {"insufficient", gpu.insufficient_energy}}}
                };
            }
            ++mismatches;
        }
    }
    mismatches += cpu_decisions.size() > gpu_decisions.size()
        ? cpu_decisions.size() - gpu_decisions.size()
        : gpu_decisions.size() - cpu_decisions.size();

    const auto total_commands = options.games * stats.commands_per_game;
    const auto total_command_iterations = total_commands * options.movement_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_movement_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.movement_iterations},
        {"commands_per_game", stats.commands_per_game},
        {"total_commands", total_commands},
        {"total_command_iterations", total_command_iterations},
        {"mismatches", mismatches},
        {"first_mismatch", first_mismatch},
        {"movable_commands", stats.movable_commands},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_movement games=" << options.games
              << " commands=" << stats.commands_per_game
              << " iterations=" << options.movement_iterations
              << " mismatches=" << mismatches
              << " gpu_cmd/s=" << report["gpu_commands_per_second"].get<double>()
              << " cpu_cmd/s=" << report["cpu_commands_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_movement_apply_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    std::vector<CommandFrame> command_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    command_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_movement_apply_frame(options.seed_base + static_cast<unsigned int>(index), config, options.movement_commands);
        command_frames.push_back(synthetic_movement_apply_commands(frame));
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    unsigned long long cpu_applied = 0;
    const auto cpu_start = Clock::now();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        cpu_applied += cpu_movement_apply_no_collision(cpu_frames[game],
                                                       config,
                                                       command_frames[game],
                                                       options.movement_iterations);
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_movement_apply_no_collision(gpu_ptrs,
                                                                   config,
                                                                   command_frames,
                                                                   options.movement_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (!movement_apply_frame_equal(cpu_frames[game], gpu_frames[game])) {
            if (first_mismatch.empty()) {
                first_mismatch["game"] = game;
                for (std::size_t entity = 0; entity < cpu_frames[game].entities.size(); ++entity) {
                    if (!dump_entity_equal(cpu_frames[game].entities[entity], gpu_frames[game].entities[entity])) {
                        const auto &cpu = cpu_frames[game].entities[entity];
                        const auto &gpu = gpu_frames[game].entities[entity];
                        first_mismatch["entity_index"] = entity;
                        first_mismatch["cpu"] = {{"id", cpu.id.value}, {"x", cpu.location.x}, {"y", cpu.location.y}, {"energy", cpu.energy}, {"alive", cpu.alive}};
                        first_mismatch["gpu"] = {{"id", gpu.id.value}, {"x", gpu.location.x}, {"y", gpu.location.y}, {"energy", gpu.energy}, {"alive", gpu.alive}};
                        break;
                    }
                }
                for (std::size_t cell = 0; cell < cpu_frames[game].cell_entity.size(); ++cell) {
                    if (cpu_frames[game].cell_entity[cell] != gpu_frames[game].cell_entity[cell]) {
                        first_mismatch["cell_index"] = cell;
                        first_mismatch["cpu_cell_entity"] = cpu_frames[game].cell_entity[cell].value;
                        first_mismatch["gpu_cell_entity"] = gpu_frames[game].cell_entity[cell].value;
                        const auto probe_id = gpu_frames[game].cell_entity[cell] != Entity::None
                                                  ? gpu_frames[game].cell_entity[cell]
                                                  : cpu_frames[game].cell_entity[cell];
                        first_mismatch["probe_entity"] = probe_id.value;
                        for (std::size_t entity = 0; entity < cpu_frames[game].entities.size(); ++entity) {
                            if (cpu_frames[game].entities[entity].id == probe_id) {
                                const auto &cpu = cpu_frames[game].entities[entity];
                                const auto &gpu = gpu_frames[game].entities[entity];
                                first_mismatch["probe_cpu_entity"] = {
                                    {"index", entity},
                                    {"x", cpu.location.x},
                                    {"y", cpu.location.y},
                                    {"energy", cpu.energy},
                                    {"alive", cpu.alive}};
                                first_mismatch["probe_gpu_entity"] = {
                                    {"index", entity},
                                    {"x", gpu.location.x},
                                    {"y", gpu.location.y},
                                    {"energy", gpu.energy},
                                    {"alive", gpu.alive}};
                                break;
                            }
                        }
                        for (std::size_t probe_cell = 0; probe_cell < cpu_frames[game].cell_entity.size(); ++probe_cell) {
                            if (cpu_frames[game].cell_entity[probe_cell] == probe_id) {
                                first_mismatch["probe_cpu_cell"] = probe_cell;
                                break;
                            }
                        }
                        for (std::size_t probe_cell = 0; probe_cell < gpu_frames[game].cell_entity.size(); ++probe_cell) {
                            if (gpu_frames[game].cell_entity[probe_cell] == probe_id) {
                                first_mismatch["probe_gpu_cell"] = probe_cell;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            ++mismatches;
        }
    }

    const auto total_command_iterations = options.games * stats.commands_per_game * options.movement_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_movement_apply_no_collision_benchmark_v1"},
        {"games", options.games},
        {"commands_per_game", stats.commands_per_game},
        {"iterations", options.movement_iterations},
        {"total_command_iterations", total_command_iterations},
        {"mismatches", mismatches},
        {"first_mismatch", first_mismatch},
        {"cpu_applied", cpu_applied},
        {"gpu_applied", stats.applied_commands},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_movement_apply games=" << options.games
              << " commands=" << stats.commands_per_game
              << " iterations=" << options.movement_iterations
              << " mismatches=" << mismatches
              << " gpu_cmd/s=" << report["gpu_commands_per_second"].get<double>()
              << " cpu_cmd/s=" << report["cpu_commands_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_destination_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    std::vector<CommandFrame> command_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    command_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_movement_frame(options.seed_base + static_cast<unsigned int>(index), config, options.movement_commands);
        command_frames.push_back(synthetic_movement_commands(frame));
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    const auto cpu_stats = cpu_destination_counts(cpu_frames, config, command_frames, options.movement_iterations);
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    const auto gpu_stats = gpu::run_cuda_batched_destination_counts(gpu_ptrs,
                                                                    config,
                                                                    command_frames,
                                                                    options.movement_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    const bool mismatch = cpu_stats.moved_commands != gpu_stats.moved_commands ||
                          cpu_stats.conflict_cells != gpu_stats.conflict_cells ||
                          cpu_stats.max_arrivals != gpu_stats.max_arrivals ||
                          cpu_stats.total_arrivals != gpu_stats.total_arrivals;
    nlohmann::json first_mismatch = nlohmann::json::object();
    if (mismatch) {
        first_mismatch["cpu_moved"] = cpu_stats.moved_commands;
        first_mismatch["gpu_moved"] = gpu_stats.moved_commands;
        first_mismatch["cpu_conflict_cells"] = cpu_stats.conflict_cells;
        first_mismatch["gpu_conflict_cells"] = gpu_stats.conflict_cells;
        first_mismatch["cpu_max_arrivals"] = cpu_stats.max_arrivals;
        first_mismatch["gpu_max_arrivals"] = gpu_stats.max_arrivals;
        first_mismatch["cpu_total_arrivals"] = cpu_stats.total_arrivals;
        first_mismatch["gpu_total_arrivals"] = gpu_stats.total_arrivals;
    }

    const auto total_command_iterations = options.games * gpu_stats.commands_per_game * options.movement_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_destination_count_benchmark_v1"},
        {"games", options.games},
        {"commands_per_game", gpu_stats.commands_per_game},
        {"iterations", options.movement_iterations},
        {"total_command_iterations", total_command_iterations},
        {"mismatches", mismatch ? 1 : 0},
        {"first_mismatch", first_mismatch},
        {"cpu_moved", cpu_stats.moved_commands},
        {"gpu_moved", gpu_stats.moved_commands},
        {"cpu_conflict_cells", cpu_stats.conflict_cells},
        {"gpu_conflict_cells", gpu_stats.conflict_cells},
        {"cpu_max_arrivals", cpu_stats.max_arrivals},
        {"gpu_max_arrivals", gpu_stats.max_arrivals},
        {"cpu_total_arrivals", cpu_stats.total_arrivals},
        {"gpu_total_arrivals", gpu_stats.total_arrivals},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_destination games=" << options.games
              << " commands=" << gpu_stats.commands_per_game
              << " iterations=" << options.movement_iterations
              << " mismatches=" << (mismatch ? 1 : 0)
              << " gpu_cmd/s=" << report["gpu_commands_per_second"].get<double>()
              << " cpu_cmd/s=" << report["cpu_commands_per_second"].get<double>() << "\n";
    return mismatch ? 1 : 0;
}

int run_batched_collision_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    std::vector<CommandFrame> command_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    command_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_collision_frame(options.seed_base + static_cast<unsigned int>(index), config, options.movement_commands);
        command_frames.push_back(synthetic_collision_commands(frame));
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    const auto cpu_stats = cpu_movement_collision_apply(cpu_frames, config, command_frames, options.movement_iterations);
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    const auto gpu_stats = gpu::run_cuda_batched_movement_collision_apply(gpu_ptrs,
                                                                          config,
                                                                          command_frames,
                                                                          options.movement_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t frame_mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (cpu_frames[game].cell_energy != gpu_frames[game].cell_energy ||
            !movement_apply_frame_equal(cpu_frames[game], gpu_frames[game])) {
            if (first_mismatch.empty()) {
                first_mismatch["game"] = game;
                for (std::size_t entity = 0; entity < cpu_frames[game].entities.size(); ++entity) {
                    if (!dump_entity_equal(cpu_frames[game].entities[entity], gpu_frames[game].entities[entity])) {
                        const auto &cpu = cpu_frames[game].entities[entity];
                        const auto &gpu = gpu_frames[game].entities[entity];
                        first_mismatch["entity_index"] = entity;
                        first_mismatch["cpu"] = {{"id", cpu.id.value}, {"x", cpu.location.x}, {"y", cpu.location.y}, {"energy", cpu.energy}, {"hp", cpu.hp}, {"alive", cpu.alive}};
                        first_mismatch["gpu"] = {{"id", gpu.id.value}, {"x", gpu.location.x}, {"y", gpu.location.y}, {"energy", gpu.energy}, {"hp", gpu.hp}, {"alive", gpu.alive}};
                        break;
                    }
                }
                for (std::size_t cell = 0; cell < cpu_frames[game].cell_energy.size(); ++cell) {
                    if (cpu_frames[game].cell_energy[cell] != gpu_frames[game].cell_energy[cell] ||
                        cpu_frames[game].cell_entity[cell] != gpu_frames[game].cell_entity[cell]) {
                        first_mismatch["cell_index"] = cell;
                        first_mismatch["cpu_cell_energy"] = cpu_frames[game].cell_energy[cell];
                        first_mismatch["gpu_cell_energy"] = gpu_frames[game].cell_energy[cell];
                        first_mismatch["cpu_cell_entity"] = cpu_frames[game].cell_entity[cell].value;
                        first_mismatch["gpu_cell_entity"] = gpu_frames[game].cell_entity[cell].value;
                        break;
                    }
                }
            }
            ++frame_mismatches;
        }
    }
    const bool stats_mismatch = cpu_stats.moved_commands != gpu_stats.moved_commands ||
                                cpu_stats.collision_cells != gpu_stats.collision_cells ||
                                cpu_stats.damaged_entities != gpu_stats.damaged_entities ||
                                cpu_stats.deaths != gpu_stats.deaths ||
                                cpu_stats.dropped_energy != gpu_stats.dropped_energy;
    const auto mismatches = frame_mismatches + (stats_mismatch ? 1 : 0);
    if (stats_mismatch && first_mismatch.empty()) {
        first_mismatch["cpu_moved"] = cpu_stats.moved_commands;
        first_mismatch["gpu_moved"] = gpu_stats.moved_commands;
        first_mismatch["cpu_collision_cells"] = cpu_stats.collision_cells;
        first_mismatch["gpu_collision_cells"] = gpu_stats.collision_cells;
        first_mismatch["cpu_damaged"] = cpu_stats.damaged_entities;
        first_mismatch["gpu_damaged"] = gpu_stats.damaged_entities;
        first_mismatch["cpu_deaths"] = cpu_stats.deaths;
        first_mismatch["gpu_deaths"] = gpu_stats.deaths;
        first_mismatch["cpu_dropped"] = cpu_stats.dropped_energy;
        first_mismatch["gpu_dropped"] = gpu_stats.dropped_energy;
    }

    const auto total_command_iterations = options.games * gpu_stats.commands_per_game * options.movement_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_movement_collision_benchmark_v1"},
        {"games", options.games},
        {"commands_per_game", gpu_stats.commands_per_game},
        {"iterations", options.movement_iterations},
        {"total_command_iterations", total_command_iterations},
        {"mismatches", mismatches},
        {"frame_mismatches", frame_mismatches},
        {"first_mismatch", first_mismatch},
        {"cpu_moved", cpu_stats.moved_commands},
        {"gpu_moved", gpu_stats.moved_commands},
        {"cpu_collision_cells", cpu_stats.collision_cells},
        {"gpu_collision_cells", gpu_stats.collision_cells},
        {"cpu_damaged", cpu_stats.damaged_entities},
        {"gpu_damaged", gpu_stats.damaged_entities},
        {"cpu_deaths", cpu_stats.deaths},
        {"gpu_deaths", gpu_stats.deaths},
        {"cpu_dropped_energy", cpu_stats.dropped_energy},
        {"gpu_dropped_energy", gpu_stats.dropped_energy},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_collision games=" << options.games
              << " commands=" << gpu_stats.commands_per_game
              << " iterations=" << options.movement_iterations
              << " mismatches=" << mismatches
              << " gpu_cmd/s=" << report["gpu_commands_per_second"].get<double>()
              << " cpu_cmd/s=" << report["cpu_commands_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_validation_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    std::vector<CommandFrame> command_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    command_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_movement_frame(options.seed_base + static_cast<unsigned int>(index), config, options.validation_commands);
        command_frames.push_back(synthetic_validation_commands(frame, options.validation_commands));
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    std::vector<gpu::ValidationFrameDecision> cpu_decisions;
    cpu_decisions.reserve(options.games * options.validation_commands);
    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.validation_iterations; ++iteration) {
        cpu_decisions.clear();
        for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
            auto decisions = cpu_validation_decisions(cpu_frames[game], config, command_frames[game]);
            cpu_decisions.insert(cpu_decisions.end(), decisions.begin(), decisions.end());
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    std::vector<gpu::ValidationFrameDecision> gpu_decisions;
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_validation_decisions(gpu_ptrs,
                                                            config,
                                                            command_frames,
                                                            gpu_decisions,
                                                            options.validation_iterations);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    const auto count = std::min(cpu_decisions.size(), gpu_decisions.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (!validation_decision_equal(cpu_decisions[index], gpu_decisions[index])) {
            if (first_mismatch.empty()) {
                const auto &cpu = cpu_decisions[index];
                const auto &gpu = gpu_decisions[index];
                first_mismatch = {
                    {"index", index},
                    {"cpu", {{"player", cpu.player.value}, {"entity", cpu.entity.value}, {"type", static_cast<int>(cpu.type)}, {"expense", cpu.expense}, {"exists", cpu.player_exists}, {"own", cpu.ownership_ok}, {"occ", cpu.occurrence_command}, {"exp", cpu.expense_command}, {"include", cpu.include_in_batch}}},
                    {"gpu", {{"player", gpu.player.value}, {"entity", gpu.entity.value}, {"type", static_cast<int>(gpu.type)}, {"expense", gpu.expense}, {"exists", gpu.player_exists}, {"own", gpu.ownership_ok}, {"occ", gpu.occurrence_command}, {"exp", gpu.expense_command}, {"include", gpu.include_in_batch}}}
                };
            }
            ++mismatches;
        }
    }
    mismatches += cpu_decisions.size() > gpu_decisions.size()
        ? cpu_decisions.size() - gpu_decisions.size()
        : gpu_decisions.size() - cpu_decisions.size();

    const auto total_commands = options.games * stats.commands_per_game;
    const auto total_command_iterations = total_commands * options.validation_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_validation_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.validation_iterations},
        {"commands_per_game", stats.commands_per_game},
        {"total_commands", total_commands},
        {"total_command_iterations", total_command_iterations},
        {"mismatches", mismatches},
        {"first_mismatch", first_mismatch},
        {"included_commands", stats.included_commands},
        {"expense_commands", stats.expense_commands},
        {"occurrence_commands", stats.occurrence_commands},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_commands_per_second", static_cast<double>(total_command_iterations) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_validation games=" << options.games
              << " commands=" << stats.commands_per_game
              << " iterations=" << options.validation_iterations
              << " mismatches=" << mismatches
              << " gpu_cmd/s=" << report["gpu_commands_per_second"].get<double>()
              << " cpu_cmd/s=" << report["cpu_commands_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_pipeline_benchmark(const Options &options) {
    auto config = load_config(options);
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 0.90;
    config.apply_to_global_constants();

    constexpr std::size_t ships_per_game = 256;
    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_pipeline_frame(options.seed_base + static_cast<unsigned int>(index), config, ships_per_game);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.dump_iterations; ++iteration) {
        if (iteration > 0) {
            for (auto &frame : cpu_frames) {
                refill_dump_cargo(frame, options.dump_refill_cargo);
            }
        }
        for (auto &frame : cpu_frames) {
            cpu_dump_frame(frame, config);
            cpu_regen_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_dump_regen_pipeline(gpu_ptrs,
                                                           config,
                                                           options.dump_iterations,
                                                           options.dump_refill_cargo);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (cpu_frames[game].map_total_energy != gpu_frames[game].map_total_energy ||
            cpu_frames[game].cell_energy != gpu_frames[game].cell_energy ||
            !dump_frame_equal(cpu_frames[game], gpu_frames[game])) {
            ++mismatches;
        }
    }

    const auto total_cells = options.games * stats.cells_per_game;
    const auto total_entities = options.games * stats.entities_per_game;
    const auto total_work_items = (total_cells + total_entities) * options.dump_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_dump_regen_pipeline_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.dump_iterations},
        {"refill_cargo", options.dump_refill_cargo},
        {"cells_per_game", stats.cells_per_game},
        {"entities_per_game", stats.entities_per_game},
        {"total_cells", total_cells},
        {"total_entities", total_entities},
        {"total_work_items", total_work_items},
        {"mismatches", mismatches},
        {"total_regen_delta", stats.total_regen_delta},
        {"total_deposited", stats.total_deposited},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_pipeline games=" << options.games
              << " iterations=" << options.dump_iterations
              << " mismatches=" << mismatches
              << " gpu_work/s=" << report["gpu_work_items_per_second"].get<double>()
              << " cpu_work/s=" << report["cpu_work_items_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_economy_pipeline_benchmark(const Options &options) {
    auto config = load_config(options);
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 0.90;
    config.ruleset.economy.mining_interference_range = 0;
    config.apply_to_global_constants();

    constexpr std::size_t ships_per_game = 256;
    constexpr energy_type mining_refill_cell_energy = 800;
    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        auto frame = synthetic_economy_frame(options.seed_base + static_cast<unsigned int>(index), config, ships_per_game);
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.dump_iterations; ++iteration) {
        if (iteration > 0) {
            for (auto &frame : cpu_frames) {
                refill_dump_cargo(frame, options.dump_refill_cargo);
                refill_mining_inputs(frame, mining_refill_cell_energy);
            }
        }
        for (auto &frame : cpu_frames) {
            cpu_dump_frame(frame, config);
            cpu_inspiration_frame(frame, config);
            cpu_mining_frame(frame, config);
            cpu_regen_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_economy_pipeline(gpu_ptrs,
                                                        config,
                                                        options.dump_iterations,
                                                        options.dump_refill_cargo,
                                                        mining_refill_cell_energy);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (cpu_frames[game].map_total_energy != gpu_frames[game].map_total_energy ||
            cpu_frames[game].cell_energy != gpu_frames[game].cell_energy ||
            !dump_frame_equal(cpu_frames[game], gpu_frames[game])) {
            ++mismatches;
        }
    }

    const auto total_cells = options.games * stats.cells_per_game;
    const auto total_entities = options.games * stats.entities_per_game;
    const auto total_work_items = (total_cells + total_entities * 2) * options.dump_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_economy_pipeline_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.dump_iterations},
        {"refill_cargo", options.dump_refill_cargo},
        {"mining_refill_cell_energy", mining_refill_cell_energy},
        {"cells_per_game", stats.cells_per_game},
        {"entities_per_game", stats.entities_per_game},
        {"total_work_items", total_work_items},
        {"mismatches", mismatches},
        {"total_regen_delta", stats.total_regen_delta},
        {"total_deposited", stats.total_deposited},
        {"total_mining_extracted", stats.total_mining_extracted},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_economy_pipeline games=" << options.games
              << " iterations=" << options.dump_iterations
              << " mismatches=" << mismatches
              << " gpu_work/s=" << report["gpu_work_items_per_second"].get<double>()
              << " cpu_work/s=" << report["cpu_work_items_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

int run_batched_real_economy_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    std::size_t entity_capacity = 0;
    std::size_t dropoff_capacity = 0;
    for (unsigned long index = 0; index < options.games; ++index) {
        auto game = make_initialized_game(options, config, index);
        auto frame = game->capture_frame();
        entity_capacity = std::max(entity_capacity, frame.entities.size());
        dropoff_capacity = std::max(dropoff_capacity, frame.dropoffs.size());
        cpu_frames.push_back(frame);
    }
    for (auto &frame : cpu_frames) {
        pad_frame_layout(frame, entity_capacity, dropoff_capacity);
        gpu_frames.push_back(frame);
    }

    const auto cpu_start = Clock::now();
    for (unsigned long iteration = 0; iteration < options.dump_iterations; ++iteration) {
        for (auto &frame : cpu_frames) {
            cpu_dump_frame(frame, config);
            cpu_inspiration_frame(frame, config);
            cpu_mining_frame(frame, config);
            cpu_regen_frame(frame, config);
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    std::vector<StateFrame *> gpu_ptrs;
    gpu_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        gpu_ptrs.push_back(&frame);
    }
    const auto gpu_start = Clock::now();
    auto stats = gpu::run_cuda_batched_economy_pipeline(gpu_ptrs,
                                                        config,
                                                        options.dump_iterations,
                                                        0,
                                                        0);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    std::size_t mismatches = 0;
    nlohmann::json first_mismatch = nlohmann::json::object();
    for (std::size_t game = 0; game < cpu_frames.size(); ++game) {
        if (!economy_frame_equal(cpu_frames[game], gpu_frames[game])) {
            if (first_mismatch.empty()) {
                first_mismatch["game"] = game;
                for (std::size_t entity = 0; entity < cpu_frames[game].entities.size(); ++entity) {
                    if (!dump_entity_equal(cpu_frames[game].entities[entity], gpu_frames[game].entities[entity])) {
                        const auto &cpu = cpu_frames[game].entities[entity];
                        const auto &gpu = gpu_frames[game].entities[entity];
                        first_mismatch["entity_index"] = entity;
                        first_mismatch["cpu"] = {{"id", cpu.id.value}, {"energy", cpu.energy}, {"inspired", cpu.is_inspired}, {"alive", cpu.alive}};
                        first_mismatch["gpu"] = {{"id", gpu.id.value}, {"energy", gpu.energy}, {"inspired", gpu.is_inspired}, {"alive", gpu.alive}};
                        break;
                    }
                }
                for (std::size_t cell = 0; cell < cpu_frames[game].cell_energy.size(); ++cell) {
                    if (cpu_frames[game].cell_energy[cell] != gpu_frames[game].cell_energy[cell]) {
                        first_mismatch["cell_index"] = cell;
                        first_mismatch["cpu_cell_energy"] = cpu_frames[game].cell_energy[cell];
                        first_mismatch["gpu_cell_energy"] = gpu_frames[game].cell_energy[cell];
                        break;
                    }
                }
            }
            ++mismatches;
        }
    }

    const auto total_cells = options.games * stats.cells_per_game;
    const auto total_entities = options.games * stats.entities_per_game;
    const auto total_work_items = (total_cells + total_entities) * options.dump_iterations;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_real_economy_benchmark_v1"},
        {"games", options.games},
        {"iterations", options.dump_iterations},
        {"cells_per_game", stats.cells_per_game},
        {"entities_per_game", stats.entities_per_game},
        {"total_cells", total_cells},
        {"total_entities", total_entities},
        {"total_work_items", total_work_items},
        {"mismatches", mismatches},
        {"first_mismatch", first_mismatch},
        {"total_regen_delta", stats.total_regen_delta},
        {"total_deposited", stats.total_deposited},
        {"total_mining_extracted", stats.total_mining_extracted},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_real_economy games=" << options.games
              << " iterations=" << options.dump_iterations
              << " mismatches=" << mismatches
              << " gpu_work/s=" << report["gpu_work_items_per_second"].get<double>()
              << " cpu_work/s=" << report["cpu_work_items_per_second"].get<double>() << "\n";
    return mismatches == 0 ? 0 : 1;
}

nlohmann::json profile_summary(const std::vector<std::unique_ptr<GameInstance>> &games) {
    std::unordered_map<std::string, std::pair<long long, long long>> phase_ns_and_count;
    for (const auto &game : games) {
        for (const auto &profile : game->turn_profiles()) {
            for (const auto &entry : profile.phase_timings) {
                auto &item = phase_ns_and_count[entry.phase_name];
                item.first += entry.duration_ns;
                item.second += 1;
            }
        }
    }
    nlohmann::json phases = nlohmann::json::object();
    for (const auto &[name, totals] : phase_ns_and_count) {
        phases[name] = {
            {"total_ms", totals.first / 1000000.0},
            {"mean_us", totals.second > 0 ? (totals.first / 1000.0) / static_cast<double>(totals.second) : 0.0},
            {"samples", totals.second}
        };
    }
    return phases;
}

void run_games_lockstep(const Options &options,
                        const GameConfig &config,
                        std::vector<std::unique_ptr<GameInstance>> &games) {
    games.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        games.push_back(make_initialized_game(options, config, index));
    }

    InlineExecutor inline_executor;
    const bool collect_profile = !options.profile_json.empty();
    const auto start = Clock::now();
    for (unsigned long turn = 1; turn <= options.turn_limit; ++turn) {
        std::vector<GameInstance *> active_games;
        active_games.reserve(games.size());
        for (auto &game : games) {
            if (!game->is_finished()) {
                active_games.push_back(game.get());
            }
        }
        if (active_games.empty()) {
            break;
        }

        std::vector<std::unique_ptr<ActionBatch>> prepared_actions;
        if (options.lockstep_batched_validation || options.lockstep_batched_dump) {
            prepared_actions.reserve(active_games.size());
            for (std::size_t index = 0; index < active_games.size(); ++index) {
                prepared_actions.push_back(std::make_unique<ActionBatch>());
                active_games[index]->prepare_actions_for_turn(turn, *prepared_actions[index]);
            }
        }

        auto run_batched_inspiration = [&] {
            std::vector<StateFrame> frames;
            frames.reserve(active_games.size());
            std::size_t max_entities = 0;
            for (auto *game : active_games) {
                game->set_turn_number(turn);
                auto frame = game->capture_frame();
                max_entities = std::max(max_entities, frame.entities.size());
                frames.push_back(std::move(frame));
            }
            if (!frames.empty()) {
                for (auto &frame : frames) {
                    pad_frame_layout(frame, max_entities, 0);
                }
                std::vector<StateFrame *> frame_ptrs;
                frame_ptrs.reserve(frames.size());
                for (auto &frame : frames) {
                    frame_ptrs.push_back(&frame);
                }
                gpu::run_cuda_batched_inspiration_resident(frame_ptrs, config, 1);
                std::size_t frame_index = 0;
                for (auto *game : active_games) {
                    game->apply_frame(frames[frame_index++]);
                }
            }
        };

        if (options.lockstep_batched_inspiration) {
            run_batched_inspiration();
        }

        if (options.lockstep_batched_validation) {
            std::vector<StateFrame> frames;
            frames.reserve(active_games.size());
            std::vector<std::vector<gpu::ValidationCommandRef>> refs_by_game;
            refs_by_game.reserve(active_games.size());
            std::vector<CommandFrame> command_frames;
            command_frames.reserve(active_games.size());
            std::size_t max_entities = 0;
            std::size_t max_commands = 0;
            for (std::size_t index = 0; index < active_games.size(); ++index) {
                active_games[index]->set_turn_number(turn);
                auto frame = active_games[index]->capture_frame();
                max_entities = std::max(max_entities, frame.entities.size());
                auto refs = gpu::flatten_validation_actions(*prepared_actions[index], active_games[index]->store_ref());
                auto command_frame = gpu::command_frame_from_validation_refs(refs);
                max_commands = std::max(max_commands, command_frame.commands.size());
                frames.push_back(std::move(frame));
                refs_by_game.push_back(std::move(refs));
                command_frames.push_back(std::move(command_frame));
            }
            if (max_commands == 0) {
                for (std::size_t index = 0; index < active_games.size(); ++index) {
                    StepResult result;
                    result.validated_commands = CommandBatch{};
                    active_games[index]->run_prevalidated_turn(turn,
                                                               *prepared_actions[index],
                                                               std::move(result),
                                                               collect_profile,
                                                               inline_executor);
                }
                continue;
            }
            for (std::size_t index = 0; index < active_games.size(); ++index) {
                pad_frame_layout(frames[index], max_entities, 0);
                while (command_frames[index].commands.size() < max_commands) {
                    command_frames[index].commands.push_back(FlatCommand{Player::None,
                                                                         Entity::None,
                                                                         Entity::None,
                                                                         Player::None,
                                                                         Location{0, 0},
                                                                         Direction::Still,
                                                                         FlatCommandType::Move});
                }
            }
            std::vector<StateFrame *> frame_ptrs;
            frame_ptrs.reserve(frames.size());
            for (auto &frame : frames) {
                frame_ptrs.push_back(&frame);
            }
            std::vector<gpu::ValidationFrameDecision> decisions;
            gpu::run_cuda_batched_validation_decisions(frame_ptrs, config, command_frames, decisions, 1);
            if (options.lockstep_batched_dump) {
                std::vector<StepResult> partial_results(active_games.size());
                std::vector<TurnExecutionProfile> profiles(active_games.size());
                for (std::size_t index = 0; index < active_games.size(); ++index) {
                    const auto begin = decisions.begin() + static_cast<std::ptrdiff_t>(index * max_commands);
                    std::vector<gpu::ValidationFrameDecision> game_decisions(begin,
                                                                             begin + static_cast<std::ptrdiff_t>(refs_by_game[index].size()));
                    GameState state = active_games[index]->state_view();
                    gpu::apply_validation_decisions(state,
                                                    *prepared_actions[index],
                                                    partial_results[index],
                                                    refs_by_game[index],
                                                    game_decisions);
                    partial_results[index] = active_games[index]->run_until_dump(turn,
                                                                                 *prepared_actions[index],
                                                                                 std::move(partial_results[index]),
                                                                                 collect_profile,
                                                                                 inline_executor,
                                                                                 profiles[index]);
                }

                std::vector<StateFrame> dump_before_frames;
                dump_before_frames.reserve(active_games.size());
                std::size_t max_entities = 0;
                std::size_t max_dropoffs = 0;
                for (auto *game : active_games) {
                    auto frame = game->capture_frame();
                    max_entities = std::max(max_entities, frame.entities.size());
                    max_dropoffs = std::max(max_dropoffs, frame.dropoffs.size());
                    dump_before_frames.push_back(std::move(frame));
                }
                std::vector<StateFrame> dump_after_frames = dump_before_frames;
                for (auto &frame : dump_after_frames) {
                    pad_frame_layout(frame, max_entities, max_dropoffs);
                }
                std::vector<StateFrame *> dump_ptrs;
                dump_ptrs.reserve(dump_after_frames.size());
                for (auto &frame : dump_after_frames) {
                    dump_ptrs.push_back(&frame);
                }
                gpu::run_cuda_batched_dump(dump_ptrs, config, 1, 0);
                for (std::size_t index = 0; index < active_games.size(); ++index) {
                    const auto &before = dump_before_frames[index];
                    auto &after = dump_after_frames[index];
                    for (const auto &before_entity : before.entities) {
                        if (!before_entity.alive) {
                            continue;
                        }
                        const auto cell_index_value = cell_index(before, before_entity.location);
                        if (before.cell_owner[cell_index_value] != before_entity.owner) {
                            continue;
                        }
                        partial_results[index].changed_cells.emplace(before_entity.location);
                        partial_results[index].changed_entities.emplace(before_entity.id);
                    }
                    active_games[index]->apply_frame(after);
                    active_games[index]->run_after_dump(turn,
                                                        *prepared_actions[index],
                                                        std::move(partial_results[index]),
                                                        collect_profile,
                                                        inline_executor,
                                                        profiles[index]);
                }
                continue;
            }
            for (std::size_t index = 0; index < active_games.size(); ++index) {
                StepResult result;
                const auto begin = decisions.begin() + static_cast<std::ptrdiff_t>(index * max_commands);
                std::vector<gpu::ValidationFrameDecision> game_decisions(begin,
                                                                         begin + static_cast<std::ptrdiff_t>(refs_by_game[index].size()));
                GameState state = active_games[index]->state_view();
                gpu::apply_validation_decisions(state,
                                                *prepared_actions[index],
                                                result,
                                                refs_by_game[index],
                                                game_decisions);
                active_games[index]->run_prevalidated_turn(turn,
                                                           *prepared_actions[index],
                                                           std::move(result),
                                                           collect_profile,
                                                           inline_executor);
            }
        } else {
            for (auto *game : active_games) {
                game->run_one_turn(turn, collect_profile, inline_executor);
            }
        }

        if (options.lockstep_batched_regen) {
            std::vector<StateFrame> frames;
            frames.reserve(active_games.size());
            for (auto *game : active_games) {
                if (game->is_finished()) {
                    continue;
                }
                game->set_turn_number(turn);
                frames.push_back(game->capture_frame());
            }
            if (!frames.empty()) {
                std::vector<StateFrame *> frame_ptrs;
                frame_ptrs.reserve(frames.size());
                for (auto &frame : frames) {
                    frame_ptrs.push_back(&frame);
                }
                gpu::run_cuda_batched_regen(frame_ptrs, config);
                std::size_t frame_index = 0;
                for (auto *game : active_games) {
                    if (game->is_finished()) {
                        continue;
                    }
                    game->apply_frame(frames[frame_index++]);
                }
            }
        }

        if (options.lockstep_batched_end_economy) {
            std::vector<StateFrame> frames;
            frames.reserve(active_games.size());
            std::size_t max_dropoffs = 0;
            for (auto *game : active_games) {
                if (game->is_finished()) {
                    continue;
                }
                game->set_turn_number(turn);
                auto frame = game->capture_frame();
                max_dropoffs = std::max(max_dropoffs, frame.dropoffs.size());
                frames.push_back(std::move(frame));
            }
            if (!frames.empty()) {
                for (auto &frame : frames) {
                    pad_frame_layout(frame, frame.entities.size(), max_dropoffs);
                }
                std::vector<StateFrame *> frame_ptrs;
                frame_ptrs.reserve(frames.size());
                for (auto &frame : frames) {
                    frame_ptrs.push_back(&frame);
                }
                gpu::run_cuda_batched_end_economy(frame_ptrs, config);
                std::size_t frame_index = 0;
                for (auto *game : active_games) {
                    if (game->is_finished()) {
                        continue;
                    }
                    game->apply_frame(frames[frame_index++]);
                }
            }
        }
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    for (auto &game : games) {
        game->finish_run(elapsed_ms);
    }
}

void run_games_parallel_chunks(const Options &options,
                               const GameConfig &config,
                               std::vector<std::unique_ptr<GameInstance>> &games) {
    games.reserve(options.games);
    unsigned long next_index = 0;
    while (next_index < options.games) {
        const auto batch_count = std::min(options.workers, options.games - next_index);
        std::vector<std::future<std::unique_ptr<GameInstance>>> futures;
        futures.reserve(batch_count);
        for (unsigned long offset = 0; offset < batch_count; ++offset) {
            const auto index = next_index + offset;
            futures.push_back(std::async(std::launch::async, [&, index] {
                auto game = make_initialized_game(options, config, index);
                game->run(!options.profile_json.empty());
                return game;
            }));
        }
        for (auto &future : futures) {
            games.push_back(future.get());
        }
        next_index += batch_count;
    }
}

int run_lockstep_economy_window_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<std::unique_ptr<GameInstance>> games;
    games.reserve(options.games);
    for (unsigned long index = 0; index < options.games; ++index) {
        games.push_back(make_initialized_game(options, config, index));
    }

    std::vector<StateFrame> persistent_frames;
    std::vector<StateFrame *> persistent_ptrs;
    std::size_t persistent_entities = 0;
    std::size_t persistent_dropoffs = 0;
    if (options.persistent_frame_window) {
        persistent_frames.reserve(games.size());
        for (auto &game : games) {
            auto frame = game->capture_frame();
            persistent_entities = std::max(persistent_entities, frame.entities.size());
            persistent_dropoffs = std::max(persistent_dropoffs, frame.dropoffs.size());
            persistent_frames.push_back(std::move(frame));
        }
        for (auto &frame : persistent_frames) {
            pad_frame_layout(frame, persistent_entities, persistent_dropoffs);
            persistent_ptrs.push_back(&frame);
        }
    }

    long long capture_ns = 0;
    long long gpu_ns = 0;
    long long apply_ns = 0;
    unsigned long long total_regen_delta = 0;
    unsigned long long total_deposited = 0;
    unsigned long long total_mining_extracted = 0;
    std::size_t max_entities = 0;
    std::size_t max_dropoffs = 0;
    const auto start = Clock::now();
    if (options.persistent_frame_window) {
        auto capture_start = Clock::now();
        for (auto &frame : persistent_frames) {
            frame.turn_number = 1;
        }
        max_entities = persistent_entities;
        max_dropoffs = persistent_dropoffs;
        capture_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - capture_start).count();

        auto gpu_start = Clock::now();
        const auto stats = gpu::run_cuda_batched_economy_pipeline_device_resident(persistent_ptrs,
                                                                                  config,
                                                                                  options.turn_limit,
                                                                                  options.dump_iterations,
                                                                                  0,
                                                                                  0);
        gpu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - gpu_start).count();
        total_regen_delta += stats.total_regen_delta;
        total_deposited += stats.total_deposited;
        total_mining_extracted += stats.total_mining_extracted;
    } else {
        for (unsigned long turn = 1; turn <= options.turn_limit; ++turn) {
            std::vector<StateFrame> frames;
            std::vector<StateFrame *> frame_ptrs;
            frames.reserve(games.size());
            auto capture_start = Clock::now();
            max_entities = 0;
            max_dropoffs = 0;
            for (auto &game : games) {
                game->set_turn_number(turn);
                auto frame = game->capture_frame();
                frame.turn_number = turn;
                max_entities = std::max(max_entities, frame.entities.size());
                max_dropoffs = std::max(max_dropoffs, frame.dropoffs.size());
                frames.push_back(std::move(frame));
            }
            for (auto &frame : frames) {
                pad_frame_layout(frame, max_entities, max_dropoffs);
            }
            frame_ptrs.reserve(frames.size());
            for (auto &frame : frames) {
                frame_ptrs.push_back(&frame);
            }
            capture_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - capture_start).count();

            auto gpu_start = Clock::now();
            const auto stats = gpu::run_cuda_batched_economy_pipeline(frame_ptrs,
                                                                      config,
                                                                      options.dump_iterations,
                                                                      0,
                                                                      0);
            gpu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - gpu_start).count();
            total_regen_delta += stats.total_regen_delta;
            total_deposited += stats.total_deposited;
            total_mining_extracted += stats.total_mining_extracted;

            auto apply_start = Clock::now();
            for (std::size_t index = 0; index < games.size(); ++index) {
                games[index]->apply_frame(frames[index]);
            }
            apply_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - apply_start).count();
        }
    }
    const auto elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
    const auto total_cells = options.games * 1024ULL * options.turn_limit * options.dump_iterations;
    const auto total_entity_slots = static_cast<unsigned long long>(options.games) * max_entities * options.turn_limit * options.dump_iterations;
    const auto total_work_items = total_cells + total_entity_slots;
    nlohmann::json report = {
        {"schema", "halite3_v3_lockstep_economy_window_benchmark_v1"},
        {"games", options.games},
        {"turn_limit", options.turn_limit},
        {"iterations_per_turn", options.dump_iterations},
        {"persistent_frame_window", options.persistent_frame_window},
        {"max_entities_per_game", max_entities},
        {"max_dropoffs_per_game", max_dropoffs},
        {"total_work_items", total_work_items},
        {"elapsed_s", elapsed_s},
        {"capture_s", capture_ns / 1.0e9},
        {"gpu_s", gpu_ns / 1.0e9},
        {"apply_s", apply_ns / 1.0e9},
        {"work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, elapsed_s)},
        {"gpu_work_items_per_second", static_cast<double>(total_work_items) / std::max(1.0e-9, gpu_ns / 1.0e9)},
        {"total_regen_delta", total_regen_delta},
        {"total_deposited", total_deposited},
        {"total_mining_extracted", total_mining_extracted}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "lockstep_economy_window games=" << options.games
              << " turns=" << options.turn_limit
              << " work/s=" << report["work_items_per_second"].get<double>()
              << " gpu_work/s=" << report["gpu_work_items_per_second"].get<double>() << "\n";
    return 0;
}

int run_batched_inspiration_resident_benchmark(const Options &options) {
    auto config = load_config(options);
    config.apply_to_global_constants();

    std::vector<std::unique_ptr<GameInstance>> games;
    games.reserve(options.games);
    std::vector<StateFrame> cpu_frames;
    std::vector<StateFrame> gpu_frames;
    cpu_frames.reserve(options.games);
    gpu_frames.reserve(options.games);
    std::size_t max_entities = 0;
    for (unsigned long index = 0; index < options.games; ++index) {
        auto game = make_initialized_game(options, config, index);
        auto frame = game->capture_frame();
        max_entities = std::max(max_entities, frame.entities.size());
        cpu_frames.push_back(frame);
        gpu_frames.push_back(std::move(frame));
        games.push_back(std::move(game));
    }
    for (auto &frame : cpu_frames) {
        pad_frame_layout(frame, max_entities, 0);
    }
    for (auto &frame : gpu_frames) {
        pad_frame_layout(frame, max_entities, 0);
    }
    std::vector<StateFrame *> frame_ptrs;
    frame_ptrs.reserve(gpu_frames.size());
    for (auto &frame : gpu_frames) {
        frame_ptrs.push_back(&frame);
    }

    unsigned long long cpu_inspired = 0;
    const auto cpu_start = Clock::now();
    for (unsigned long turn = 0; turn < options.turn_limit; ++turn) {
        (void)turn;
        for (auto &frame : cpu_frames) {
            cpu_inspiration_frame(frame, config);
        }
    }
    for (const auto &frame : cpu_frames) {
        for (const auto &entity : frame.entities) {
            if (entity.alive && entity.is_inspired) {
                ++cpu_inspired;
            }
        }
    }
    const auto cpu_elapsed_s = std::chrono::duration<double>(Clock::now() - cpu_start).count();

    const auto gpu_start = Clock::now();
    const auto stats = gpu::run_cuda_batched_inspiration_resident(frame_ptrs, config, options.turn_limit);
    const auto gpu_elapsed_s = std::chrono::duration<double>(Clock::now() - gpu_start).count();

    const auto total_entity_turns = static_cast<unsigned long long>(options.games) * max_entities * options.turn_limit;
    const bool mismatch = cpu_inspired != stats.inspired_entities;
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_inspiration_resident_benchmark_v1"},
        {"games", options.games},
        {"turn_limit", options.turn_limit},
        {"entities_per_game", max_entities},
        {"total_entity_turns", total_entity_turns},
        {"mismatches", mismatch ? 1 : 0},
        {"cpu_inspired", cpu_inspired},
        {"gpu_inspired", stats.inspired_entities},
        {"cpu_elapsed_s", cpu_elapsed_s},
        {"gpu_elapsed_s", gpu_elapsed_s},
        {"cpu_entity_turns_per_second", static_cast<double>(total_entity_turns) / std::max(1.0e-9, cpu_elapsed_s)},
        {"gpu_entity_turns_per_second", static_cast<double>(total_entity_turns) / std::max(1.0e-9, gpu_elapsed_s)}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    std::cerr << "batched_inspiration_resident games=" << options.games
              << " turns=" << options.turn_limit
              << " mismatches=" << (mismatch ? 1 : 0)
              << " gpu_entity_turns/s=" << report["gpu_entity_turns_per_second"].get<double>()
              << " cpu_entity_turns/s=" << report["cpu_entity_turns_per_second"].get<double>() << "\n";
    return mismatch ? 1 : 0;
}

int run(const Options &options) {
    if (options.lockstep_economy_window_benchmark) {
        return run_lockstep_economy_window_benchmark(options);
    }
    if (options.batched_inspiration_resident_benchmark) {
        return run_batched_inspiration_resident_benchmark(options);
    }
    if (options.batched_regen_benchmark) {
        return run_batched_regen_benchmark(options);
    }
    if (options.batched_dump_benchmark) {
        return run_batched_dump_benchmark(options);
    }
    if (options.batched_mining_benchmark) {
        return run_batched_mining_benchmark(options);
    }
    if (options.batched_spawn_benchmark) {
        return run_batched_spawn_benchmark(options);
    }
    if (options.batched_movement_benchmark) {
        return run_batched_movement_benchmark(options);
    }
    if (options.batched_movement_apply_benchmark) {
        return run_batched_movement_apply_benchmark(options);
    }
    if (options.batched_destination_benchmark) {
        return run_batched_destination_benchmark(options);
    }
    if (options.batched_collision_benchmark) {
        return run_batched_collision_benchmark(options);
    }
    if (options.batched_validation_benchmark) {
        return run_batched_validation_benchmark(options);
    }
    if (options.batched_pipeline_benchmark) {
        return run_batched_pipeline_benchmark(options);
    }
    if (options.batched_economy_pipeline_benchmark) {
        return run_batched_economy_pipeline_benchmark(options);
    }
    if (options.batched_real_economy_benchmark) {
        return run_batched_real_economy_benchmark(options);
    }
    auto config = load_config(options);
    if (options.lockstep_batched_inspiration) {
        config.runtime.skip_inspiration_phase = true;
    }
    if (options.lockstep_batched_regen) {
        config.runtime.skip_regen_phase = true;
    }
    if (options.lockstep_batched_end_economy) {
        config.runtime.skip_end_economy_phases = true;
    }
    std::vector<std::unique_ptr<GameInstance>> games;
    const auto batch_start = Clock::now();
    if (options.lockstep_runner) {
        run_games_lockstep(options, config, games);
    } else {
        run_games_parallel_chunks(options, config, games);
    }
    const auto elapsed_s = std::chrono::duration<double>(Clock::now() - batch_start).count();
    unsigned long total_turns = 0;
    std::size_t errors = 0;
    nlohmann::json game_rows = nlohmann::json::array();
    for (const auto &game : games) {
        auto row = game->summary_json();
        total_turns += row["turns"].get<unsigned long>();
        errors += row["errors"].get<std::size_t>();
        game_rows.push_back(std::move(row));
    }
    nlohmann::json report = {
        {"schema", "halite3_v3_batched_runner_v1"},
        {"games", options.games},
        {"backend", options.backend},
        {"config", options.config_path},
        {"turn_limit", options.turn_limit},
        {"seed_base", options.seed_base},
        {"provider_p0", options.provider_p0},
        {"provider_p1", options.provider_p1},
        {"workers", options.workers},
        {"elapsed_s", elapsed_s},
        {"games_per_second", static_cast<double>(options.games) / std::max(1.0e-9, elapsed_s)},
        {"turns_per_second", static_cast<double>(total_turns) / std::max(1.0e-9, elapsed_s)},
        {"total_turns", total_turns},
        {"errors", errors},
        {"results", game_rows}
    };
    if (!options.results_json.empty()) {
        std::ofstream out(options.results_json);
        out << report.dump(2) << "\n";
    } else {
        std::cout << report.dump(2) << std::endl;
    }
    if (!options.profile_json.empty()) {
        nlohmann::json profile = {
            {"schema", "halite3_v3_batched_runner_profile_v1"},
            {"phase_summary", profile_summary(games)}
        };
        std::ofstream out(options.profile_json);
        out << profile.dump(2) << "\n";
    }
    std::cerr << "batched_runner games=" << options.games
              << " turns/s=" << report["turns_per_second"].get<double>()
              << " games/s=" << report["games_per_second"].get<double>()
              << " errors=" << errors << "\n";
    return errors == 0 ? 0 : 1;
}

} // namespace
} // namespace hlt::batched

int main(int argc, char **argv) {
    try {
        return hlt::batched::run(hlt::batched::parse_args(argc, argv));
    } catch (const std::exception &err) {
        std::cerr << "halite_batched_runner: " << err.what() << std::endl;
        return 2;
    }
}
