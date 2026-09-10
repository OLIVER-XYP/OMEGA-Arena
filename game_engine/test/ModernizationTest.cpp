#include "catch.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <tuple>
#include <variant>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#define private public
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "Store.hpp"
#include "config/GameConfig.hpp"
#include "core/engine/CommandFrame.hpp"
#include "core/engine/CpuTurnBackend.hpp"
#include "core/engine/StateFrame.hpp"
#include "core/engine/TurnEngine.hpp"
#include "core/engine/LocalStepResult.hpp"
#include "core/engine/TaskExecutor.hpp"
#include "observers/TurnStatsCollector.hpp"
#include "protocol/TextProtocolCodec.hpp"
#include "rules/RuleContext.hpp"
#include "rules/phases/DefendPhase.hpp"
#include "rules/phases/DumpPhase.hpp"
#include "rules/phases/HaliteRebalancePhase.hpp"
#include "rules/phases/InspirationPhase.hpp"
#include "rules/phases/MiningPhase.hpp"
#include "rules/phases/OverShipTaxPhase.hpp"
#include "rules/phases/PlunderPhase.hpp"
#include "rules/phases/RegenPhase.hpp"
#include "rules/phases/ValidationPhase.hpp"
#include "events/EventBuffer.hpp"
#include "events/EventSink.hpp"
#include "gpu/CudaCapture.hpp"
#include "gpu/CudaCombat.hpp"
#include "gpu/CudaConstruction.hpp"
#include "gpu/CudaDefend.hpp"
#include "gpu/CudaDump.hpp"
#include "gpu/CudaHaliteRebalance.hpp"
#include "gpu/CudaInspiration.hpp"
#include "gpu/CudaMining.hpp"
#include "gpu/CudaMovement.hpp"
#include "gpu/CudaOverShipTax.hpp"
#include "gpu/CudaPhaseRunner.hpp"
#include "gpu/CudaPlunder.hpp"
#include "gpu/CudaRegen.hpp"
#include "gpu/CudaSpawn.hpp"
#include "gpu/CudaStateMirror.hpp"
#include "gpu/CudaTurnBackend.hpp"
#include "gpu/CudaValidation.hpp"
#include "rules/phases/CapturePhase.hpp"

namespace {

class CountingExecutor final : public hlt::TaskExecutor {
public:
    std::size_t calls = 0;
    std::size_t last_begin = 0;
    std::size_t last_end = 0;

    void parallel_for(std::size_t begin,
                      std::size_t end,
                      const std::function<void(std::size_t)> &fn) override {
        ++calls;
        last_begin = begin;
        last_end = end;
        for (std::size_t index = begin; index < end; ++index) {
            fn(index);
        }
    }

    std::string executor_name() const override { return "counting"; }

    std::size_t executor_thread_count() const override { return 1; }

    hlt::ExecutorRunStats run_stats() const override {
        return hlt::ExecutorRunStats{calls, last_end >= last_begin ? (last_end - last_begin) : 0};
    }

    void reset_run_stats() override {
        calls = 0;
        last_begin = 0;
        last_end = 0;
    }
};

} // namespace

TEST_CASE("GameConfig mirrors current constants", "[modernization][config]") {
    const auto config = hlt::GameConfig::from_constants();
    REQUIRE(config.match.max_players == hlt::Constants::get().MAX_PLAYERS);
    REQUIRE(config.ruleset.economy.max_energy == hlt::Constants::get().MAX_ENERGY);
    REQUIRE(config.ruleset.combat.enable_combat_commands == hlt::Constants::get().ENABLE_COMBAT_COMMANDS);
    REQUIRE(config.runtime.engine_backend == hlt::EngineBackendMode::Cpu);
    REQUIRE(config.runtime.gpu_batch_size == 1);
}

TEST_CASE("Engine backend mode parser accepts supported modes", "[modernization][config]") {
    REQUIRE(hlt::parse_engine_backend_mode("cpu") == hlt::EngineBackendMode::Cpu);
    REQUIRE(hlt::parse_engine_backend_mode("gpu") == hlt::EngineBackendMode::Gpu);
    REQUIRE(hlt::parse_engine_backend_mode("auto") == hlt::EngineBackendMode::Auto);
    REQUIRE(std::string(hlt::engine_backend_mode_name(hlt::EngineBackendMode::Gpu)) == "gpu");
    REQUIRE_THROWS_AS(hlt::parse_engine_backend_mode("bogus"), std::invalid_argument);
}

TEST_CASE("Cuda turn backend reports capability status", "[modernization][gpu]") {
#ifndef HALITE_ENABLE_CUDA
    REQUIRE_FALSE(hlt::gpu::cuda_turn_backend_available());
#else
    REQUIRE(hlt::gpu::cuda_turn_backend_available());
#endif
    REQUIRE(std::string(hlt::gpu::cuda_turn_backend_unavailable_reason()).empty() == false);
    REQUIRE(std::string(hlt::gpu::cuda_turn_backend_capability_summary()).empty() == false);
}

TEST_CASE("TextProtocolCodec preserves raw commands", "[modernization][protocol]") {
    hlt::protocol::TextProtocolCodec codec;
    auto raw = codec.decode_commands("m 1 n g");
    REQUIRE(raw.raw_text == "m 1 n g");
}

TEST_CASE("ThreadPoolExecutor runs all iterations", "[modernization][engine]") {
    hlt::ThreadPoolExecutor executor{2};
    std::atomic<std::size_t> counter{0};

    executor.parallel_for(0, 16, [&](std::size_t index) {
        counter.fetch_add(index + 1);
    });

    REQUIRE(counter.load() == 136);
}

TEST_CASE("TurnEngine can execute an empty in-memory step", "[modernization][engine]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();
    hlt::TurnEngine engine(config);

    auto result = engine.step(state, actions, sink);
    REQUIRE(result.events.empty());
    REQUIRE(result.non_fatal_errors.empty());
}

TEST_CASE("TurnEngine can execute using explicit executor", "[modernization][engine]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();
    hlt::TurnEngine engine(config);
    hlt::InlineExecutor executor;

    auto result = engine.step(state, actions, sink, executor);
    REQUIRE(result.events.empty());
    REQUIRE(result.non_fatal_errors.empty());
}

TEST_CASE("CpuTurnBackend delegates to TurnEngine", "[modernization][engine]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();
    hlt::TurnEngine engine(config);
    hlt::CpuTurnBackend backend(engine);
    hlt::InlineExecutor executor;

    auto result = backend.step(state, actions, sink, executor, nullptr);
    REQUIRE(backend.backend_name() == "cpu");
    REQUIRE(result.events.empty());
    REQUIRE(result.non_fatal_errors.empty());
}

TEST_CASE("CommandFrame flattens raw action batches", "[modernization][engine]") {
    hlt::ActionBatch actions{};
    const auto player = hlt::Player::id_type{0};
    const auto ship = hlt::Entity::id_type{7};
    const auto target = hlt::Entity::id_type{9};
    actions[player].push_back(std::make_unique<hlt::MoveCommand>(ship, hlt::Direction::North));
    actions[player].push_back(std::make_unique<hlt::AttackCommand>(ship, target));
    actions[player].push_back(std::make_unique<hlt::SpawnCommand>());

    const auto frame = hlt::flatten_action_batch(actions);

    REQUIRE(frame.commands.size() == 3);
    REQUIRE(frame.commands[0].type == hlt::FlatCommandType::Move);
    REQUIRE(frame.commands[0].entity == ship);
    REQUIRE(frame.commands[0].direction == hlt::Direction::North);
    REQUIRE(frame.commands[1].type == hlt::FlatCommandType::AttackShip);
    REQUIRE(frame.commands[1].target == target);
    REQUIRE(frame.commands[2].type == hlt::FlatCommandType::Spawn);
    REQUIRE(frame.commands[2].player == player);
}

TEST_CASE("StateFrame captures map players and entities as arrays", "[modernization][engine]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    state.turn.number = 3;

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    player.energy = 1234;
    player.factory_halite = 4567;
    auto &entity = store.new_entity(222, player.id);
    entity.hp = 77;
    entity.is_defending = true;
    player.add_entity(entity.id, hlt::Location{2, 1});
    map.at(hlt::Location{2, 1}).entity = entity.id;
    map.at(hlt::Location{2, 1}).energy = 99;
    map.at(hlt::Location{0, 0}).owner = player.id;

    const auto frame = hlt::capture_state_frame(state);
    const auto occupied_index = hlt::cell_index(frame, hlt::Location{2, 1});

    REQUIRE(frame.width == 4);
    REQUIRE(frame.height == 4);
    REQUIRE(frame.turn_number == 3);
    REQUIRE(frame.cell_energy[occupied_index] == 99);
    REQUIRE(frame.cell_entity[occupied_index] == entity.id);
    REQUIRE(frame.players.size() == 1);
    REQUIRE(frame.players[0].energy == 1234);
    REQUIRE(frame.players[0].ship_count == 1);
    REQUIRE(frame.entities.size() == 1);
    REQUIRE(frame.entities[0].location == hlt::Location{2, 1});
    REQUIRE(frame.entities[0].hp == 77);
    REQUIRE(frame.entities[0].is_defending);
    REQUIRE(frame.entity_slot_by_id.at(entity.id) == 0);
}

TEST_CASE("StateFrame applies core fields back to GameState", "[modernization][engine]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    player.energy = 100;
    auto &entity = store.new_entity(10, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;
    map.at(hlt::Location{1, 1}).energy = 50;

    auto frame = hlt::capture_state_frame(state);
    frame.players[0].energy = 777;
    frame.entities[0].energy = 333;
    frame.entities[0].is_inspired = true;
    frame.cell_energy[hlt::cell_index(frame, hlt::Location{1, 1})] = 12;

    hlt::apply_state_frame(state, frame);

    REQUIRE(player.energy == 777);
    REQUIRE(store.get_entity(entity.id).energy == 333);
    REQUIRE(store.get_entity(entity.id).is_inspired);
    REQUIRE(map.at(hlt::Location{1, 1}).energy == 12);
}

TEST_CASE("CudaStateMirror roundtrips state frames", "[modernization][gpu]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &entity = store.new_entity(321, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;
    map.at(hlt::Location{1, 1}).energy = 88;

    hlt::gpu::CudaStateMirror mirror;
    const auto frame = hlt::capture_state_frame(state);
    mirror.upload(frame);
    const auto roundtrip = mirror.download();

    (void)mirror.has_device_storage();
    REQUIRE(mirror.cell_count() == 16);
    REQUIRE(mirror.entity_count() == 1);
    REQUIRE(mirror.player_count() == 1);
    REQUIRE(roundtrip.cell_energy[hlt::cell_index(roundtrip, hlt::Location{1, 1})] == 88);
    REQUIRE(roundtrip.entities[0].energy == 321);
}

TEST_CASE("Cuda inspiration kernel matches CPU rule on simple frame", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);

    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    auto &center = store.new_entity(0, p0.id);
    p0.add_entity(center.id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = center.id;
    auto &enemy_a = store.new_entity(0, p1.id);
    p1.add_entity(enemy_a.id, hlt::Location{2, 3});
    map.at(hlt::Location{2, 3}).entity = enemy_a.id;
    auto &enemy_b = store.new_entity(0, p1.id);
    p1.add_entity(enemy_b.id, hlt::Location{3, 2});
    map.at(hlt::Location{3, 2}).entity = enemy_b.id;

    auto frame = hlt::capture_state_frame(state);
    auto config = hlt::GameConfig::from_constants().ruleset.inspiration;
    config.enabled = true;
    config.radius = 2;
    config.ship_count = 2;

    hlt::gpu::run_cuda_inspiration(frame, config);

    const auto center_slot = frame.entity_slot_by_id.at(center.id);
    REQUIRE(frame.entities[center_slot].is_inspired);
#endif
}

TEST_CASE("Cuda inspiration kernel matches InspirationPhase", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = true;
    config.ruleset.inspiration.radius = 2;
    config.ruleset.inspiration.ship_count = 2;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;
        auto &center = store.new_entity(0, p0.id);
        p0.add_entity(center.id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = center.id;
        auto &far = store.new_entity(0, p0.id);
        p0.add_entity(far.id, hlt::Location{0, 4});
        map.at(hlt::Location{0, 4}).entity = far.id;
        auto &enemy_a = store.new_entity(0, p1.id);
        p1.add_entity(enemy_a.id, hlt::Location{2, 3});
        map.at(hlt::Location{2, 3}).entity = enemy_a.id;
        auto &enemy_b = store.new_entity(0, p1.id);
        p1.add_entity(enemy_b.id, hlt::Location{3, 2});
        map.at(hlt::Location{3, 2}).entity = enemy_b.id;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics>{std::move(store), std::move(map), std::move(statistics)};
    };

    auto [cpu_store, cpu_map, cpu_stats] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    hlt::StepResult cpu_result{};
    hlt::events::RecordingEventSink sink{};
    hlt::InlineExecutor executor{};
    hlt::rules::RuleContext context(cpu_state, cpu_actions, cpu_result, config, sink, &executor);
    hlt::rules::phases::InspirationPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_inspiration(frame, config.ruleset.inspiration);

    for (const auto &entry : frame.entities) {
        REQUIRE(entry.is_inspired == cpu_store.get_entity(entry.id).is_inspired);
    }
#endif
}

TEST_CASE("Cuda inspiration phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = true;
    config.ruleset.inspiration.radius = 2;
    config.ruleset.inspiration.ship_count = 2;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    auto &center = store.new_entity(0, p0.id);
    p0.add_entity(center.id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = center.id;
    auto &enemy_a = store.new_entity(0, p1.id);
    p1.add_entity(enemy_a.id, hlt::Location{2, 3});
    map.at(hlt::Location{2, 3}).entity = enemy_a.id;
    auto &enemy_b = store.new_entity(0, p1.id);
    p1.add_entity(enemy_b.id, hlt::Location{3, 2});
    map.at(hlt::Location{3, 2}).entity = enemy_b.id;

    REQUIRE_FALSE(store.get_entity(center.id).is_inspired);
    hlt::gpu::run_cuda_inspiration_phase(state, config);
    REQUIRE(store.get_entity(center.id).is_inspired);
#endif
}

TEST_CASE("Cuda construction decisions match ConstructionPhase planning", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.dropoff_cost = 4000;
    config.ruleset.economy.dropoff_cost_growth = 0.25;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    p0.dropoffs.emplace_back(store.new_dropoff(hlt::Location{1, 0}));

    auto &valid_ship = store.new_entity(100, p0.id);
    const auto valid_ship_id = valid_ship.id;
    p0.add_entity(valid_ship_id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = valid_ship_id;

    auto &blocked_ship = store.new_entity(200, p1.id);
    const auto blocked_ship_id = blocked_ship.id;
    p1.add_entity(blocked_ship_id, hlt::Location{3, 3});
    map.at(hlt::Location{3, 3}).entity = blocked_ship_id;
    map.at(hlt::Location{3, 3}).owner = p0.id;

    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{p0.id,
                                                 valid_ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Construct});
    commands.commands.push_back(hlt::FlatCommand{p1.id,
                                                 blocked_ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Construct});

    const auto decisions = hlt::gpu::run_cuda_construction_decisions(hlt::capture_state_frame(state), commands, config);
    REQUIRE(decisions.size() == 2);
    REQUIRE(decisions[0].command);
    REQUIRE_FALSE(decisions[0].entity_missing);
    REQUIRE_FALSE(decisions[0].cell_owned);
    REQUIRE(decisions[0].location == hlt::Location{2, 2});
    REQUIRE(decisions[0].cost == 5000);
    REQUIRE(decisions[1].command);
    REQUIRE_FALSE(decisions[1].entity_missing);
    REQUIRE(decisions[1].cell_owned);
    REQUIRE(decisions[1].location == hlt::Location{3, 3});
    REQUIRE(decisions[1].cell_owner == p0.id);
#endif
}

TEST_CASE("Cuda mining kernel matches MiningPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.mining_interference_range = 0;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        hlt::GameState state(store, map, statistics);
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        auto &ship = store.new_entity(0, player.id);
        player.add_entity(ship.id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = ship.id;
        map.at(hlt::Location{2, 2}).energy = 100;
        store.map_total_energy = 100;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type>{std::move(store), std::move(map), std::move(statistics), ship.id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_ship] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::MiningPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_ship] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::TurnFrame turn_frame{};
    hlt::gpu::run_cuda_mining(frame, turn_frame, config);

    const auto gpu_slot = frame.entity_slot_by_id.at(gpu_ship);
    REQUIRE(frame.entities[gpu_slot].energy == cpu_store.get_entity(cpu_ship).energy);
    REQUIRE(frame.cell_energy[hlt::cell_index(frame, hlt::Location{2, 2})] == cpu_map.at(hlt::Location{2, 2}).energy);
    REQUIRE(frame.map_total_energy == cpu_store.map_total_energy);
#endif
}

TEST_CASE("Cuda mining applies same-cell effects deterministically", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.mining_interference_range = 0;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        statistics.player_statistics.emplace_back(hlt::Player::id_type{1}, 0);
        hlt::GameState state(store, map, statistics);
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);

        const hlt::Location mining_cell{2, 2};
        auto &ship0 = store.new_entity(0, p0_it->second.id);
        const auto ship0_id = ship0.id;
        p0_it->second.add_entity(ship0_id, mining_cell);
        auto &ship1 = store.new_entity(0, p1_it->second.id);
        const auto ship1_id = ship1.id;
        p1_it->second.add_entity(ship1_id, mining_cell);
        map.at(mining_cell).entity = ship0_id;
        map.at(mining_cell).energy = 100;
        store.map_total_energy = 100;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), ship0_id, ship1_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_ship0, cpu_ship1] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::MiningPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_ship0, gpu_ship1] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::TurnFrame turn_frame{};
    std::vector<hlt::gpu::MiningEffectFrameEntry> effects;
    hlt::gpu::run_cuda_mining(frame, turn_frame, config, &effects);

    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_ship0)].energy == cpu_store.get_entity(cpu_ship0).energy);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_ship1)].energy == cpu_store.get_entity(cpu_ship1).energy);
    REQUIRE(frame.cell_energy[hlt::cell_index(frame, hlt::Location{2, 2})] == cpu_map.at(hlt::Location{2, 2}).energy);
    REQUIRE(frame.map_total_energy == cpu_store.map_total_energy);

    const auto mined_effects = std::count_if(effects.begin(), effects.end(), [](const auto &effect) {
        return effect.entity != hlt::Entity::None;
    });
    REQUIRE(mined_effects == 2);
    for (const auto &effect : effects) {
        if (effect.entity == hlt::Entity::None) {
            continue;
        }
        REQUIRE(effect.location == hlt::Location{2, 2});
        REQUIRE(effect.extracted == 25);
        REQUIRE(effect.gained == 25);
    }
#endif
}

TEST_CASE("Cuda mining phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.mining_interference_range = 0;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &ship = store.new_entity(0, player.id);
    player.add_entity(ship.id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = ship.id;
    map.at(hlt::Location{2, 2}).energy = 100;
    store.map_total_energy = 100;

    hlt::StepResult result{};
    hlt::gpu::run_cuda_mining_phase(state, result, config);

    REQUIRE(store.get_entity(ship.id).energy > 0);
    REQUIRE(map.at(hlt::Location{2, 2}).energy < 100);
    REQUIRE(store.map_total_energy == static_cast<unsigned long long>(map.at(hlt::Location{2, 2}).energy));
#endif
}

TEST_CASE("Cuda plunder kernel matches PlunderPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.plunder_range = 2;
    config.ruleset.economy.plunder_halite_per_turn = 25;
    config.ruleset.economy.max_energy = 1000;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(8, 8);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;
        map.at(p0.factory).owner = p0.id;
        map.at(p1.factory).owner = p1.id;

        auto &near_enemy_factory = store.new_entity(100, p0.id);
        const auto near_id = near_enemy_factory.id;
        p0.add_entity(near_id, hlt::Location{4, 2});
        map.at(hlt::Location{4, 2}).entity = near_id;

        auto &far_ship = store.new_entity(100, p0.id);
        const auto far_id = far_ship.id;
        p0.add_entity(far_id, hlt::Location{1, 1});
        map.at(hlt::Location{1, 1}).entity = far_id;

        auto &capped_ship = store.new_entity(995, p1.id);
        const auto capped_id = capped_ship.id;
        p1.add_entity(capped_id, hlt::Location{0, 1});
        map.at(hlt::Location{0, 1}).entity = capped_id;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), near_id, far_id, capped_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_near, cpu_far, cpu_capped] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::PlunderPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_near, gpu_far, gpu_capped] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_plunder(frame, config);

    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_near)].energy == cpu_store.get_entity(cpu_near).energy);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_far)].energy == cpu_store.get_entity(cpu_far).energy);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_capped)].energy == cpu_store.get_entity(cpu_capped).energy);
#endif
}

TEST_CASE("Cuda plunder phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.plunder_range = 2;
    config.ruleset.economy.plunder_halite_per_turn = 25;

    hlt::Store store{};
    hlt::Map map(8, 8);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    auto &ship = store.new_entity(100, p0.id);
    p0.add_entity(ship.id, hlt::Location{4, 2});
    map.at(hlt::Location{4, 2}).entity = ship.id;
    map.at(p1.factory).owner = p1.id;

    hlt::gpu::run_cuda_plunder_phase(state, config);

    REQUIRE(store.get_entity(ship.id).energy == 125);
#endif
}

TEST_CASE("Cuda regen kernel matches RegenPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 0.50;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(4, 4);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        map.at(hlt::Location{1, 1}).initial_energy = 100;
        map.at(hlt::Location{1, 1}).energy = 20;
        map.at(hlt::Location{2, 2}).initial_energy = 100;
        map.at(hlt::Location{2, 2}).energy = 60;
        map.at(hlt::Location{3, 3}).owner = hlt::Player::id_type{0};
        map.at(hlt::Location{3, 3}).initial_energy = 100;
        map.at(hlt::Location{3, 3}).energy = 20;
        store.map_total_energy = 100;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics>{std::move(store), std::move(map), std::move(statistics)};
    };

    auto [cpu_store, cpu_map, cpu_stats] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::RegenPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_regen(frame, config);

    REQUIRE(frame.cell_energy[hlt::cell_index(frame, hlt::Location{1, 1})] == cpu_map.at(hlt::Location{1, 1}).energy);
    REQUIRE(frame.cell_energy[hlt::cell_index(frame, hlt::Location{2, 2})] == cpu_map.at(hlt::Location{2, 2}).energy);
    REQUIRE(frame.cell_energy[hlt::cell_index(frame, hlt::Location{3, 3})] == cpu_map.at(hlt::Location{3, 3}).energy);
    REQUIRE(frame.map_total_energy == cpu_store.map_total_energy);
#endif
}

TEST_CASE("Cuda regen phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 0.50;

    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    map.at(hlt::Location{1, 1}).initial_energy = 100;
    map.at(hlt::Location{1, 1}).energy = 20;
    store.map_total_energy = 20;

    hlt::gpu::run_cuda_regen_phase(state, config);

    REQUIRE(map.at(hlt::Location{1, 1}).energy == 30);
    REQUIRE(store.map_total_energy == 30);
#endif
}

TEST_CASE("Cuda over-ship tax kernel matches OverShipTaxPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.ship_income_per_turn = 10;
    config.ruleset.economy.over_ship_tax_threshold = 2;
    config.ruleset.economy.over_ship_tax_per_turn = 50;
    config.ruleset.economy.ship_count_target = 3;
    config.ruleset.economy.ship_count_deviation_penalty = 7;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        player.energy = 1000;
        player.factory_halite = 500;
        for (int i = 0; i < 4; ++i) {
            auto &ship = store.new_entity(0, player.id);
            const hlt::Location location{i, 1};
            player.add_entity(ship.id, location);
            map.at(location).entity = ship.id;
        }
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics>{std::move(store), std::move(map), std::move(statistics)};
    };

    auto [cpu_store, cpu_map, cpu_stats] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::OverShipTaxPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_over_ship_tax(frame, config);

    REQUIRE(frame.players[0].energy == cpu_store.players_ref().at(hlt::Player::id_type{0}).energy);
    REQUIRE(frame.players[0].factory_halite == cpu_store.players_ref().at(hlt::Player::id_type{0}).factory_halite);
#endif
}

TEST_CASE("Cuda over-ship tax phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.ship_income_per_turn = 10;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    player.factory_halite = 100;
    auto &ship = store.new_entity(0, player.id);
    player.add_entity(ship.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = ship.id;

    hlt::gpu::run_cuda_over_ship_tax_phase(state, config);

    REQUIRE(player.factory_halite == 110);
#endif
}

TEST_CASE("Cuda halite rebalance kernel matches HaliteRebalancePhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.halite_rebalance_enabled = true;
    config.ruleset.economy.halite_rebalance_period = 5;
    config.ruleset.economy.halite_rebalance_fraction = 0.5;
    config.ruleset.economy.halite_rebalance_min_gap_frac = 0.0;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        state.turn.number = 10;
        auto [leader_it, leader_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "leader"});
        auto [trailer_it, trailer_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "trailer"});
        REQUIRE(leader_inserted);
        REQUIRE(trailer_inserted);
        auto &leader = leader_it->second;
        auto &trailer = trailer_it->second;
        leader.energy = 700;
        leader.factory_halite = 100;
        leader.dropoffs.emplace_back(store.new_dropoff(hlt::Location{1, 0}));
        leader.dropoffs.back().halite_pool = 900;
        leader.dropoffs.back().destroyed = false;
        trailer.energy = 50;
        trailer.factory_halite = 100;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics>{std::move(store), std::move(map), std::move(statistics)};
    };

    auto [cpu_store, cpu_map, cpu_stats] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::HaliteRebalancePhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_halite_rebalance(frame, config);

    const auto leader_slot = frame.player_slot_by_id.at(hlt::Player::id_type{0});
    const auto trailer_slot = frame.player_slot_by_id.at(hlt::Player::id_type{1});
    REQUIRE(frame.players[leader_slot].energy == cpu_store.players_ref().at(hlt::Player::id_type{0}).energy);
    REQUIRE(frame.players[leader_slot].factory_halite == cpu_store.players_ref().at(hlt::Player::id_type{0}).factory_halite);
    REQUIRE(frame.dropoffs[frame.players[leader_slot].dropoff_begin].halite_pool == cpu_store.players_ref().at(hlt::Player::id_type{0}).dropoffs[0].halite_pool);
    REQUIRE(frame.players[trailer_slot].energy == cpu_store.players_ref().at(hlt::Player::id_type{1}).energy);
    REQUIRE(frame.players[trailer_slot].factory_halite == cpu_store.players_ref().at(hlt::Player::id_type{1}).factory_halite);
#endif
}

TEST_CASE("Cuda halite rebalance phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.halite_rebalance_enabled = true;
    config.ruleset.economy.halite_rebalance_period = 5;
    config.ruleset.economy.halite_rebalance_fraction = 1.0;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    state.turn.number = 10;
    auto [leader_it, leader_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "leader"});
    auto [trailer_it, trailer_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "trailer"});
    REQUIRE(leader_inserted);
    REQUIRE(trailer_inserted);
    auto &leader = leader_it->second;
    auto &trailer = trailer_it->second;
    leader.energy = 500;
    leader.factory_halite = 500;
    trailer.energy = 0;
    trailer.factory_halite = 0;
    trailer.factory_destroyed = true;
    trailer.dropoffs.emplace_back(store.new_dropoff(hlt::Location{3, 4}));
    trailer.dropoffs.back().halite_pool = 25;
    trailer.dropoffs.back().destroyed = false;

    hlt::gpu::run_cuda_halite_rebalance_phase(state, config);

    REQUIRE(leader.energy == 263);
    REQUIRE(leader.factory_halite == 263);
    REQUIRE(trailer.energy == 237);
    REQUIRE(trailer.dropoffs[0].halite_pool == 262);
#endif
}

TEST_CASE("Cuda capture decisions match CapturePhase ownership decision", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        state.turn.number = 4;
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;

        auto &target = store.new_entity(123, p0.id);
        const auto target_id = target.id;
        p0.add_entity(target_id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = target_id;

        auto &enemy_a = store.new_entity(0, p1.id);
        const auto enemy_a_id = enemy_a.id;
        p1.add_entity(enemy_a_id, hlt::Location{2, 1});
        map.at(hlt::Location{2, 1}).entity = enemy_a_id;

        auto &enemy_b = store.new_entity(0, p1.id);
        const auto enemy_b_id = enemy_b.id;
        p1.add_entity(enemy_b_id, hlt::Location{2, 3});
        map.at(hlt::Location{2, 3}).entity = enemy_b_id;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), target_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_target] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::CapturePhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_target] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    const auto decisions = hlt::gpu::run_cuda_capture_decisions(hlt::capture_state_frame(gpu_state), config);

    const auto captured = std::find_if(decisions.begin(), decisions.end(), [&](const auto &decision) {
        return decision.entity == gpu_target;
    });
    REQUIRE(captured != decisions.end());
    REQUIRE(captured->should_capture);
    REQUIRE(captured->new_owner == hlt::Player::id_type{1});
    REQUIRE(cpu_store.entities_ref().find(cpu_target) == cpu_store.entities_ref().end());
    REQUIRE(cpu_map.at(hlt::Location{2, 2}).entity != hlt::Entity::None);
    REQUIRE(cpu_store.get_entity(cpu_map.at(hlt::Location{2, 2}).entity).owner == captured->new_owner);
#endif
}

TEST_CASE("Cuda turn backend matches CPU core state on hybrid phases", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = true;
    config.ruleset.inspiration.radius = 2;
    config.ruleset.inspiration.ship_count = 1;
    config.ruleset.economy.cell_regen_enabled = true;
    config.ruleset.economy.cell_regen_rate = 0.10;
    config.ruleset.economy.cell_regen_cap_fraction = 1.0;
    config.ruleset.economy.mining_interference_range = 0;
    config.ruleset.economy.ship_income_per_turn = 5;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        statistics.player_statistics.emplace_back(hlt::Player::id_type{1}, 1);
        hlt::GameState state(store, map, statistics);
        state.turn.number = 7;

        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;
        p0.energy = 5000;
        p0.factory_halite = 1000;
        p1.energy = 5000;
        p1.factory_halite = 1000;

        map.at(p0.factory).owner = p0.id;
        map.at(p1.factory).owner = p1.id;

        auto &miner = store.new_entity(0, p0.id);
        const auto miner_id = miner.id;
        p0.add_entity(miner_id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = miner_id;
        map.at(hlt::Location{2, 2}).energy = 100;
        map.at(hlt::Location{2, 2}).initial_energy = 120;

        auto &enemy = store.new_entity(0, p1.id);
        const auto enemy_id = enemy.id;
        p1.add_entity(enemy_id, hlt::Location{2, 3});
        map.at(hlt::Location{2, 3}).entity = enemy_id;
        map.at(hlt::Location{2, 3}).energy = 50;
        map.at(hlt::Location{2, 3}).initial_energy = 80;

        map.at(hlt::Location{1, 1}).energy = 20;
        map.at(hlt::Location{1, 1}).initial_energy = 100;
        store.map_total_energy = 170;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), miner_id, enemy_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_miner, cpu_enemy] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    hlt::DefendCommand cpu_defend{cpu_miner};
    cpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::DefendCommand>(cpu_defend));
    hlt::events::RecordingEventSink cpu_sink{};
    hlt::InlineExecutor cpu_executor{};
    hlt::TurnEngine cpu_engine(config);
    auto cpu_result = cpu_engine.step(cpu_state, cpu_actions, cpu_sink, cpu_executor);

    auto [gpu_store, gpu_map, gpu_stats, gpu_miner, gpu_enemy] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::ActionBatch gpu_actions{};
    hlt::DefendCommand gpu_defend{gpu_miner};
    gpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::DefendCommand>(gpu_defend));
    hlt::events::RecordingEventSink gpu_sink{};
    hlt::InlineExecutor gpu_executor{};
    hlt::TurnEngine gpu_cpu_reference(config);
    auto gpu_backend = hlt::gpu::make_cuda_turn_backend(gpu_cpu_reference, config);
    REQUIRE(gpu_backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto gpu_result = gpu_backend->step(gpu_state, gpu_actions, gpu_sink, gpu_executor, &profile);

    REQUIRE(gpu_backend->backend_name() == "cuda");
    REQUIRE_FALSE(profile.phase_timings.empty());
    REQUIRE(cpu_store.get_entity(cpu_miner).energy == gpu_store.get_entity(gpu_miner).energy);
    REQUIRE(cpu_store.get_entity(cpu_miner).is_defending == gpu_store.get_entity(gpu_miner).is_defending);
    REQUIRE(cpu_store.get_entity(cpu_enemy).is_inspired == gpu_store.get_entity(gpu_enemy).is_inspired);
    REQUIRE(cpu_map.at(hlt::Location{2, 2}).energy == gpu_map.at(hlt::Location{2, 2}).energy);
    REQUIRE(cpu_map.at(hlt::Location{1, 1}).energy == gpu_map.at(hlt::Location{1, 1}).energy);
    REQUIRE(cpu_store.map_total_energy == gpu_store.map_total_energy);
    REQUIRE(cpu_stats.player_statistics[0].total_mined == gpu_stats.player_statistics[0].total_mined);
    REQUIRE(cpu_sink.events().size() == gpu_sink.events().size());
    REQUIRE(cpu_result.changed_cells == gpu_result.changed_cells);
#endif
}

TEST_CASE("Cuda turn backend resolves construction through GPU decisions", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = false;
    config.ruleset.economy.dropoff_cost = 4000;
    config.ruleset.economy.dropoff_cost_growth = 0.25;
    config.ruleset.economy.initial_dropoff_halite = 50;
    config.ruleset.economy.cell_regen_enabled = false;
    config.ruleset.economy.ship_income_per_turn = 0;
    config.ruleset.capture.enabled = false;

    auto build_state = [&] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        hlt::GameState state(store, map, statistics);
        state.turn.number = 3;
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        player.energy = 6000;
        map.at(player.factory).owner = player.id;
        player.dropoffs.emplace_back(store.new_dropoff(hlt::Location{1, 0}));

        auto &ship = store.new_entity(700, player.id);
        const auto ship_id = ship.id;
        player.add_entity(ship_id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = ship_id;
        map.at(hlt::Location{2, 2}).energy = 300;
        map.at(hlt::Location{2, 2}).initial_energy = 300;
        store.map_total_energy = 300;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), ship_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_ship] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    cpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::ConstructCommand>(hlt::ConstructCommand{cpu_ship}));
    hlt::events::RecordingEventSink cpu_sink{};
    hlt::InlineExecutor cpu_executor{};
    hlt::TurnEngine cpu_engine(config);
    auto cpu_result = cpu_engine.step(cpu_state, cpu_actions, cpu_sink, cpu_executor);

    auto [gpu_store, gpu_map, gpu_stats, gpu_ship] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::ActionBatch gpu_actions{};
    gpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::ConstructCommand>(hlt::ConstructCommand{gpu_ship}));
    hlt::events::RecordingEventSink gpu_sink{};
    hlt::InlineExecutor gpu_executor{};
    hlt::TurnEngine gpu_cpu_reference(config);
    auto gpu_backend = hlt::gpu::make_cuda_turn_backend(gpu_cpu_reference, config);
    REQUIRE(gpu_backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto gpu_result = gpu_backend->step(gpu_state, gpu_actions, gpu_sink, gpu_executor, &profile);

    const auto construction_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cuda:construction-decision";
    });
    REQUIRE(construction_profile != profile.phase_timings.end());
    const auto fallback_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cpu-fallback:construction";
    });
    REQUIRE(fallback_profile == profile.phase_timings.end());
    REQUIRE(cpu_store.entities_ref().find(cpu_ship) == cpu_store.entities_ref().end());
    REQUIRE(gpu_store.entities_ref().find(gpu_ship) == gpu_store.entities_ref().end());
    REQUIRE(cpu_map.at(hlt::Location{2, 2}).owner == gpu_map.at(hlt::Location{2, 2}).owner);
    REQUIRE(cpu_map.at(hlt::Location{2, 2}).energy == gpu_map.at(hlt::Location{2, 2}).energy);
    REQUIRE(cpu_store.get_player(hlt::Player::id_type{0}).dropoffs.size() == gpu_store.get_player(hlt::Player::id_type{0}).dropoffs.size());
    REQUIRE(cpu_store.get_player(hlt::Player::id_type{0}).energy == gpu_store.get_player(hlt::Player::id_type{0}).energy);
    REQUIRE(cpu_store.map_total_energy == gpu_store.map_total_energy);
    REQUIRE(cpu_sink.events().size() == gpu_sink.events().size());
    REQUIRE(cpu_result.changed_cells == gpu_result.changed_cells);
#endif
}

TEST_CASE("Cuda spawn decisions match SpawnPhase planning inputs", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.new_entity_energy_cost = 1000;
    config.ruleset.economy.spawn_cost_growth = 0.5;
    config.ruleset.economy.spawn_quad_threshold = 1;
    config.ruleset.economy.spawn_quad_growth = 0.25;
    config.ruleset.economy.emergency_spawn_enabled = true;
    config.ruleset.economy.emergency_spawn_period = 5;
    config.ruleset.economy.emergency_spawn_count = 2;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    state.turn.number = 5;
    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{3, 3}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    p0.energy = 5000;
    p1.energy = 100;
    map.at(p0.factory).owner = p0.id;
    map.at(p1.factory).owner = p1.id;

    auto &existing = store.new_entity(10, p0.id);
    p0.add_entity(existing.id, p0.factory);
    map.at(p0.factory).entity = existing.id;

    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{p0.id,
                                                 hlt::Entity::None,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Spawn});

    const auto decisions = hlt::gpu::run_cuda_spawn_decisions(hlt::capture_state_frame(state), commands, config);
    REQUIRE(decisions.spawns.size() == 1);
    REQUIRE(decisions.spawns[0].command);
    REQUIRE(decisions.spawns[0].factory == p0.factory);
    REQUIRE(decisions.spawns[0].cost == 1750);
    REQUIRE(decisions.spawns[0].factory_occupied);
    REQUIRE(decisions.spawns[0].existing_entity == existing.id);
    REQUIRE(decisions.spawns[0].self_collision);

    const auto emergency = std::find_if(decisions.emergency_spawns.begin(), decisions.emergency_spawns.end(), [&](const auto &decision) {
        return decision.player == p1.id;
    });
    REQUIRE(emergency != decisions.emergency_spawns.end());
    REQUIRE(emergency->count == 2);
    REQUIRE(emergency->locations[0] == p1.factory);
    REQUIRE(emergency->locations[1] == hlt::Location{4, 3});
#endif
}

TEST_CASE("Cuda turn backend resolves spawn through GPU decisions", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = false;
    config.ruleset.economy.new_entity_energy_cost = 1000;
    config.ruleset.economy.spawn_cost_growth = 0.25;
    config.ruleset.economy.spawn_quad_threshold = 100;
    config.ruleset.economy.spawn_quad_growth = 0.0;
    config.ruleset.economy.cell_regen_enabled = false;
    config.ruleset.economy.ship_income_per_turn = 0;
    config.ruleset.economy.emergency_spawn_enabled = false;
    config.ruleset.capture.enabled = false;

    auto build_state = [&] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        hlt::GameState state(store, map, statistics);
        state.turn.number = 4;
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        player.energy = 5000;
        map.at(player.factory).owner = player.id;
        auto &existing = store.new_entity(25, player.id);
        const auto existing_id = existing.id;
        player.add_entity(existing_id, player.factory);
        map.at(player.factory).entity = existing_id;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), existing_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_existing] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    cpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::SpawnCommand>(hlt::SpawnCommand{}));
    hlt::events::RecordingEventSink cpu_sink{};
    hlt::InlineExecutor cpu_executor{};
    hlt::TurnEngine cpu_engine(config);
    auto cpu_result = cpu_engine.step(cpu_state, cpu_actions, cpu_sink, cpu_executor);

    auto [gpu_store, gpu_map, gpu_stats, gpu_existing] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::ActionBatch gpu_actions{};
    gpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::SpawnCommand>(hlt::SpawnCommand{}));
    hlt::events::RecordingEventSink gpu_sink{};
    hlt::InlineExecutor gpu_executor{};
    hlt::TurnEngine gpu_cpu_reference(config);
    auto gpu_backend = hlt::gpu::make_cuda_turn_backend(gpu_cpu_reference, config);
    REQUIRE(gpu_backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto gpu_result = gpu_backend->step(gpu_state, gpu_actions, gpu_sink, gpu_executor, &profile);

    const auto spawn_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cuda:spawn-decision";
    });
    REQUIRE(spawn_profile != profile.phase_timings.end());
    const auto fallback_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cpu-fallback:spawn";
    });
    REQUIRE(fallback_profile == profile.phase_timings.end());
    REQUIRE(cpu_store.entities_ref().find(cpu_existing) == cpu_store.entities_ref().end());
    REQUIRE(gpu_store.entities_ref().find(gpu_existing) == gpu_store.entities_ref().end());
    REQUIRE(cpu_map.at(hlt::Location{0, 0}).entity == gpu_map.at(hlt::Location{0, 0}).entity);
    REQUIRE(cpu_store.get_player(hlt::Player::id_type{0}).energy == gpu_store.get_player(hlt::Player::id_type{0}).energy);
    REQUIRE(cpu_sink.events().size() == gpu_sink.events().size());
    REQUIRE(cpu_result.changed_cells == gpu_result.changed_cells);
#endif
}

TEST_CASE("Cuda movement decisions match MovementPhase planning inputs", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.move_cost_ratio = 10;
    config.ruleset.inspiration.move_cost_ratio = 2;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;

    auto &ship = store.new_entity(30, player.id);
    const auto ship_id = ship.id;
    ship.is_inspired = true;
    player.add_entity(ship_id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = ship_id;
    map.at(hlt::Location{2, 2}).energy = 50;

    auto &poor_ship = store.new_entity(1, player.id);
    const auto poor_ship_id = poor_ship.id;
    player.add_entity(poor_ship_id, hlt::Location{4, 4});
    map.at(hlt::Location{4, 4}).entity = poor_ship_id;
    map.at(hlt::Location{4, 4}).energy = 50;

    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::North,
                                                 hlt::FlatCommandType::Move});
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 poor_ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::West,
                                                 hlt::FlatCommandType::Move});

    const auto decisions = hlt::gpu::run_cuda_movement_decisions(hlt::capture_state_frame(state), commands, config);
    REQUIRE(decisions.size() == 2);
    REQUIRE(decisions[0].command);
    REQUIRE_FALSE(decisions[0].entity_missing);
    REQUIRE_FALSE(decisions[0].insufficient_energy);
    REQUIRE(decisions[0].from == hlt::Location{2, 2});
    REQUIRE(decisions[0].to == hlt::Location{2, 1});
    REQUIRE(decisions[0].required == 25);
    REQUIRE(decisions[1].command);
    REQUIRE(decisions[1].insufficient_energy);
    REQUIRE(decisions[1].from == hlt::Location{4, 4});
    REQUIRE(decisions[1].to == hlt::Location{3, 4});
    REQUIRE(decisions[1].required == 5);
#endif
}

TEST_CASE("Cuda turn backend resolves movement through GPU decisions", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = false;
    config.ruleset.economy.move_cost_ratio = 10;
    config.ruleset.economy.cell_regen_enabled = false;
    config.ruleset.economy.ship_income_per_turn = 0;
    config.ruleset.economy.emergency_spawn_enabled = false;
    config.ruleset.capture.enabled = false;
    config.ruleset.combat.collision_hp_damage = 255;

    auto build_state = [&] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        statistics.player_statistics.emplace_back(hlt::Player::id_type{1}, 1);
        hlt::GameState state(store, map, statistics);
        state.turn.number = 6;
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;
        map.at(p0.factory).owner = p0.id;
        map.at(p1.factory).owner = p1.id;

        auto &ship0 = store.new_entity(30, p0.id);
        const auto ship0_id = ship0.id;
        p0.add_entity(ship0_id, hlt::Location{1, 2});
        map.at(hlt::Location{1, 2}).entity = ship0_id;
        map.at(hlt::Location{1, 2}).energy = 20;

        auto &ship1 = store.new_entity(40, p1.id);
        const auto ship1_id = ship1.id;
        p1.add_entity(ship1_id, hlt::Location{3, 2});
        map.at(hlt::Location{3, 2}).entity = ship1_id;
        map.at(hlt::Location{3, 2}).energy = 10;

        store.map_total_energy = 30;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), ship0_id, ship1_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_ship0, cpu_ship1] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    cpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::MoveCommand>(hlt::MoveCommand{cpu_ship0, hlt::Direction::East}));
    cpu_actions[hlt::Player::id_type{1}].push_back(std::make_unique<hlt::MoveCommand>(hlt::MoveCommand{cpu_ship1, hlt::Direction::West}));
    hlt::events::RecordingEventSink cpu_sink{};
    hlt::InlineExecutor cpu_executor{};
    hlt::TurnEngine cpu_engine(config);
    auto cpu_result = cpu_engine.step(cpu_state, cpu_actions, cpu_sink, cpu_executor);

    auto [gpu_store, gpu_map, gpu_stats, gpu_ship0, gpu_ship1] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::ActionBatch gpu_actions{};
    gpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::MoveCommand>(hlt::MoveCommand{gpu_ship0, hlt::Direction::East}));
    gpu_actions[hlt::Player::id_type{1}].push_back(std::make_unique<hlt::MoveCommand>(hlt::MoveCommand{gpu_ship1, hlt::Direction::West}));
    hlt::events::RecordingEventSink gpu_sink{};
    hlt::InlineExecutor gpu_executor{};
    hlt::TurnEngine gpu_cpu_reference(config);
    auto gpu_backend = hlt::gpu::make_cuda_turn_backend(gpu_cpu_reference, config);
    REQUIRE(gpu_backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto gpu_result = gpu_backend->step(gpu_state, gpu_actions, gpu_sink, gpu_executor, &profile);

    const auto movement_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cuda:movement-decision";
    });
    REQUIRE(movement_profile != profile.phase_timings.end());
    const auto fallback_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cpu-fallback:movement";
    });
    REQUIRE(fallback_profile == profile.phase_timings.end());
    REQUIRE(cpu_store.entities_ref().size() == gpu_store.entities_ref().size());
    REQUIRE(cpu_map.at(hlt::Location{2, 2}).entity == gpu_map.at(hlt::Location{2, 2}).entity);
    REQUIRE(cpu_store.map_total_energy == gpu_store.map_total_energy);
    REQUIRE(cpu_sink.events().size() == gpu_sink.events().size());
    REQUIRE(cpu_result.changed_cells == gpu_result.changed_cells);
#endif
}

TEST_CASE("Cuda combat decisions match CombatPhase planning inputs", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.combat.attack_range = 2;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
    REQUIRE(p0_inserted);
    REQUIRE(p1_inserted);
    auto &p0 = p0_it->second;
    auto &p1 = p1_it->second;
    map.at(p0.factory).owner = p0.id;
    map.at(p1.factory).owner = p1.id;

    auto &attacker = store.new_entity(50, p0.id);
    const auto attacker_id = attacker.id;
    p0.add_entity(attacker_id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = attacker_id;

    auto &target = store.new_entity(60, p1.id);
    const auto target_id = target.id;
    p1.add_entity(target_id, hlt::Location{2, 3});
    map.at(hlt::Location{2, 3}).entity = target_id;

    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{p0.id,
                                                 attacker_id,
                                                 target_id,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::AttackShip});
    commands.commands.push_back(hlt::FlatCommand{p0.id,
                                                 attacker_id,
                                                 hlt::Entity::None,
                                                 p1.id,
                                                 p1.factory,
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::AttackStructure});

    const auto decisions = hlt::gpu::run_cuda_combat_decisions(hlt::capture_state_frame(state), commands, config);
    REQUIRE(decisions.size() == 2);
    REQUIRE(decisions[0].command);
    REQUIRE(decisions[0].valid_ship);
    REQUIRE_FALSE(decisions[0].invalid);
    REQUIRE(decisions[0].attacker_location == hlt::Location{2, 2});
    REQUIRE(decisions[0].target_location == hlt::Location{2, 3});
    REQUIRE(decisions[0].target_owner == p1.id);
    REQUIRE(decisions[1].command);
    REQUIRE(decisions[1].structure_target);
    REQUIRE(decisions[1].invalid);
#endif
}

TEST_CASE("Cuda validation decisions match ValidationPhase command classification", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.economy.new_entity_energy_cost = 1000;
    config.ruleset.economy.spawn_cost_growth = 0.5;
    config.ruleset.economy.dropoff_cost = 4000;
    config.ruleset.economy.dropoff_cost_growth = 0.25;
    config.ruleset.combat.enable_combat_commands = true;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    player.energy = 5000;
    player.dropoffs.emplace_back(store.new_dropoff(hlt::Location{1, 0}));
    auto &ship = store.new_entity(100, player.id);
    const auto ship_id = ship.id;
    player.add_entity(ship_id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = ship_id;
    map.at(hlt::Location{2, 2}).energy = 500;

    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::North,
                                                 hlt::FlatCommandType::Move});
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 ship_id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Construct});
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 hlt::Entity::None,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Spawn});
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 hlt::Entity::id_type{9999},
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Defend});

    const auto decisions = hlt::gpu::run_cuda_validation_decisions(hlt::capture_state_frame(state), commands, config);
    REQUIRE(decisions.size() == 4);
    REQUIRE(decisions[0].include_in_batch);
    REQUIRE(decisions[0].occurrence_command);
    REQUIRE(decisions[0].ownership_ok);
    REQUIRE(decisions[1].include_in_batch);
    REQUIRE(decisions[1].expense_command);
    REQUIRE(decisions[1].expense == 4400);
    REQUIRE(decisions[2].include_in_batch);
    REQUIRE(decisions[2].expense == 1500);
    REQUIRE_FALSE(decisions[3].include_in_batch);
    REQUIRE(decisions[3].combat_enabled);
    REQUIRE_FALSE(decisions[3].ownership_ok);
#endif
}

TEST_CASE("Cuda turn backend resolves combat through GPU decisions", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = false;
    config.ruleset.economy.cell_regen_enabled = false;
    config.ruleset.economy.ship_income_per_turn = 0;
    config.ruleset.economy.emergency_spawn_enabled = false;
    config.ruleset.capture.enabled = false;
    config.ruleset.combat.attack_range = 2;
    config.ruleset.combat.attack_hp_damage = 255;
    config.ruleset.combat.attack_halite_steal_ratio = 0.5;
    config.ruleset.combat.enable_attacker_self_damage = false;
    config.ruleset.combat.kill_credit_to_attacker = true;
    config.ruleset.combat.kill_halite_bonus_ratio = 0.25;

    auto build_state = [&] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
        statistics.player_statistics.emplace_back(hlt::Player::id_type{1}, 1);
        hlt::GameState state(store, map, statistics);
        state.turn.number = 8;
        auto [p0_it, p0_inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        auto [p1_it, p1_inserted] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{4, 4}, "p1"});
        REQUIRE(p0_inserted);
        REQUIRE(p1_inserted);
        auto &p0 = p0_it->second;
        auto &p1 = p1_it->second;
        p0.energy = 100;
        p1.energy = 100;
        map.at(p0.factory).owner = p0.id;
        map.at(p1.factory).owner = p1.id;

        auto &attacker = store.new_entity(50, p0.id);
        const auto attacker_id = attacker.id;
        p0.add_entity(attacker_id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = attacker_id;

        auto &target = store.new_entity(80, p1.id);
        const auto target_id = target.id;
        p1.add_entity(target_id, hlt::Location{2, 3});
        map.at(hlt::Location{2, 3}).entity = target_id;
        store.map_total_energy = 0;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), attacker_id, target_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_attacker, cpu_target] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    cpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::AttackCommand>(hlt::AttackCommand{cpu_attacker, cpu_target}));
    hlt::events::RecordingEventSink cpu_sink{};
    hlt::InlineExecutor cpu_executor{};
    hlt::TurnEngine cpu_engine(config);
    auto cpu_result = cpu_engine.step(cpu_state, cpu_actions, cpu_sink, cpu_executor);

    auto [gpu_store, gpu_map, gpu_stats, gpu_attacker, gpu_target] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::ActionBatch gpu_actions{};
    gpu_actions[hlt::Player::id_type{0}].push_back(std::make_unique<hlt::AttackCommand>(hlt::AttackCommand{gpu_attacker, gpu_target}));
    hlt::events::RecordingEventSink gpu_sink{};
    hlt::InlineExecutor gpu_executor{};
    hlt::TurnEngine gpu_cpu_reference(config);
    auto gpu_backend = hlt::gpu::make_cuda_turn_backend(gpu_cpu_reference, config);
    REQUIRE(gpu_backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto gpu_result = gpu_backend->step(gpu_state, gpu_actions, gpu_sink, gpu_executor, &profile);

    const auto combat_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cuda:combat-decision";
    });
    REQUIRE(combat_profile != profile.phase_timings.end());
    const auto fallback_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cpu-fallback:combat";
    });
    REQUIRE(fallback_profile == profile.phase_timings.end());
    REQUIRE(cpu_store.entities_ref().size() == gpu_store.entities_ref().size());
    REQUIRE(cpu_store.entities_ref().find(cpu_target) == cpu_store.entities_ref().end());
    REQUIRE(gpu_store.entities_ref().find(gpu_target) == gpu_store.entities_ref().end());
    REQUIRE(cpu_store.get_player(hlt::Player::id_type{0}).energy == gpu_store.get_player(hlt::Player::id_type{0}).energy);
    REQUIRE(cpu_store.map_total_energy == gpu_store.map_total_energy);
    REQUIRE(cpu_sink.events().size() == gpu_sink.events().size());
    REQUIRE(cpu_result.changed_cells == gpu_result.changed_cells);
#endif
}

TEST_CASE("Cuda turn backend has no CPU fallback phases on validated turn", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.inspiration.enabled = false;
    config.ruleset.economy.cell_regen_enabled = false;
    config.ruleset.economy.ship_income_per_turn = 0;
    config.ruleset.economy.emergency_spawn_enabled = false;
    config.ruleset.capture.enabled = false;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    player.energy = 5000;
    map.at(player.factory).owner = player.id;
    auto &ship = store.new_entity(100, player.id);
    const auto ship_id = ship.id;
    player.add_entity(ship_id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = ship_id;

    hlt::ActionBatch actions{};
    actions[player.id].push_back(std::make_unique<hlt::MoveCommand>(hlt::MoveCommand{ship_id, hlt::Direction::North}));
    hlt::events::RecordingEventSink sink{};
    hlt::InlineExecutor executor{};
    hlt::TurnEngine cpu_reference(config);
    auto backend = hlt::gpu::make_cuda_turn_backend(cpu_reference, config);
    REQUIRE(backend != nullptr);
    hlt::TurnExecutionProfile profile{};
    auto result = backend->step(state, actions, sink, executor, &profile);

    REQUIRE(result.validated_commands.has_value());
    const auto fallback_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name.find("cpu-fallback:") == 0;
    });
    REQUIRE(fallback_profile == profile.phase_timings.end());
    const auto validation_profile = std::find_if(profile.phase_timings.begin(), profile.phase_timings.end(), [](const auto &entry) {
        return entry.phase_name == "cuda:validation-decision";
    });
    REQUIRE(validation_profile != profile.phase_timings.end());
#endif
}

TEST_CASE("Cuda defend kernel matches DefendPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(4, 4);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        auto &protected_ship = store.new_entity(0, player.id);
        protected_ship.protection_turns = 2;
        protected_ship.is_defending = true;
        player.add_entity(protected_ship.id, hlt::Location{1, 1});
        map.at(hlt::Location{1, 1}).entity = protected_ship.id;
        auto &commanded_ship = store.new_entity(0, player.id);
        player.add_entity(commanded_ship.id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = commanded_ship.id;
        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), protected_ship.id, commanded_ship.id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_protected, cpu_commanded] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch cpu_actions{};
    hlt::StepResult cpu_result{};
    cpu_result.validated_commands = hlt::CommandBatch{};
    hlt::DefendCommand cpu_defend{cpu_commanded};
    cpu_result.validated_commands->defends[hlt::Player::id_type{0}].push_back(std::cref(cpu_defend));
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, cpu_actions, cpu_result, config, sink);
    hlt::rules::phases::DefendPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_protected, gpu_commanded] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    hlt::CommandBatch command_batch{};
    hlt::DefendCommand gpu_defend{gpu_commanded};
    command_batch.defends[hlt::Player::id_type{0}].push_back(std::cref(gpu_defend));
    auto frame = hlt::capture_state_frame(gpu_state);
    const auto commands = hlt::flatten_command_batch(command_batch);
    hlt::gpu::run_cuda_defend(frame, commands, config);

    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_protected)].is_defending == cpu_store.get_entity(cpu_protected).is_defending);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_protected)].protection_turns == cpu_store.get_entity(cpu_protected).protection_turns);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_commanded)].is_defending == cpu_store.get_entity(cpu_commanded).is_defending);
#endif
}

TEST_CASE("Cuda defend phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &ship = store.new_entity(0, player.id);
    player.add_entity(ship.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = ship.id;
    hlt::CommandFrame commands;
    commands.commands.push_back(hlt::FlatCommand{player.id,
                                                 ship.id,
                                                 hlt::Entity::None,
                                                 hlt::Player::None,
                                                 hlt::Location{0, 0},
                                                 hlt::Direction::Still,
                                                 hlt::FlatCommandType::Defend});

    hlt::gpu::run_cuda_defend_phase(state, commands, config);

    REQUIRE(store.get_entity(ship.id).is_defending);
#endif
}

TEST_CASE("Cuda dump kernel matches DumpPhase core state", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.combat.initial_hp = 64;

    auto build_state = [] {
        hlt::Store store{};
        hlt::Map map(5, 5);
        hlt::GameStatistics statistics{};
        hlt::GameState state(store, map, statistics);
        auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
        REQUIRE(inserted);
        auto &player = player_it->second;
        player.energy = 100;
        player.factory_halite = 200;
        map.at(player.factory).owner = player.id;
        auto &factory_ship = store.new_entity(75, player.id);
        const auto factory_ship_id = factory_ship.id;
        factory_ship.hp = 10;
        player.add_entity(factory_ship_id, player.factory);
        map.at(player.factory).entity = factory_ship_id;

        player.dropoffs.emplace_back(store.new_dropoff(hlt::Location{2, 2}));
        player.dropoffs.back().halite_pool = 50;
        map.at(hlt::Location{2, 2}).owner = player.id;
        auto &dropoff_ship = store.new_entity(25, player.id);
        const auto dropoff_ship_id = dropoff_ship.id;
        player.add_entity(dropoff_ship_id, hlt::Location{2, 2});
        map.at(hlt::Location{2, 2}).entity = dropoff_ship_id;

        auto &field_ship = store.new_entity(30, player.id);
        const auto field_ship_id = field_ship.id;
        player.add_entity(field_ship_id, hlt::Location{3, 3});
        map.at(hlt::Location{3, 3}).entity = field_ship_id;

        return std::tuple<hlt::Store, hlt::Map, hlt::GameStatistics, hlt::Entity::id_type, hlt::Entity::id_type, hlt::Entity::id_type>{
            std::move(store), std::move(map), std::move(statistics), factory_ship_id, dropoff_ship_id, field_ship_id};
    };

    auto [cpu_store, cpu_map, cpu_stats, cpu_factory_ship, cpu_dropoff_ship, cpu_field_ship] = build_state();
    hlt::GameState cpu_state(cpu_store, cpu_map, cpu_stats);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    hlt::rules::RuleContext context(cpu_state, actions, result, config, sink);
    hlt::rules::phases::DumpPhase phase;
    phase.execute(context);

    auto [gpu_store, gpu_map, gpu_stats, gpu_factory_ship, gpu_dropoff_ship, gpu_field_ship] = build_state();
    hlt::GameState gpu_state(gpu_store, gpu_map, gpu_stats);
    auto frame = hlt::capture_state_frame(gpu_state);
    hlt::gpu::run_cuda_dump(frame, config);

    const auto player_slot = frame.player_slot_by_id.at(hlt::Player::id_type{0});
    REQUIRE(frame.players[player_slot].energy == cpu_store.players_ref().at(hlt::Player::id_type{0}).energy);
    REQUIRE(frame.players[player_slot].factory_halite == cpu_store.players_ref().at(hlt::Player::id_type{0}).factory_halite);
    REQUIRE(frame.players[player_slot].factory_energy_deposited == cpu_store.players_ref().at(hlt::Player::id_type{0}).factory_energy_deposited);
    REQUIRE(frame.players[player_slot].total_energy_deposited == cpu_store.players_ref().at(hlt::Player::id_type{0}).total_energy_deposited);
    REQUIRE(frame.dropoffs[frame.players[player_slot].dropoff_begin].halite_pool == cpu_store.players_ref().at(hlt::Player::id_type{0}).dropoffs[0].halite_pool);
    REQUIRE(frame.dropoffs[frame.players[player_slot].dropoff_begin].deposited_halite == cpu_store.players_ref().at(hlt::Player::id_type{0}).dropoffs[0].deposited_halite);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_factory_ship)].energy == cpu_store.get_entity(cpu_factory_ship).energy);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_factory_ship)].hp == cpu_store.get_entity(cpu_factory_ship).hp);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_dropoff_ship)].energy == cpu_store.get_entity(cpu_dropoff_ship).energy);
    REQUIRE(frame.entities[frame.entity_slot_by_id.at(gpu_field_ship)].energy == cpu_store.get_entity(cpu_field_ship).energy);
#endif
}

TEST_CASE("Cuda dump phase runner writes back to GameState", "[modernization][gpu][cuda]") {
#ifndef HALITE_ENABLE_CUDA
    SUCCEED();
    return;
#else
    auto config = hlt::GameConfig::from_constants();
    config.ruleset.combat.initial_hp = 64;

    hlt::Store store{};
    hlt::Map map(5, 5);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "p0"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &ship = store.new_entity(75, player.id);
    ship.hp = 10;
    player.add_entity(ship.id, player.factory);
    map.at(player.factory).owner = player.id;
    map.at(player.factory).entity = ship.id;

    hlt::gpu::run_cuda_dump_phase(state, config);

    REQUIRE(store.get_entity(ship.id).energy == 0);
    REQUIRE(store.get_entity(ship.id).hp == 64);
    REQUIRE(player.energy == 75);
    REQUIRE(player.factory_halite == 5075);
#endif
}

TEST_CASE("Validation phase can run through RuleContext", "[modernization][rules]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();
    hlt::rules::RuleContext context(state, actions, result, config, sink);
    hlt::rules::phases::ValidationPhase phase;

    phase.execute(context);
    REQUIRE(phase.name() == "validation");
    REQUIRE(context.executor() == nullptr);
}

TEST_CASE("Validation phase merges per-player shard results into batch", "[modernization][rules]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &entity = store.new_entity(0, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;

    std::vector<std::unique_ptr<hlt::Command>> player_commands;
    player_commands.push_back(std::make_unique<hlt::MoveCommand>(entity.id, hlt::Direction::North));
    actions[player.id] = std::move(player_commands);

    hlt::rules::RuleContext context(state, actions, result, config, sink);
    hlt::rules::phases::ValidationPhase phase;
    phase.execute(context);

    REQUIRE(result.non_fatal_errors.empty());
    REQUIRE(result.validated_commands.has_value());
    REQUIRE(result.validated_commands->moves[player.id].size() == 1);
}

TEST_CASE("Validation phase uses provided executor", "[modernization][rules]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();

    auto [player0_it, inserted0] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot0"});
    REQUIRE(inserted0);
    auto &player0 = player0_it->second;
    auto &entity0 = store.new_entity(0, player0.id);
    player0.add_entity(entity0.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity0.id;
    std::vector<std::unique_ptr<hlt::Command>> commands0;
    commands0.push_back(std::make_unique<hlt::MoveCommand>(entity0.id, hlt::Direction::North));
    actions[player0.id] = std::move(commands0);

    auto [player1_it, inserted1] = store.players_ref().emplace(hlt::Player::id_type{1}, hlt::Player{hlt::Player::id_type{1}, hlt::Location{3, 3}, "bot1"});
    REQUIRE(inserted1);
    auto &player1 = player1_it->second;
    auto &entity1 = store.new_entity(0, player1.id);
    player1.add_entity(entity1.id, hlt::Location{2, 2});
    map.at(hlt::Location{2, 2}).entity = entity1.id;
    std::vector<std::unique_ptr<hlt::Command>> commands1;
    commands1.push_back(std::make_unique<hlt::MoveCommand>(entity1.id, hlt::Direction::South));
    actions[player1.id] = std::move(commands1);

    CountingExecutor executor;
    hlt::rules::RuleContext context(state, actions, result, config, sink, &executor);
    hlt::rules::phases::ValidationPhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 2);
    REQUIRE(result.validated_commands.has_value());
}

TEST_CASE("TurnStatsCollector uses provided executor", "[modernization][observers]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
    hlt::GameState state(store, map, statistics);
    const auto config = hlt::GameConfig::from_constants();

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &entity = store.new_entity(0, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;

    CountingExecutor executor;
    hlt::observers::TurnStatsCollector collector;
    collector.collect(state, config, executor);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 1);
}

TEST_CASE("MiningPhase uses provided executor for planning", "[modernization][rules]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &entity = store.new_entity(0, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;
    map.at(hlt::Location{1, 1}).energy = 100;

    CountingExecutor executor;
    hlt::rules::RuleContext context(state, actions, result, config, sink, &executor);
    hlt::rules::phases::MiningPhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 1);
}

TEST_CASE("LocalStepResult merges into StepResult", "[modernization][engine]") {
    hlt::StepResult global{};
    hlt::LocalStepResult local{};
    local.non_fatal_errors.push_back("warn");
    local.eliminated_players.push_back(hlt::Player::id_type{1});
    local.changed_entities.insert(hlt::Entity::id_type{2});
    local.changed_cells.insert(hlt::Location{1, 2});

    hlt::merge_into(global, std::move(local));

    REQUIRE(global.non_fatal_errors.size() == 1);
    REQUIRE(global.eliminated_players.size() == 1);
    REQUIRE(global.changed_entities.count(hlt::Entity::id_type{2}) == 1);
    REQUIRE(global.changed_cells.count(hlt::Location{1, 2}) == 1);
}

TEST_CASE("EventBuffer flushes buffered events", "[modernization][events]") {
    hlt::events::EventBuffer buffer;
    hlt::events::RecordingEventSink sink;
    buffer.emit(hlt::events::PlayerEliminatedEvent{hlt::Player::id_type{0}, "test"});

    REQUIRE_FALSE(buffer.empty());
    buffer.flush_into(sink);

    REQUIRE(buffer.empty());
    REQUIRE(sink.events().size() == 1);
}

TEST_CASE("MiningPhase emits mined event through buffered apply path", "[modernization][events]") {
    hlt::Store store{};
    hlt::Map map(4, 4);
    hlt::GameStatistics statistics{};
    statistics.player_statistics.emplace_back(hlt::Player::id_type{0}, 0);
    hlt::GameState state(store, map, statistics);
    hlt::ActionBatch actions{};
    hlt::StepResult result{};
    hlt::events::RecordingEventSink sink{};
    const auto config = hlt::GameConfig::from_constants();

    auto [player_it, inserted] = store.players_ref().emplace(hlt::Player::id_type{0}, hlt::Player{hlt::Player::id_type{0}, hlt::Location{0, 0}, "bot"});
    REQUIRE(inserted);
    auto &player = player_it->second;
    auto &entity = store.new_entity(0, player.id);
    player.add_entity(entity.id, hlt::Location{1, 1});
    map.at(hlt::Location{1, 1}).entity = entity.id;
    map.at(hlt::Location{1, 1}).energy = 100;

    hlt::rules::RuleContext context(state, actions, result, config, sink);
    hlt::rules::phases::MiningPhase phase;
    phase.execute(context);

    REQUIRE_FALSE(sink.events().empty());
    REQUIRE(std::holds_alternative<hlt::events::MinedEvent>(sink.events().front()));
}
