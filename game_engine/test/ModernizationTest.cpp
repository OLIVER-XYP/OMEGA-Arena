#include "catch.hpp"

#include <atomic>
#include <memory>
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
#include "core/engine/TurnEngine.hpp"
#include "core/engine/LocalStepResult.hpp"
#include "core/engine/TaskExecutor.hpp"
#include "observers/TurnStatsCollector.hpp"
#include "protocol/TextProtocolCodec.hpp"
#include "rules/RuleContext.hpp"
#include "rules/phases/MiningPhase.hpp"
#include "rules/phases/ValidationPhase.hpp"
#include "events/EventBuffer.hpp"
#include "events/EventSink.hpp"

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
