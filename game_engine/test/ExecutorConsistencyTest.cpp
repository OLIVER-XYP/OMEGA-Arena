#include "catch.hpp"

#include <memory>

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
#include "core/engine/TaskExecutor.hpp"
#include "core/engine/TurnEngine.hpp"
#include "events/EventSink.hpp"
#include "rules/RuleContext.hpp"
#include "rules/phases/CapturePhase.hpp"
#include "rules/phases/CombatPhase.hpp"
#include "rules/phases/ConstructionPhase.hpp"
#include "rules/phases/InspirationPhase.hpp"
#include "rules/phases/MiningPhase.hpp"
#include "rules/phases/MovementPhase.hpp"
#include "rules/phases/SpawnPhase.hpp"
#include "rules/phases/ValidationPhase.hpp"

namespace {

using namespace hlt;

Player &add_player(Store &store, Player::id_type id, Location factory, energy_type energy = 5000) {
    auto [it, inserted] = store.players_ref().emplace(id, Player{id, factory, "bot"});
    REQUIRE(inserted);
    it->second.energy = energy;
    it->second.factory_halite = Constants::get().INITIAL_FACTORY_HALITE;
    return it->second;
}

Entity &add_ship(Store &store, Map &map, Player &player, Location location, energy_type energy = 0) {
    auto &entity = store.new_entity(energy, player.id);
    player.add_entity(entity.id, location);
    map.at(location).entity = entity.id;
    return entity;
}

class CountingExecutor final : public TaskExecutor {
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

    ExecutorRunStats run_stats() const override {
        return ExecutorRunStats{calls, last_end >= last_begin ? (last_end - last_begin) : 0};
    }

    void reset_run_stats() override {
        calls = 0;
        last_begin = 0;
        last_end = 0;
    }
};

} // namespace

TEST_CASE("ValidationPhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    const auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(4, 4);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};

    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0});
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{3, 3});
    auto &e0_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 1});
    auto &e1_inline = add_ship(store_inline, map_inline, p1_inline, Location{2, 2});
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(e0_inline.id, Direction::North));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(e1_inline.id, Direction::South));

    Store store_pool{};
    Map map_pool(4, 4);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};

    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0});
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{3, 3});
    auto &e0_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 1});
    auto &e1_pool = add_ship(store_pool, map_pool, p1_pool, Location{2, 2});
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(e0_pool.id, Direction::North));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(e1_pool.id, Direction::South));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::ValidationPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.eliminated_players == result_pool.eliminated_players);
    REQUIRE(result_inline.validated_commands.has_value());
    REQUIRE(result_pool.validated_commands.has_value());
    REQUIRE(result_inline.validated_commands->moves[p0_inline.id].size() == result_pool.validated_commands->moves[p0_pool.id].size());
    REQUIRE(result_inline.validated_commands->moves[p1_inline.id].size() == result_pool.validated_commands->moves[p1_pool.id].size());
}

TEST_CASE("MiningPhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    const auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(4, 4);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0});
    auto &e_inline = add_ship(store_inline, map_inline, p_inline, Location{1, 1});
    map_inline.at(Location{1, 1}).energy = 100;

    Store store_pool{};
    Map map_pool(4, 4);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0});
    auto &e_pool = add_ship(store_pool, map_pool, p_pool, Location{1, 1});
    map_pool.at(Location{1, 1}).energy = 100;

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::MiningPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.get_entity(e_inline.id).energy == store_pool.get_entity(e_pool.id).energy);
    REQUIRE(map_inline.at(Location{1, 1}).energy == map_pool.at(Location{1, 1}).energy);
}

TEST_CASE("InspirationPhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(4, 4);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0});
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{3, 3});
    auto &center_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 1});
    add_ship(store_inline, map_inline, p1_inline, Location{1, 2});
    add_ship(store_inline, map_inline, p1_inline, Location{2, 1});
    add_ship(store_inline, map_inline, p1_inline, Location{2, 2});

    Store store_pool{};
    Map map_pool(4, 4);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0});
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{3, 3});
    auto &center_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 1});
    add_ship(store_pool, map_pool, p1_pool, Location{1, 2});
    add_ship(store_pool, map_pool, p1_pool, Location{2, 1});
    add_ship(store_pool, map_pool, p1_pool, Location{2, 2});

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::InspirationPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(store_inline.get_entity(center_inline.id).is_inspired == store_pool.get_entity(center_pool.id).is_inspired);
}

TEST_CASE("CapturePhase uses provided executor", "[modernization][rules]") {
    auto config = GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    Store store{};
    Map map(5, 5);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &p1 = add_player(store, Player::id_type{1}, Location{4, 4});
    add_ship(store, map, p0, Location{2, 2});
    add_ship(store, map, p1, Location{2, 1});
    add_ship(store, map, p1, Location{1, 2});

    CountingExecutor executor;
    rules::RuleContext context(state, actions, result, config, sink, &executor);
    rules::phases::CapturePhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 3);
}

TEST_CASE("CapturePhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    Store store_inline{};
    Map map_inline(5, 5);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0});
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{4, 4});
    add_ship(store_inline, map_inline, p0_inline, Location{2, 2});
    add_ship(store_inline, map_inline, p1_inline, Location{2, 1});
    add_ship(store_inline, map_inline, p1_inline, Location{1, 2});

    Store store_pool{};
    Map map_pool(5, 5);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0});
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{4, 4});
    add_ship(store_pool, map_pool, p0_pool, Location{2, 2});
    add_ship(store_pool, map_pool, p1_pool, Location{2, 1});
    add_ship(store_pool, map_pool, p1_pool, Location{1, 2});

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::CapturePhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(map_inline.at(Location{2, 2}).entity != Entity::None);
    REQUIRE(map_pool.at(Location{2, 2}).entity != Entity::None);
    REQUIRE(store_inline.get_entity(map_inline.at(Location{2, 2}).entity).owner == store_pool.get_entity(map_pool.at(Location{2, 2}).entity).owner);
}

TEST_CASE("CombatPhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(4, 4);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0});
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{3, 3});
    auto &attacker_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 1}, 0);
    auto &target_inline = add_ship(store_inline, map_inline, p1_inline, Location{1, 2}, 1000);
    AttackCommand attack_inline{attacker_inline.id, target_inline.id};
    result_inline.validated_commands = CommandBatch{};
    result_inline.validated_commands->attacks[p0_inline.id].push_back(std::cref(attack_inline));

    Store store_pool{};
    Map map_pool(4, 4);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0});
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{3, 3});
    auto &attacker_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 1}, 0);
    auto &target_pool = add_ship(store_pool, map_pool, p1_pool, Location{1, 2}, 1000);
    AttackCommand attack_pool{attacker_pool.id, target_pool.id};
    result_pool.validated_commands = CommandBatch{};
    result_pool.validated_commands->attacks[p0_pool.id].push_back(std::cref(attack_pool));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::CombatPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(store_inline.get_entity(target_inline.id).hp == store_pool.get_entity(target_pool.id).hp);
    REQUIRE(store_inline.get_entity(target_inline.id).energy == store_pool.get_entity(target_pool.id).energy);
    REQUIRE(store_inline.get_entity(attacker_inline.id).energy == store_pool.get_entity(attacker_pool.id).energy);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
}

TEST_CASE("TurnEngine step stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(5, 5);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_inline.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 1}, 0);
    auto &e1_inline = add_ship(store_inline, map_inline, p1_inline, Location{1, 2}, 1000);
    map_inline.at(Location{1, 1}).energy = 100;
    map_inline.at(Location{1, 2}).energy = 80;
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(e0_inline.id, Direction::Still));
    actions_inline[p0_inline.id].push_back(std::make_unique<AttackCommand>(e0_inline.id, e1_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<SpawnCommand>());
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(e1_inline.id, Direction::Still));

    Store store_pool{};
    Map map_pool(5, 5);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_pool.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 1}, 0);
    auto &e1_pool = add_ship(store_pool, map_pool, p1_pool, Location{1, 2}, 1000);
    map_pool.at(Location{1, 1}).energy = 100;
    map_pool.at(Location{1, 2}).energy = 80;
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(e0_pool.id, Direction::Still));
    actions_pool[p0_pool.id].push_back(std::make_unique<AttackCommand>(e0_pool.id, e1_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<SpawnCommand>());
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(e1_pool.id, Direction::Still));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    TurnEngine engine(config);

    auto result_inline = engine.step(state_inline, actions_inline, sink_inline, inline_executor);
    auto result_pool = engine.step(state_pool, actions_pool, sink_pool, threadpool_executor);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.eliminated_players == result_pool.eliminated_players);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.players_ref().at(p0_inline.id).energy == store_pool.players_ref().at(p0_pool.id).energy);
    REQUIRE(store_inline.players_ref().at(p1_inline.id).energy == store_pool.players_ref().at(p1_pool.id).energy);
}

TEST_CASE("ConstructionPhase uses provided executor", "[modernization][rules]") {
    auto config = GameConfig::from_constants();

    Store store{};
    Map map(5, 5);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1 = add_player(store, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0 = add_ship(store, map, p0, Location{1, 1}, 500);
    auto &e1 = add_ship(store, map, p1, Location{3, 3}, 400);
    (void)e0;
    (void)e1;
    result.validated_commands = CommandBatch{};
    static ConstructCommand c0{p0.entities.begin()->first};
    static ConstructCommand c1{p1.entities.begin()->first};
    result.validated_commands->constructs[p0.id].push_back(std::cref(c0));
    result.validated_commands->constructs[p1.id].push_back(std::cref(c1));

    CountingExecutor executor;
    rules::RuleContext context(state, actions, result, config, sink, &executor);
    rules::phases::ConstructionPhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 2);
}

TEST_CASE("SpawnPhase uses provided executor", "[modernization][rules]") {
    auto config = GameConfig::from_constants();

    Store store{};
    Map map(5, 5);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1 = add_player(store, Player::id_type{1}, Location{4, 4}, 5000);
    result.validated_commands = CommandBatch{};
    static SpawnCommand s0{};
    static SpawnCommand s1{};
    result.validated_commands->spawns[p0.id].push_back(std::cref(s0));
    result.validated_commands->spawns[p1.id].push_back(std::cref(s1));

    CountingExecutor executor;
    rules::RuleContext context(state, actions, result, config, sink, &executor);
    rules::phases::SpawnPhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 2);
}

TEST_CASE("MovementPhase uses provided executor", "[modernization][rules]") {
    auto config = GameConfig::from_constants();

    Store store{};
    Map map(5, 5);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1 = add_player(store, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0 = add_ship(store, map, p0, Location{1, 1}, 1000);
    auto &e1 = add_ship(store, map, p1, Location{3, 3}, 1000);
    map.at(Location{1, 1}).energy = 100;
    map.at(Location{3, 3}).energy = 80;
    result.validated_commands = CommandBatch{};
    static MoveCommand m0{e0.id, Direction::North};
    static MoveCommand m1{e1.id, Direction::South};
    result.validated_commands->moves[p0.id].push_back(std::cref(m0));
    result.validated_commands->moves[p1.id].push_back(std::cref(m1));

    CountingExecutor executor;
    rules::RuleContext context(state, actions, result, config, sink, &executor);
    rules::phases::MovementPhase phase;
    phase.execute(context);

    REQUIRE(executor.calls == 1);
    REQUIRE(executor.last_begin == 0);
    REQUIRE(executor.last_end == 2);
}

TEST_CASE("MovementPhase stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(5, 5);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 1}, 1000);
    auto &e1_inline = add_ship(store_inline, map_inline, p1_inline, Location{3, 3}, 1000);
    map_inline.at(Location{1, 1}).energy = 100;
    map_inline.at(Location{3, 3}).energy = 80;
    result_inline.validated_commands = CommandBatch{};
    static MoveCommand mi0{e0_inline.id, Direction::North};
    static MoveCommand mi1{e1_inline.id, Direction::South};
    result_inline.validated_commands->moves[p0_inline.id].push_back(std::cref(mi0));
    result_inline.validated_commands->moves[p1_inline.id].push_back(std::cref(mi1));

    Store store_pool{};
    Map map_pool(5, 5);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{4, 4}, 5000);
    auto &e0_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 1}, 1000);
    auto &e1_pool = add_ship(store_pool, map_pool, p1_pool, Location{3, 3}, 1000);
    map_pool.at(Location{1, 1}).energy = 100;
    map_pool.at(Location{3, 3}).energy = 80;
    result_pool.validated_commands = CommandBatch{};
    static MoveCommand mp0{e0_pool.id, Direction::North};
    static MoveCommand mp1{e1_pool.id, Direction::South};
    result_pool.validated_commands->moves[p0_pool.id].push_back(std::cref(mp0));
    result_pool.validated_commands->moves[p1_pool.id].push_back(std::cref(mp1));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::MovementPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);

    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.get_entity(e0_inline.id).energy == store_pool.get_entity(e0_pool.id).energy);
    REQUIRE(store_inline.get_entity(e1_inline.id).energy == store_pool.get_entity(e1_pool.id).energy);
}



TEST_CASE("CombatPhase stays consistent with structure attack between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(5, 5);
    GameStatistics stats_inline{};
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    StepResult result_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{4, 4}, 5000);
    auto &attacker_inline = add_ship(store_inline, map_inline, p0_inline, Location{4, 3}, 1000);
    (void)attacker_inline;
    map_inline.at(p1_inline.factory).owner = p1_inline.id;
    AttackCommand atk_inline{p0_inline.entities.begin()->first, p1_inline.id, p1_inline.factory};
    result_inline.validated_commands = CommandBatch{};
    result_inline.validated_commands->attacks[p0_inline.id].push_back(std::cref(atk_inline));

    Store store_pool{};
    Map map_pool(5, 5);
    GameStatistics stats_pool{};
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    StepResult result_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{4, 4}, 5000);
    auto &attacker_pool = add_ship(store_pool, map_pool, p0_pool, Location{4, 3}, 1000);
    (void)attacker_pool;
    map_pool.at(p1_pool.factory).owner = p1_pool.id;
    AttackCommand atk_pool{p0_pool.entities.begin()->first, p1_pool.id, p1_pool.factory};
    result_pool.validated_commands = CommandBatch{};
    result_pool.validated_commands->attacks[p0_pool.id].push_back(std::cref(atk_pool));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    rules::phases::CombatPhase phase;
    rules::RuleContext inline_context(state_inline, actions_inline, result_inline, config, sink_inline, &inline_executor);
    rules::RuleContext pool_context(state_pool, actions_pool, result_pool, config, sink_pool, &threadpool_executor);
    phase.execute(inline_context);
    phase.execute(pool_context);

    REQUIRE(store_inline.players_ref().at(p1_inline.id).factory_halite == store_pool.players_ref().at(p1_pool.id).factory_halite);
    REQUIRE(store_inline.get_entity(p0_inline.entities.begin()->first).energy == store_pool.get_entity(p0_pool.entities.begin()->first).energy);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
}

TEST_CASE("TurnEngine mixed step stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    Store store_inline{};
    Map map_inline(6, 6);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_inline.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 6000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{5, 5}, 6000);
    auto &a0_inline = add_ship(store_inline, map_inline, p0_inline, Location{2, 2}, 1000);
    auto &a1_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 0}, 500);
    auto &b0_inline = add_ship(store_inline, map_inline, p1_inline, Location{2, 3}, 1000);
    auto &b1_inline = add_ship(store_inline, map_inline, p1_inline, Location{3, 2}, 500);
    map_inline.at(Location{2, 2}).energy = 120;
    map_inline.at(Location{1, 0}).energy = 60;
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(a0_inline.id, Direction::Still));
    actions_inline[p0_inline.id].push_back(std::make_unique<AttackCommand>(a0_inline.id, b0_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<ConstructCommand>(a1_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<SpawnCommand>());
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(b0_inline.id, Direction::Still));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(b1_inline.id, Direction::West));

    Store store_pool{};
    Map map_pool(6, 6);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_pool.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 6000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{5, 5}, 6000);
    auto &a0_pool = add_ship(store_pool, map_pool, p0_pool, Location{2, 2}, 1000);
    auto &a1_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 0}, 500);
    auto &b0_pool = add_ship(store_pool, map_pool, p1_pool, Location{2, 3}, 1000);
    auto &b1_pool = add_ship(store_pool, map_pool, p1_pool, Location{3, 2}, 500);
    map_pool.at(Location{2, 2}).energy = 120;
    map_pool.at(Location{1, 0}).energy = 60;
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(a0_pool.id, Direction::Still));
    actions_pool[p0_pool.id].push_back(std::make_unique<AttackCommand>(a0_pool.id, b0_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<ConstructCommand>(a1_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<SpawnCommand>());
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(b0_pool.id, Direction::Still));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(b1_pool.id, Direction::West));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    TurnEngine engine(config);
    auto result_inline = engine.step(state_inline, actions_inline, sink_inline, inline_executor);
    auto result_pool = engine.step(state_pool, actions_pool, sink_pool, threadpool_executor);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.players_ref().at(p0_inline.id).energy == store_pool.players_ref().at(p0_pool.id).energy);
    REQUIRE(store_inline.players_ref().at(p1_inline.id).energy == store_pool.players_ref().at(p1_pool.id).energy);
}

TEST_CASE("TurnEngine high-mix step stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    Store store_inline{};
    Map map_inline(7, 7);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_inline.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 7000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{6, 6}, 7000);
    auto &m0_inline = add_ship(store_inline, map_inline, p0_inline, Location{2, 2}, 1000);
    auto &c0_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 0}, 600);
    auto &f0_inline = add_ship(store_inline, map_inline, p0_inline, Location{4, 4}, 300);
    auto &e0_inline = add_ship(store_inline, map_inline, p1_inline, Location{2, 3}, 1000);
    auto &e1_inline = add_ship(store_inline, map_inline, p1_inline, Location{3, 2}, 600);
    auto &e2_inline = add_ship(store_inline, map_inline, p1_inline, Location{5, 4}, 600);
    map_inline.at(Location{2, 2}).energy = 120;
    map_inline.at(Location{1, 0}).energy = 100;
    map_inline.at(Location{4, 4}).energy = 90;
    map_inline.at(Location{5, 4}).owner = p1_inline.id;
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(m0_inline.id, Direction::Still));
    actions_inline[p0_inline.id].push_back(std::make_unique<AttackCommand>(m0_inline.id, e0_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<ConstructCommand>(c0_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(f0_inline.id, Direction::East));
    actions_inline[p0_inline.id].push_back(std::make_unique<SpawnCommand>());
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(e0_inline.id, Direction::Still));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(e1_inline.id, Direction::West));
    actions_inline[p1_inline.id].push_back(std::make_unique<ConstructCommand>(e2_inline.id));

    Store store_pool{};
    Map map_pool(7, 7);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_pool.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 7000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{6, 6}, 7000);
    auto &m0_pool = add_ship(store_pool, map_pool, p0_pool, Location{2, 2}, 1000);
    auto &c0_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 0}, 600);
    auto &f0_pool = add_ship(store_pool, map_pool, p0_pool, Location{4, 4}, 300);
    auto &e0_pool = add_ship(store_pool, map_pool, p1_pool, Location{2, 3}, 1000);
    auto &e1_pool = add_ship(store_pool, map_pool, p1_pool, Location{3, 2}, 600);
    auto &e2_pool = add_ship(store_pool, map_pool, p1_pool, Location{5, 4}, 600);
    map_pool.at(Location{2, 2}).energy = 120;
    map_pool.at(Location{1, 0}).energy = 100;
    map_pool.at(Location{4, 4}).energy = 90;
    map_pool.at(Location{5, 4}).owner = p1_pool.id;
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(m0_pool.id, Direction::Still));
    actions_pool[p0_pool.id].push_back(std::make_unique<AttackCommand>(m0_pool.id, e0_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<ConstructCommand>(c0_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(f0_pool.id, Direction::East));
    actions_pool[p0_pool.id].push_back(std::make_unique<SpawnCommand>());
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(e0_pool.id, Direction::Still));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(e1_pool.id, Direction::West));
    actions_pool[p1_pool.id].push_back(std::make_unique<ConstructCommand>(e2_pool.id));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    TurnEngine engine(config);
    auto result_inline = engine.step(state_inline, actions_inline, sink_inline, inline_executor);
    auto result_pool = engine.step(state_pool, actions_pool, sink_pool, threadpool_executor);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.players_ref().at(p0_inline.id).energy == store_pool.players_ref().at(p0_pool.id).energy);
    REQUIRE(store_inline.players_ref().at(p1_inline.id).energy == store_pool.players_ref().at(p1_pool.id).energy);
}

TEST_CASE("TurnEngine move-combat interaction stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();

    Store store_inline{};
    Map map_inline(7, 7);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_inline.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 7000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{6, 6}, 7000);
    auto &mover_inline = add_ship(store_inline, map_inline, p0_inline, Location{2, 2}, 1000);
    auto &support_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 2}, 800);
    auto &defender_inline = add_ship(store_inline, map_inline, p1_inline, Location{4, 2}, 1000);
    auto &ally_inline = add_ship(store_inline, map_inline, p1_inline, Location{5, 2}, 700);
    map_inline.at(Location{2, 2}).energy = 100;
    map_inline.at(Location{1, 2}).energy = 60;
    map_inline.at(Location{4, 2}).energy = 80;
    map_inline.at(Location{5, 2}).energy = 50;
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(mover_inline.id, Direction::East));
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(support_inline.id, Direction::East));
    actions_inline[p0_inline.id].push_back(std::make_unique<AttackCommand>(mover_inline.id, defender_inline.id));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(defender_inline.id, Direction::Still));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(ally_inline.id, Direction::West));

    Store store_pool{};
    Map map_pool(7, 7);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_pool.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 7000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{6, 6}, 7000);
    auto &mover_pool = add_ship(store_pool, map_pool, p0_pool, Location{2, 2}, 1000);
    auto &support_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 2}, 800);
    auto &defender_pool = add_ship(store_pool, map_pool, p1_pool, Location{4, 2}, 1000);
    auto &ally_pool = add_ship(store_pool, map_pool, p1_pool, Location{5, 2}, 700);
    map_pool.at(Location{2, 2}).energy = 100;
    map_pool.at(Location{1, 2}).energy = 60;
    map_pool.at(Location{4, 2}).energy = 80;
    map_pool.at(Location{5, 2}).energy = 50;
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(mover_pool.id, Direction::East));
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(support_pool.id, Direction::East));
    actions_pool[p0_pool.id].push_back(std::make_unique<AttackCommand>(mover_pool.id, defender_pool.id));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(defender_pool.id, Direction::Still));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(ally_pool.id, Direction::West));

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    TurnEngine engine(config);
    auto result_inline = engine.step(state_inline, actions_inline, sink_inline, inline_executor);
    auto result_pool = engine.step(state_pool, actions_pool, sink_pool, threadpool_executor);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.players_ref().at(p0_inline.id).energy == store_pool.players_ref().at(p0_pool.id).energy);
    REQUIRE(store_inline.players_ref().at(p1_inline.id).energy == store_pool.players_ref().at(p1_pool.id).energy);
}

TEST_CASE("TurnEngine extreme mixed step stays consistent between inline and threadpool", "[modernization][consistency]") {
    auto config = GameConfig::from_constants();
    config.ruleset.capture.enabled = true;
    config.ruleset.capture.radius = 1;
    config.ruleset.capture.ships_above_for_capture = 1;

    Store store_inline{};
    Map map_inline(7, 7);
    GameStatistics stats_inline{};
    stats_inline.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_inline.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_inline(store_inline, map_inline, stats_inline);
    ActionBatch actions_inline{};
    events::RecordingEventSink sink_inline{};
    auto &p0_inline = add_player(store_inline, Player::id_type{0}, Location{0, 0}, 8000);
    auto &p1_inline = add_player(store_inline, Player::id_type{1}, Location{6, 6}, 8000);

    auto &collider_a_inline = add_ship(store_inline, map_inline, p0_inline, Location{2, 2}, 1000);
    auto &attacker_inline = add_ship(store_inline, map_inline, p0_inline, Location{2, 4}, 1000);
    auto &builder_inline = add_ship(store_inline, map_inline, p0_inline, Location{1, 0}, 700);
    auto &capture_pressure_inline = add_ship(store_inline, map_inline, p0_inline, Location{4, 4}, 400);

    auto &collider_b_inline = add_ship(store_inline, map_inline, p1_inline, Location{4, 2}, 1000);
    auto &target_inline = add_ship(store_inline, map_inline, p1_inline, Location{2, 5}, 900);
    auto &builder_b_inline = add_ship(store_inline, map_inline, p1_inline, Location{5, 4}, 700);
    auto &capture_target_inline = add_ship(store_inline, map_inline, p1_inline, Location{4, 5}, 500);

    map_inline.at(Location{2, 2}).energy = 100;
    map_inline.at(Location{4, 2}).energy = 90;
    map_inline.at(Location{2, 4}).energy = 70;
    map_inline.at(Location{2, 5}).energy = 60;
    map_inline.at(Location{1, 0}).energy = 110;
    map_inline.at(Location{5, 4}).energy = 120;
    map_inline.at(Location{4, 4}).energy = 50;
    map_inline.at(Location{4, 5}).energy = 50;
    map_inline.at(Location{5, 4}).owner = p1_inline.id;

    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(collider_a_inline.id, Direction::East));
    actions_inline[p0_inline.id].push_back(std::make_unique<MoveCommand>(attacker_inline.id, Direction::Still));
    actions_inline[p0_inline.id].push_back(std::make_unique<AttackCommand>(attacker_inline.id, target_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<ConstructCommand>(builder_inline.id));
    actions_inline[p0_inline.id].push_back(std::make_unique<SpawnCommand>());

    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(collider_b_inline.id, Direction::West));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(target_inline.id, Direction::Still));
    actions_inline[p1_inline.id].push_back(std::make_unique<ConstructCommand>(builder_b_inline.id));
    actions_inline[p1_inline.id].push_back(std::make_unique<MoveCommand>(capture_target_inline.id, Direction::Still));

    Store store_pool{};
    Map map_pool(7, 7);
    GameStatistics stats_pool{};
    stats_pool.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats_pool.player_statistics.emplace_back(Player::id_type{1}, 0);
    GameState state_pool(store_pool, map_pool, stats_pool);
    ActionBatch actions_pool{};
    events::RecordingEventSink sink_pool{};
    auto &p0_pool = add_player(store_pool, Player::id_type{0}, Location{0, 0}, 8000);
    auto &p1_pool = add_player(store_pool, Player::id_type{1}, Location{6, 6}, 8000);

    auto &collider_a_pool = add_ship(store_pool, map_pool, p0_pool, Location{2, 2}, 1000);
    auto &attacker_pool = add_ship(store_pool, map_pool, p0_pool, Location{2, 4}, 1000);
    auto &builder_pool = add_ship(store_pool, map_pool, p0_pool, Location{1, 0}, 700);
    auto &capture_pressure_pool = add_ship(store_pool, map_pool, p0_pool, Location{4, 4}, 400);

    auto &collider_b_pool = add_ship(store_pool, map_pool, p1_pool, Location{4, 2}, 1000);
    auto &target_pool = add_ship(store_pool, map_pool, p1_pool, Location{2, 5}, 900);
    auto &builder_b_pool = add_ship(store_pool, map_pool, p1_pool, Location{5, 4}, 700);
    auto &capture_target_pool = add_ship(store_pool, map_pool, p1_pool, Location{4, 5}, 500);

    map_pool.at(Location{2, 2}).energy = 100;
    map_pool.at(Location{4, 2}).energy = 90;
    map_pool.at(Location{2, 4}).energy = 70;
    map_pool.at(Location{2, 5}).energy = 60;
    map_pool.at(Location{1, 0}).energy = 110;
    map_pool.at(Location{5, 4}).energy = 120;
    map_pool.at(Location{4, 4}).energy = 50;
    map_pool.at(Location{4, 5}).energy = 50;
    map_pool.at(Location{5, 4}).owner = p1_pool.id;

    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(collider_a_pool.id, Direction::East));
    actions_pool[p0_pool.id].push_back(std::make_unique<MoveCommand>(attacker_pool.id, Direction::Still));
    actions_pool[p0_pool.id].push_back(std::make_unique<AttackCommand>(attacker_pool.id, target_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<ConstructCommand>(builder_pool.id));
    actions_pool[p0_pool.id].push_back(std::make_unique<SpawnCommand>());

    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(collider_b_pool.id, Direction::West));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(target_pool.id, Direction::Still));
    actions_pool[p1_pool.id].push_back(std::make_unique<ConstructCommand>(builder_b_pool.id));
    actions_pool[p1_pool.id].push_back(std::make_unique<MoveCommand>(capture_target_pool.id, Direction::Still));

    (void)capture_pressure_inline;
    (void)capture_target_inline;
    (void)capture_pressure_pool;
    (void)capture_target_pool;

    InlineExecutor inline_executor;
    ThreadPoolExecutor threadpool_executor{2};
    TurnEngine engine(config);
    auto result_inline = engine.step(state_inline, actions_inline, sink_inline, inline_executor);
    auto result_pool = engine.step(state_pool, actions_pool, sink_pool, threadpool_executor);

    REQUIRE(result_inline.non_fatal_errors == result_pool.non_fatal_errors);
    REQUIRE(result_inline.eliminated_players == result_pool.eliminated_players);
    REQUIRE(result_inline.changed_cells == result_pool.changed_cells);
    REQUIRE(result_inline.changed_entities == result_pool.changed_entities);
    REQUIRE(sink_inline.events().size() == sink_pool.events().size());
    REQUIRE(store_inline.players_ref().at(p0_inline.id).energy == store_pool.players_ref().at(p0_pool.id).energy);
    REQUIRE(store_inline.players_ref().at(p1_inline.id).energy == store_pool.players_ref().at(p1_pool.id).energy);
}

TEST_CASE("CombatPhase processes all pending deaths even with duplicate entries", "[modernization][rules]") {
    auto config = GameConfig::from_constants();
    config.ruleset.combat.attack_hp_damage = 1;

    Store store{};
    Map map(7, 7);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    auto &p1 = add_player(store, Player::id_type{1}, Location{6, 6}, 5000);

    auto &a0 = add_ship(store, map, p0, Location{2, 2}, 1000);
    auto &a1 = add_ship(store, map, p0, Location{2, 4}, 1000);
    auto &a2 = add_ship(store, map, p0, Location{4, 2}, 1000);

    auto &t0 = add_ship(store, map, p1, Location{2, 3}, 50);
    auto &t1 = add_ship(store, map, p1, Location{5, 2}, 50);
    t0.hp = 1;
    t1.hp = 1;

    result.validated_commands = CommandBatch{};
    static AttackCommand c0{a0.id, t0.id};
    static AttackCommand c1{a1.id, t0.id};
    static AttackCommand c2{a2.id, t1.id};
    result.validated_commands->attacks[p0.id].push_back(std::cref(c0));
    result.validated_commands->attacks[p0.id].push_back(std::cref(c1));
    result.validated_commands->attacks[p0.id].push_back(std::cref(c2));

    InlineExecutor inline_executor;
    rules::RuleContext context(state, actions, result, config, sink, &inline_executor);
    rules::phases::CombatPhase phase;
    phase.execute(context);

    REQUIRE(store.players_ref().at(p1.id).has_entity(t0.id) == false);
    REQUIRE(store.players_ref().at(p1.id).has_entity(t1.id) == false);
}
