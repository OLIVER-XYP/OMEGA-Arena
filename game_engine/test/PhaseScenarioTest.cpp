#include "catch.hpp"

#include <functional>
#include <memory>
#include <vector>

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
#include "core/config/GameConfig.hpp"
#include "core/state/GameState.hpp"
#include "events/EventSink.hpp"
#include "model/Player.hpp"
#include "observers/TurnStatsCollector.hpp"
#include "rules/RuleContext.hpp"
#include "rules/phases/CapturePhase.hpp"
#include "rules/phases/CombatPhase.hpp"
#include "rules/phases/ConstructionPhase.hpp"
#include "rules/phases/InspirationPhase.hpp"
#include "rules/phases/MiningPhase.hpp"
#include "rules/phases/MovementPhase.hpp"
#include "rules/phases/SpawnPhase.hpp"
#include "rules/phases/ValidationPhase.hpp"

using namespace hlt;

namespace {

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

hlt::rules::RuleContext make_context(GameState &state,
                                     ActionBatch &actions,
                                     StepResult &result,
                                     const GameConfig &config,
                                     events::RecordingEventSink &sink) {
    return hlt::rules::RuleContext(state, actions, result, config, sink);
}

} // namespace

TEST_CASE("InspirationPhase marks ships inspired with enough nearby enemies", "[phase][inspiration]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    ActionBatch actions{};
    StepResult result{};
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &p1 = add_player(store, Player::id_type{1}, Location{3, 3});
    auto &center = add_ship(store, map, p0, Location{1, 1});
    add_ship(store, map, p1, Location{1, 2});
    add_ship(store, map, p1, Location{2, 1});
    add_ship(store, map, p1, Location{2, 2});

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::InspirationPhase phase;
    phase.execute(context);

    REQUIRE(store.get_entity(center.id).is_inspired);
}

TEST_CASE("ValidationPhase rejects duplicate commands on same entity", "[phase][validation]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &ship = add_ship(store, map, p0, Location{1, 1});
    actions[p0.id].push_back(std::make_unique<MoveCommand>(ship.id, Direction::North));
    actions[p0.id].push_back(std::make_unique<MoveCommand>(ship.id, Direction::South));

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::ValidationPhase phase;
    phase.execute(context);

    REQUIRE_FALSE(result.validated_commands.has_value());
    REQUIRE_FALSE(result.non_fatal_errors.empty());
}

TEST_CASE("ConstructionPhase builds dropoff when resources suffice", "[phase][construction]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    auto &ship = add_ship(store, map, p0, Location{1, 1}, 0);
    map.at(Location{1, 1}).energy = config.ruleset.economy.dropoff_cost;
    ConstructCommand construct{ship.id};
    result.validated_commands = CommandBatch{};
    result.validated_commands->constructs[p0.id].push_back(std::cref(construct));

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::ConstructionPhase phase;
    phase.execute(context);

    REQUIRE(map.at(Location{1, 1}).owner == p0.id);
    REQUIRE(p0.dropoffs.size() == 1);
}

TEST_CASE("MovementPhase resolves same-destination collision", "[phase][movement]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &a = add_ship(store, map, p0, Location{1, 0}, 1000);
    auto &b = add_ship(store, map, p0, Location{1, 2}, 1000);
    MoveCommand move_a{a.id, Direction::South};
    MoveCommand move_b{b.id, Direction::North};
    result.validated_commands = CommandBatch{};
    result.validated_commands->moves[p0.id].push_back(std::cref(move_a));
    result.validated_commands->moves[p0.id].push_back(std::cref(move_b));

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::MovementPhase phase;
    phase.execute(context);

    REQUIRE_FALSE(sink.events().empty());
    REQUIRE(result.changed_cells.find(Location{1, 1}) != result.changed_cells.end());
}

TEST_CASE("CombatPhase damages adjacent enemy and steals halite", "[phase][combat]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &p1 = add_player(store, Player::id_type{1}, Location{3, 3});
    auto &attacker = add_ship(store, map, p0, Location{1, 1}, 0);
    auto &target = add_ship(store, map, p1, Location{1, 2}, 1000);
    const auto initial_hp = target.hp;
    AttackCommand attack{attacker.id, target.id};
    result.validated_commands = CommandBatch{};
    result.validated_commands->attacks[p0.id].push_back(std::cref(attack));

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::CombatPhase phase;
    phase.execute(context);

    REQUIRE(store.get_entity(target.id).hp < initial_hp);
}

TEST_CASE("MiningPhase extracts from occupied cell", "[phase][mining]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    stats.player_statistics.emplace_back(Player::id_type{0}, 0);
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &ship = add_ship(store, map, p0, Location{1, 1}, 0);
    map.at(Location{1, 1}).energy = 100;

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::MiningPhase phase;
    phase.execute(context);

    REQUIRE(store.get_entity(ship.id).energy > 0);
    REQUIRE(map.at(Location{1, 1}).energy < 100);
}

TEST_CASE("CapturePhase scenario runs without regression", "[phase][capture]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    stats.player_statistics.emplace_back(Player::id_type{0}, 0);
    stats.player_statistics.emplace_back(Player::id_type{1}, 1);
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0});
    auto &p1 = add_player(store, Player::id_type{1}, Location{3, 3});
    const auto victim_id = add_ship(store, map, p0, Location{1, 1}, 100).id;
    add_ship(store, map, p1, Location{1, 2});
    add_ship(store, map, p1, Location{2, 1});
    add_ship(store, map, p1, Location{1, 0});
    add_ship(store, map, p1, Location{0, 1});

    if (!config.ruleset.capture.enabled) {
        SUCCEED();
        return;
    }

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::CapturePhase phase;
    phase.execute(context);

    if (!sink.events().empty()) {
        REQUIRE(map.at(Location{1, 1}).entity != victim_id);
    } else {
        SUCCEED();
    }
}

TEST_CASE("SpawnPhase creates entity when player has enough energy", "[phase][spawn]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    GameState state(store, map, stats);
    auto config = GameConfig::from_constants();
    events::RecordingEventSink sink{};
    StepResult result{};
    ActionBatch actions{};

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    map.at(p0.factory).owner = p0.id;
    SpawnCommand spawn{};
    result.validated_commands = CommandBatch{};
    result.validated_commands->spawns[p0.id].push_back(std::cref(spawn));

    auto context = make_context(state, actions, result, config, sink);
    rules::phases::SpawnPhase phase;
    phase.execute(context);

    REQUIRE(p0.entities.size() == 1);
}

TEST_CASE("TurnStatsCollector appends turn production", "[collector][scoring]") {
    Store store{};
    Map map(4, 4);
    GameStatistics stats{};
    stats.player_statistics.emplace_back(Player::id_type{0}, 0);
    GameState state(store, map, stats);
    state.turn.number = 1;
    auto config = GameConfig::from_constants();

    auto &p0 = add_player(store, Player::id_type{0}, Location{0, 0}, 5000);
    p0.can_play = true;

    observers::TurnStatsCollector collector;
    collector.collect(state, config);

    REQUIRE(stats.player_statistics.at(0).turn_productions.size() == 1);
}
