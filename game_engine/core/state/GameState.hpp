#ifndef GAMESTATE_HPP
#define GAMESTATE_HPP

#include "Map.hpp"
#include "Statistics.hpp"
#include "Store.hpp"

namespace hlt {

struct TurnState {
    unsigned long number{};
};

class GameStateView;

struct GameState {
    TurnState turn;
    Store *store{};
    Map *map{};
    GameStatistics *statistics{};

    GameState() = default;
    GameState(Store &store_ref, Map &map_ref, GameStatistics &statistics_ref)
        : store(&store_ref), map(&map_ref), statistics(&statistics_ref) {}

    Store &store_ref() { return *store; }
    const Store &store_ref() const { return *store; }
    Map &map_ref() { return *map; }
    const Map &map_ref() const { return *map; }
    GameStatistics &statistics_ref() { return *statistics; }
    const GameStatistics &statistics_ref() const { return *statistics; }
    GameStateView view() const;
};

class GameStateView {
    const GameState &state;

public:
    explicit GameStateView(const GameState &state) : state(state) {}
    unsigned long turn_number() const { return state.turn.number; }
    const Store &store_ref() const { return state.store_ref(); }
    const Map &map_ref() const { return state.map_ref(); }
    const GameStatistics &statistics_ref() const { return state.statistics_ref(); }
};

inline GameStateView GameState::view() const {
    return GameStateView(*this);
}

} // namespace hlt

#endif // GAMESTATE_HPP
