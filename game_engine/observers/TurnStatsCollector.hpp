#ifndef TURNSTATSCOLLECTOR_HPP
#define TURNSTATSCOLLECTOR_HPP

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "GameConfig.hpp"
#include "GameState.hpp"
#include "Map.hpp"
#include "Statistics.hpp"
#include "Store.hpp"
#include "TaskExecutor.hpp"

namespace hlt::observers {

class TurnStatsCollector final {
    struct PlayerTurnStatsUpdate {
        bool active{};
        long last_turn_alive{};
        energy_type carried_at_end{};
        energy_type structure_score{};
        energy_type total_production{};
        long number_dropoffs{};
        long ships_peak{};
        dimension_type max_entity_distance{};
        dimension_type total_distance{};
        dimension_type total_entity_lifespan{};
        long interaction_opportunities{};
        std::unordered_map<Location, energy_type> halite_per_dropoff{};
    };

    static PlayerTurnStatsUpdate collect_player_update(const GameState &state,
                                                       const GameConfig &config,
                                                       const PlayerStatistics &player_stats) {
        PlayerTurnStatsUpdate update{};
        const auto &store = state.store_ref();
        const auto &map = state.map_ref();
        const auto &player = store.players_ref().at(player_stats.player_id);

        if (player.terminated || !player.can_play) {
            return update;
        }

        update.active = true;
        update.number_dropoffs = static_cast<long>(player.dropoffs.size());
        update.ships_peak = static_cast<long>(player.entities.size());

        if (!player.entities.empty() || player.energy >= config.ruleset.economy.new_entity_energy_cost) {
            update.last_turn_alive = state.turn.number;
            for (const auto &[_entity_id, location] : player.entities) {
                (void)location;
                update.carried_at_end += store.get_entity(_entity_id).energy;
            }
        }

        update.structure_score = player.factory_halite;
        update.halite_per_dropoff[player.factory] = player.factory_halite;
        update.total_production = player.factory_halite;
        for (const auto &dropoff : player.dropoffs) {
            update.structure_score += dropoff.halite_pool;
            update.halite_per_dropoff[dropoff.location] = dropoff.halite_pool;
            update.total_production += dropoff.halite_pool;
        }

        for (const auto &[_entity_id, location] : player.entities) {
            const dimension_type entity_distance = map.distance(location, player.factory);
            if (entity_distance > update.max_entity_distance) {
                update.max_entity_distance = entity_distance;
            }
            update.total_distance += entity_distance;
            update.total_entity_lifespan++;

            std::unordered_set<Location> close_cells{};
            const auto neighbors = map.get_neighbors(location);
            close_cells.insert(neighbors.begin(), neighbors.end());
            for (const Location &neighbor : neighbors) {
                const auto cells_once_removed = map.get_neighbors(neighbor);
                close_cells.insert(cells_once_removed.begin(), cells_once_removed.end());
            }
            for (const Location &cell_location : close_cells) {
                const Cell &cell = map.at(cell_location);
                if (cell.entity != Entity::None && store.get_entity(cell.entity).owner != player.id) {
                    update.interaction_opportunities++;
                    break;
                }
                if (cell.owner != Player::None && cell.owner != player.id) {
                    update.interaction_opportunities++;
                    break;
                }
            }
        }

        return update;
    }

    static void apply_player_update(const GameState &state,
                                    const PlayerTurnStatsUpdate &update,
                                    PlayerStatistics &player_stats) {
        const auto &store = state.store_ref();
        const auto &player = store.players_ref().at(player_stats.player_id);

        if (!update.active) {
            player_stats.turn_productions.push_back(0);
            player_stats.turn_deposited.push_back(0);
            return;
        }

        if (update.last_turn_alive != 0) {
            player_stats.last_turn_alive = update.last_turn_alive;
            player_stats.carried_at_end = update.carried_at_end;
        }
        player_stats.turn_productions.push_back(update.structure_score);
        player_stats.turn_deposited.push_back(player.total_energy_deposited);
        player_stats.number_dropoffs = update.number_dropoffs;
        player_stats.ships_peak = std::max(player_stats.ships_peak, update.ships_peak);
        if (update.max_entity_distance > player_stats.max_entity_distance) {
            player_stats.max_entity_distance = update.max_entity_distance;
        }
        player_stats.total_distance += update.total_distance;
        player_stats.total_entity_lifespan += update.total_entity_lifespan;
        player_stats.interaction_opportunities += update.interaction_opportunities;
        player_stats.halite_per_dropoff = update.halite_per_dropoff;
        player_stats.total_production = update.total_production;
    }

public:
    void collect(GameState &state, const GameConfig &config, TaskExecutor &executor) const {
        auto &statistics = state.statistics_ref();
        std::vector<PlayerTurnStatsUpdate> updates(statistics.player_statistics.size());

        executor.parallel_for(0, statistics.player_statistics.size(), [&](std::size_t index) {
            updates[index] = collect_player_update(state, config, statistics.player_statistics[index]);
        });

        for (std::size_t index = 0; index < statistics.player_statistics.size(); ++index) {
            apply_player_update(state, updates[index], statistics.player_statistics[index]);
        }
    }

    void collect(GameState &state, const GameConfig &config) const {
        InlineExecutor executor;
        collect(state, config, executor);
    }
};

} // namespace hlt::observers

#endif // TURNSTATSCOLLECTOR_HPP
