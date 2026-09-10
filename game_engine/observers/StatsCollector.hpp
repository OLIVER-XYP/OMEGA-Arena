#ifndef STATSCOLLECTOR_HPP
#define STATSCOLLECTOR_HPP

#include "EventSink.hpp"
#include "Map.hpp"
#include "Statistics.hpp"
#include "Store.hpp"

namespace hlt::observers {

class StatsCollector final : public events::EventSink {
    GameStatistics &stats;
    const Store &store;
    const Map &map;

public:
    StatsCollector(GameStatistics &stats, const Store &store, const Map &map)
        : stats(stats), store(store), map(map) {}

    void emit(const events::DomainEvent &event) override {
        std::visit([this](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, events::SpawnedEvent>) {
                stats.player_statistics.at(value.player.value).ships_spawned++;
                stats.player_statistics.at(value.player.value).last_turn_ship_spawn = stats.turn_number;
            } else if constexpr (std::is_same_v<T, events::MinedEvent>) {
                auto &player_stats = stats.player_statistics.at(value.owner.value);
                player_stats.total_mined += value.extracted;
                player_stats.total_bonus += value.gained > value.extracted ? value.gained - value.extracted : 0;
                if (value.was_captured) {
                    player_stats.total_mined_from_captured += value.gained;
                }
            } else if constexpr (std::is_same_v<T, events::CapturedEvent>) {
                stats.player_statistics.at(value.old_owner.value).ships_given++;
                stats.player_statistics.at(value.new_owner.value).ships_captured++;
            } else if constexpr (std::is_same_v<T, events::CollisionResolvedEvent>) {
                ordered_id_map<Player, int> ships_involved;
                for (const auto &ship_id : value.ships) {
                    const auto &entity = store.get_entity(ship_id);
                    auto &player_stats = stats.player_statistics.at(entity.owner.value);
                    ships_involved[entity.owner]++;
                    player_stats.all_collisions++;
                    if (map.at(value.location).owner == entity.owner) {
                        player_stats.dropoff_collisions++;
                    } else {
                        player_stats.total_dropped += entity.energy;
                    }
                }
                for (const auto &[player_id, num_ships] : ships_involved) {
                    if (num_ships > 1) {
                        stats.player_statistics.at(player_id.value).self_collisions += num_ships;
                    }
                }
            }
        }, event);
    }
};

} // namespace hlt::observers

#endif // STATSCOLLECTOR_HPP
