#ifndef PHASEHELPERS_HPP
#define PHASEHELPERS_HPP

#include <functional>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "CommandError.hpp"
#include "RuleContext.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {

inline void append_error(StepResult &result, CommandError error) {
    result.non_fatal_errors.push_back(error->log_message());
}

inline void dump_energy(Store &store, const Location &location, Cell &cell, energy_type energy) {
    if (cell.owner == Player::None) {
        cell.energy += energy;
        store.map_total_energy += energy;
    } else {
        auto &player = store.get_player(cell.owner);
        player.energy += energy;
        player.total_energy_deposited += energy;
        if (location == player.factory) {
            player.factory_energy_deposited += energy;
            if (!player.factory_destroyed) {
                player.factory_halite += energy;
            }
        } else {
            for (auto &dropoff : player.dropoffs) {
                if (dropoff.location == location) {
                    dropoff.deposited_halite += energy;
                    if (!dropoff.destroyed) {
                        dropoff.halite_pool += energy;
                    }
                    return;
                }
            }
        }
    }
}

inline void dump_energy(Store &store, Entity &entity, const Location &location, Cell &cell, energy_type energy) {
    entity.energy -= energy;
    if (cell.owner == entity.owner) {
        entity.lifetime_deposited += energy;
    }
    dump_energy(store, location, cell, energy);
}

inline energy_type scaled_spawn_cost(energy_type base, std::size_t num_ships, double growth) {
    if (growth <= 0.0) return base;
    return static_cast<energy_type>(base * (1.0 + growth * static_cast<double>(num_ships)));
}

// Overload that also adds a quadratic over-threshold term: when this spawn
// would push the ship count strictly above quad_threshold (i.e., the new
// total num_ships + 1 > quad_threshold), the cost picks up an extra
//      base × quad_growth × (num_ships + 1 - quad_threshold)²
// component on top of the linear-growth term.  The first quad_threshold
// spawns (those that bring the count to exactly quad_threshold or below) are
// untouched, preserving v4 spawn behaviour up to that count.
inline energy_type scaled_spawn_cost(energy_type base,
                                     std::size_t num_ships,
                                     double growth,
                                     std::size_t quad_threshold,
                                     double quad_growth) {
    double factor = 1.0;
    if (growth > 0.0) {
        factor += growth * static_cast<double>(num_ships);
    }
    if (quad_growth > 0.0 && (num_ships + 1) > quad_threshold) {
        const auto excess = (num_ships + 1) - quad_threshold;
        factor += quad_growth * static_cast<double>(excess) * static_cast<double>(excess);
    }
    return static_cast<energy_type>(base * factor);
}

inline energy_type scaled_dropoff_cost(energy_type base, std::size_t num_dropoffs, double growth) {
    if (growth <= 0.0) return base;
    return static_cast<energy_type>(base * (1.0 + growth * static_cast<double>(num_dropoffs)));
}

template <typename TargetRange, typename SourceRange>
void append_range(TargetRange &target, const SourceRange &source) {
    target.insert(target.end(), source.begin(), source.end());
}

template <typename TargetRange, typename SourceRange>
void append_move_range(TargetRange &target, SourceRange &&source) {
    target.insert(target.end(),
                  std::make_move_iterator(source.begin()),
                  std::make_move_iterator(source.end()));
}

template <typename Plan>
std::vector<std::optional<Plan>> parallel_plan(
    RuleContext &context,
    std::size_t count,
    const std::function<void(std::size_t, std::vector<std::optional<Plan>> &)> &planner) {
    std::vector<std::optional<Plan>> plans(count);
    InlineExecutor fallback_executor;
    auto *executor = context.executor() != nullptr ? context.executor() : &fallback_executor;
    executor->parallel_for(0, count, [&](std::size_t index) {
        planner(index, plans);
    });
    return plans;
}

template <typename PlanRange, typename Fn>
void for_each_planned(const PlanRange &plans, Fn &&fn) {
    for (const auto &planned : plans) {
        if (planned.has_value()) {
            fn(*planned);
        }
    }
}

template <typename CommandT, typename MapT>
auto flatten_player_commands(const MapT &commands_by_player) {
    std::vector<std::pair<Player::id_type, const CommandT *>> commands;
    for (const auto &[player_id, command_list] : commands_by_player) {
        for (const CommandT &command : command_list) {
            commands.emplace_back(player_id, &command);
        }
    }
    return commands;
}

template <typename CommandT, typename MapT, typename Predicate>
auto flatten_player_commands_if(const MapT &commands_by_player, Predicate &&predicate) {
    std::vector<std::pair<Player::id_type, const CommandT *>> commands;
    for (const auto &[player_id, command_list] : commands_by_player) {
        for (const CommandT &command : command_list) {
            if (predicate(command)) {
                commands.emplace_back(player_id, &command);
            }
        }
    }
    return commands;
}

} // namespace hlt::rules::phases

#endif // PHASEHELPERS_HPP
