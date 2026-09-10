#include "StateFrame.hpp"

#include <vector>

namespace hlt {

std::size_t cell_index(const StateFrame &frame, Location location) {
    return static_cast<std::size_t>(location.y * frame.width + location.x);
}

StateFrame capture_state_frame(const GameState &state) {
    StateFrame frame;
    const auto &map = state.map_ref();
    const auto &store = state.store_ref();
    frame.width = map.width;
    frame.height = map.height;
    frame.turn_number = state.turn.number;
    frame.map_total_energy = store.map_total_energy;

    const std::size_t cell_count = static_cast<std::size_t>(map.width * map.height);
    frame.cell_energy.resize(cell_count);
    frame.cell_initial_energy.resize(cell_count);
    frame.cell_entity.resize(cell_count);
    frame.cell_owner.resize(cell_count);

    for (dimension_type y = 0; y < map.height; ++y) {
        for (dimension_type x = 0; x < map.width; ++x) {
            const Location location{x, y};
            const auto index = cell_index(frame, location);
            const auto &cell = map.at(location);
            frame.cell_energy[index] = cell.energy;
            frame.cell_initial_energy[index] = cell.initial_energy;
            frame.cell_entity[index] = cell.entity;
            frame.cell_owner[index] = cell.owner;
        }
    }

    frame.players.reserve(store.players_ref().size());
    for (const auto &[player_id, player] : store.players_ref()) {
        PlayerFrameEntry entry;
        entry.id = player_id;
        entry.factory = player.factory;
        entry.energy = player.energy;
        entry.factory_energy_deposited = player.factory_energy_deposited;
        entry.total_energy_deposited = player.total_energy_deposited;
        entry.factory_halite = player.factory_halite;
        entry.factory_destroyed = player.factory_destroyed;
        entry.ship_begin = frame.player_entity_ids.size();
        entry.ship_count = player.entities.size();
        for (const auto &[entity_id, _location] : player.entities) {
            (void)_location;
            frame.player_entity_ids.push_back(entity_id);
        }
        entry.dropoff_begin = frame.dropoffs.size();
        entry.dropoff_count = player.dropoffs.size();
        for (const auto &dropoff : player.dropoffs) {
            frame.dropoffs.push_back(DropoffFrameEntry{player_id,
                                                       dropoff.location,
                                                       dropoff.deposited_halite,
                                                       dropoff.halite_pool,
                                                       dropoff.destroyed});
        }
        frame.player_slot_by_id.emplace(player_id, frame.players.size());
        frame.players.push_back(entry);
    }

    frame.entities.reserve(store.entities_ref().size());
    for (const auto &[entity_id, entity] : store.entities_ref()) {
        EntityFrameEntry entry;
        entry.id = entity_id;
        entry.owner = entity.owner;
        entry.energy = entity.energy;
        entry.lifetime_deposited = entity.lifetime_deposited;
        entry.enemy_halite_taken = entity.enemy_halite_taken;
        entry.enemy_hp_dealt = entity.enemy_hp_dealt;
        entry.hp = entity.hp;
        entry.alive = true;
        entry.was_captured = entity.was_captured;
        entry.is_inspired = entity.is_inspired;
        entry.is_defending = entity.is_defending;
        entry.protection_turns = entity.protection_turns;
        const auto player_it = store.players_ref().find(entity.owner);
        if (player_it != store.players_ref().end()) {
            const auto location_it = player_it->second.entities.find(entity_id);
            if (location_it != player_it->second.entities.end()) {
                entry.location = location_it->second;
            }
        }
        frame.entity_slot_by_id.emplace(entity_id, frame.entities.size());
        frame.entities.push_back(entry);
    }

    return frame;
}

void apply_state_frame(GameState &state, const StateFrame &frame) {
    auto &map = state.map_ref();
    auto &store = state.store_ref();
    store.map_total_energy = frame.map_total_energy;

    for (dimension_type y = 0; y < map.height; ++y) {
        for (dimension_type x = 0; x < map.width; ++x) {
            const Location location{x, y};
            const auto index = cell_index(frame, location);
            auto &cell = map.at(location);
            cell.energy = frame.cell_energy[index];
            cell.initial_energy = frame.cell_initial_energy[index];
            cell.entity = frame.cell_entity[index];
            cell.owner = frame.cell_owner[index];
        }
    }

    for (const auto &player_entry : frame.players) {
        auto player_it = store.players_ref().find(player_entry.id);
        if (player_it == store.players_ref().end()) {
            continue;
        }
        auto &player = player_it->second;
        player.energy = player_entry.energy;
        player.factory_energy_deposited = player_entry.factory_energy_deposited;
        player.total_energy_deposited = player_entry.total_energy_deposited;
        player.factory_halite = player_entry.factory_halite;
        player.factory_destroyed = player_entry.factory_destroyed;
        const auto dropoff_end = player_entry.dropoff_begin + player_entry.dropoff_count;
        for (std::size_t index = player_entry.dropoff_begin; index < dropoff_end && index < frame.dropoffs.size(); ++index) {
            const auto &dropoff_entry = frame.dropoffs[index];
            for (auto &dropoff : player.dropoffs) {
                if (dropoff.location == dropoff_entry.location) {
                    dropoff.deposited_halite = dropoff_entry.deposited_halite;
                    dropoff.halite_pool = dropoff_entry.halite_pool;
                    dropoff.destroyed = dropoff_entry.destroyed;
                    break;
                }
            }
        }
    }

    for (const auto &entity_entry : frame.entities) {
        auto entity_it = store.entities_ref().find(entity_entry.id);
        if (entity_it == store.entities_ref().end()) {
            continue;
        }
        auto &entity = entity_it->second;
        entity.energy = entity_entry.energy;
        entity.lifetime_deposited = entity_entry.lifetime_deposited;
        entity.enemy_halite_taken = entity_entry.enemy_halite_taken;
        entity.enemy_hp_dealt = entity_entry.enemy_hp_dealt;
        entity.hp = entity_entry.hp;
        entity.was_captured = entity_entry.was_captured;
        entity.is_inspired = entity_entry.is_inspired;
        entity.is_defending = entity_entry.is_defending;
        entity.protection_turns = entity_entry.protection_turns;
    }
}

void apply_state_frame_entities(GameState &state, const StateFrame &frame) {
    apply_state_frame(state, frame);

    auto &store = state.store_ref();
    for (auto &[player_id, player] : store.players_ref()) {
        (void)player_id;
        player.entities.clear();
    }

    std::vector<Entity::id_type> dead_entities;
    for (const auto &entity_entry : frame.entities) {
        if (entity_entry.id == Entity::None) {
            continue;
        }
        auto entity_it = store.entities_ref().find(entity_entry.id);
        if (!entity_entry.alive) {
            if (entity_it != store.entities_ref().end()) {
                dead_entities.push_back(entity_entry.id);
            }
            continue;
        }
        if (entity_it == store.entities_ref().end()) {
            continue;
        }
        auto &entity = entity_it->second;
        entity.energy = entity_entry.energy;
        entity.lifetime_deposited = entity_entry.lifetime_deposited;
        entity.enemy_halite_taken = entity_entry.enemy_halite_taken;
        entity.enemy_hp_dealt = entity_entry.enemy_hp_dealt;
        entity.hp = entity_entry.hp;
        entity.was_captured = entity_entry.was_captured;
        entity.is_inspired = entity_entry.is_inspired;
        entity.is_defending = entity_entry.is_defending;
        entity.protection_turns = entity_entry.protection_turns;
        auto player_it = store.players_ref().find(entity.owner);
        if (player_it != store.players_ref().end()) {
            player_it->second.add_entity(entity.id, entity_entry.location);
        }
    }

    for (const auto entity_id : dead_entities) {
        if (store.entities_ref().find(entity_id) != store.entities_ref().end()) {
            store.delete_entity(entity_id);
        }
    }
}

} // namespace hlt
