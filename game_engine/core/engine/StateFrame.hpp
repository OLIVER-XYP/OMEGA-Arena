#ifndef STATEFRAME_HPP
#define STATEFRAME_HPP

#include <unordered_map>
#include <vector>

#include "GameState.hpp"

namespace hlt {

struct PlayerFrameEntry {
    Player::id_type id{Player::None};
    Location factory{0, 0};
    energy_type energy{};
    energy_type factory_energy_deposited{};
    energy_type total_energy_deposited{};
    energy_type factory_halite{};
    bool factory_destroyed{};
    std::size_t ship_begin{};
    std::size_t ship_count{};
    std::size_t dropoff_begin{};
    std::size_t dropoff_count{};
};

struct EntityFrameEntry {
    Entity::id_type id{Entity::None};
    Player::id_type owner{Player::None};
    Location location{0, 0};
    energy_type energy{};
    energy_type lifetime_deposited{};
    energy_type enemy_halite_taken{};
    int enemy_hp_dealt{};
    int hp{};
    bool alive{};
    bool was_captured{};
    bool is_inspired{};
    bool is_defending{};
    int protection_turns{};
};

struct DropoffFrameEntry {
    Player::id_type owner{Player::None};
    Location location{0, 0};
    energy_type deposited_halite{};
    energy_type halite_pool{};
    bool destroyed{};
};

struct StateFrame {
    dimension_type width{};
    dimension_type height{};
    unsigned long turn_number{};
    unsigned long long map_total_energy{};
    std::vector<energy_type> cell_energy;
    std::vector<energy_type> cell_initial_energy;
    std::vector<Entity::id_type> cell_entity;
    std::vector<Player::id_type> cell_owner;
    std::vector<PlayerFrameEntry> players;
    std::vector<EntityFrameEntry> entities;
    std::vector<Entity::id_type> player_entity_ids;
    std::vector<DropoffFrameEntry> dropoffs;
    std::unordered_map<Entity::id_type, std::size_t> entity_slot_by_id;
    std::unordered_map<Player::id_type, std::size_t> player_slot_by_id;
};

std::size_t cell_index(const StateFrame &frame, Location location);
StateFrame capture_state_frame(const GameState &state);
void apply_state_frame(GameState &state, const StateFrame &frame);
void apply_state_frame_entities(GameState &state, const StateFrame &frame);

} // namespace hlt

#endif // STATEFRAME_HPP
