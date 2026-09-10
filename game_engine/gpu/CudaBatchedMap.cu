#include "CudaBatchedMap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {
namespace {

struct RegenConfigDevice {
    bool enabled{};
    double rate{};
    double cap_fraction{};
};

struct MiningConfigDevice {
    energy_type max_energy{};
    unsigned long extract_ratio{};
    unsigned long inspired_extract_ratio{};
    double inspired_bonus_multiplier{};
};

struct InspirationConfigDevice {
    bool enabled{};
    int radius{};
    unsigned long ship_count_threshold{};
};

struct SpawnApplyConfigDevice {
    energy_type base_cost{};
    double growth{};
    unsigned long quad_threshold{};
    double quad_growth{};
    int initial_hp{};
    unsigned long iterations{};
};

struct MovementConfigDevice {
    energy_type move_cost_ratio{};
    energy_type inspired_move_cost_ratio{};
};

struct CollisionConfigDevice {
    energy_type move_cost_ratio{};
    energy_type inspired_move_cost_ratio{};
    int collision_hp_damage{};
};

struct ValidationConfigDevice {
    energy_type dropoff_cost{};
    double dropoff_growth{};
    energy_type spawn_cost{};
    double spawn_growth{};
    unsigned long spawn_quad_threshold{};
    double spawn_quad_growth{};
    bool combat_enabled{};
};

struct OverShipTaxConfigDevice {
    bool tax_on{};
    bool income_on{};
    bool quad_on{};
    unsigned long over_ship_tax_threshold{};
    energy_type over_ship_tax_per_turn{};
    energy_type ship_income_per_turn{};
    unsigned long ship_count_target{};
    energy_type ship_count_deviation_penalty{};
};

struct HaliteRebalanceConfigDevice {
    bool enabled{};
    unsigned long period{};
    double fraction{};
    double min_gap_frac{};
};

static_assert(sizeof(energy_type) == sizeof(int), "batched dump atomics assume 32-bit energy_type");

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceVector {
    T *ptr{};
    std::size_t count{};

public:
    explicit DeviceVector(std::size_t count) : count(count) {
        if (count > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceVector() {
        cudaFree(ptr);
    }

    DeviceVector(const DeviceVector &) = delete;
    DeviceVector &operator=(const DeviceVector &) = delete;

    T *data() { return ptr; }
    const T *data() const { return ptr; }
    std::size_t size() const { return count; }
};

template <typename T>
class ReusableDeviceVector {
    T *ptr{};
    std::size_t count{};
    std::size_t capacity{};

public:
    ReusableDeviceVector() = default;

    ~ReusableDeviceVector() {
        cudaFree(ptr);
    }

    ReusableDeviceVector(const ReusableDeviceVector &) = delete;
    ReusableDeviceVector &operator=(const ReusableDeviceVector &) = delete;

    void resize(std::size_t new_count) {
        count = new_count;
        if (new_count <= capacity) {
            return;
        }
        if (ptr != nullptr) {
            check_cuda(cudaFree(ptr), "cudaFree reusable vector");
        }
        capacity = new_count;
        if (capacity > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr), capacity * sizeof(T)), "cudaMalloc reusable vector");
        } else {
            ptr = nullptr;
        }
    }

    T *data() { return ptr; }
    const T *data() const { return ptr; }
    std::size_t size() const { return count; }
};

__global__ void batched_regen_kernel(energy_type *cell_energy,
                                     const energy_type *cell_initial_energy,
                                     const Player::id_type *cell_owner,
                                     unsigned long long *game_deltas,
                                     std::size_t total_cells,
                                     std::size_t cells_per_game,
                                     RegenConfigDevice config,
                                     unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_cells || !config.enabled || config.rate <= 0.0 || config.cap_fraction <= 0.0) {
        return;
    }
    if (cell_owner[index].value != Player::None.value || cell_initial_energy[index] <= 0) {
        return;
    }
    const auto cap = static_cast<energy_type>(config.cap_fraction * static_cast<double>(cell_initial_energy[index]));
    const auto regen = static_cast<energy_type>(ceil(config.rate * static_cast<double>(cell_initial_energy[index])));
    if (regen <= 0) {
        return;
    }
    unsigned long long total_delta = 0;
    auto energy = cell_energy[index];
    for (unsigned long iteration = 0; iteration < iterations && energy < cap; ++iteration) {
        const auto candidate = energy + regen;
        const auto new_energy = candidate < cap ? candidate : cap;
        const auto delta = new_energy - energy;
        if (delta <= 0) {
            break;
        }
        energy = new_energy;
        total_delta += static_cast<unsigned long long>(delta);
    }
    if (total_delta == 0) {
        return;
    }
    cell_energy[index] = energy;
    const auto game_index = index / cells_per_game;
    atomicAdd(&game_deltas[game_index], total_delta);
}

__device__ int find_batched_player_index(const PlayerFrameEntry *players,
                                         std::size_t players_per_game,
                                         std::size_t player_offset,
                                         Player::id_type player_id) {
    for (std::size_t index = 0; index < players_per_game; ++index) {
        if (players[player_offset + index].id.value == player_id.value) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

__device__ void atomic_add_energy(energy_type *target, energy_type value) {
    atomicAdd(reinterpret_cast<int *>(target), static_cast<int>(value));
}

__device__ energy_type scaled_spawn_cost_device(energy_type base,
                                                std::size_t num_ships,
                                                double growth,
                                                unsigned long quad_threshold,
                                                double quad_growth) {
    double factor = 1.0;
    if (growth > 0.0) {
        factor += growth * static_cast<double>(num_ships);
    }
    if (quad_growth > 0.0 && (num_ships + 1) > quad_threshold) {
        const auto excess = static_cast<double>((num_ships + 1) - quad_threshold);
        factor += quad_growth * excess * excess;
    }
    return static_cast<energy_type>(static_cast<double>(base) * factor + 1.0e-9);
}

__device__ energy_type scaled_dropoff_cost_device(energy_type base, std::size_t num_dropoffs, double growth) {
    if (growth <= 0.0) {
        return base;
    }
    return static_cast<energy_type>(static_cast<double>(base) * (1.0 + growth * static_cast<double>(num_dropoffs)) + 1.0e-9);
}

__device__ int wrapped_distance(int width, int height, int ax, int ay, int bx, int by) {
    int dx = ax > bx ? ax - bx : bx - ax;
    int dy = ay > by ? ay - by : by - ay;
    const int wx = width - dx;
    const int wy = height - dy;
    return (dx < wx ? dx : wx) + (dy < wy ? dy : wy);
}

__device__ Location move_location_device(Location location, Direction direction, int width, int height) {
    switch (direction) {
    case Direction::North:
        location.y = (location.y + height - 1) % height;
        break;
    case Direction::South:
        location.y = (location.y + 1) % height;
        break;
    case Direction::East:
        location.x = (location.x + 1) % width;
        break;
    case Direction::West:
        location.x = (location.x + width - 1) % width;
        break;
    case Direction::Still:
        break;
    }
    return location;
}

__global__ void batched_dump_kernel(EntityFrameEntry *entities,
                                    std::size_t total_entities,
                                    std::size_t entities_per_game,
                                    PlayerFrameEntry *players,
                                    std::size_t players_per_game,
                                    DropoffFrameEntry *dropoffs,
                                    std::size_t dropoffs_per_game,
                                    const Player::id_type *cell_owner,
                                    std::size_t cells_per_game,
                                    unsigned long long *game_deposits,
                                    int width,
                                    int max_hp,
                                    unsigned long iterations,
                                    energy_type refill_cargo,
                                    bool refill_first) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive) {
        return;
    }

    const auto game_index = index / entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto player_offset = game_index * players_per_game;
    const auto dropoff_offset = game_index * dropoffs_per_game;
    const auto cell_index = cell_offset + static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    if (cell_owner[cell_index].value != entity.owner.value) {
        return;
    }

    const auto player_slot = find_batched_player_index(players, players_per_game, player_offset, entity.owner);
    if (player_slot < 0) {
        return;
    }
    const auto player_index = player_offset + static_cast<std::size_t>(player_slot);
    auto player = players[player_index];
    unsigned long long total_deposited = 0;
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        if ((iteration > 0 || refill_first) && refill_cargo > 0) {
            entity.energy = refill_cargo;
        }
        const auto deposited = entity.energy;
        if (deposited <= 0) {
            continue;
        }
        entity.energy = 0;
        entity.lifetime_deposited += deposited;
        entity.hp = max_hp;
        total_deposited += static_cast<unsigned long long>(deposited);
    }
    if (total_deposited == 0) {
        return;
    }
    entities[index] = entity;

    const auto deposited_energy = static_cast<energy_type>(total_deposited);
    atomic_add_energy(&players[player_index].energy, deposited_energy);
    atomic_add_energy(&players[player_index].total_energy_deposited, deposited_energy);
    atomicAdd(&game_deposits[game_index], total_deposited);

    if (entity.location.x == player.factory.x && entity.location.y == player.factory.y) {
        atomic_add_energy(&players[player_index].factory_energy_deposited, deposited_energy);
        if (!player.factory_destroyed) {
            atomic_add_energy(&players[player_index].factory_halite, deposited_energy);
        }
        return;
    }

    const auto dropoff_end = player.dropoff_begin + player.dropoff_count;
    for (std::size_t dropoff_index = player.dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
        const auto absolute_dropoff_index = dropoff_offset + dropoff_index;
        if (absolute_dropoff_index >= dropoff_offset + dropoffs_per_game) {
            break;
        }
        auto dropoff = dropoffs[absolute_dropoff_index];
        if (dropoff.location.x != entity.location.x || dropoff.location.y != entity.location.y) {
            continue;
        }
        atomic_add_energy(&dropoffs[absolute_dropoff_index].deposited_halite, deposited_energy);
        if (!dropoff.destroyed) {
            atomic_add_energy(&dropoffs[absolute_dropoff_index].halite_pool, deposited_energy);
        }
        return;
    }
}

__global__ void batched_mining_kernel(EntityFrameEntry *entities,
                                      std::size_t total_entities,
                                      std::size_t entities_per_game,
                                      energy_type *cell_energy,
                                      std::size_t cells_per_game,
                                      unsigned long long *game_extracted,
                                      int width,
                                      MiningConfigDevice config,
                                      unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive) {
        return;
    }

    unsigned long long total_extracted = 0;
    const auto game_index = index / entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto cell_index = cell_offset + static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        if (entity.energy >= config.max_energy || cell_energy[cell_index] <= 0) {
            break;
        }
        const auto ratio = entity.is_inspired ? config.inspired_extract_ratio : config.extract_ratio;
        energy_type extracted = static_cast<energy_type>(ceil(static_cast<double>(cell_energy[cell_index]) /
                                                              static_cast<double>(ratio)));
        energy_type gained = extracted;
        if (extracted == 0 && cell_energy[cell_index] > 0) {
            extracted = cell_energy[cell_index];
            gained = extracted;
        }
        if (extracted + entity.energy > config.max_energy) {
            extracted = config.max_energy - entity.energy;
        }
        if (entity.is_inspired && config.inspired_bonus_multiplier > 0.0) {
            gained += static_cast<energy_type>(config.inspired_bonus_multiplier * static_cast<double>(gained));
        }
        if (config.max_energy - entity.energy < gained) {
            gained = config.max_energy - entity.energy;
        }
        if (extracted <= 0 && gained <= 0) {
            break;
        }
        entity.energy += gained;
        cell_energy[cell_index] -= extracted;
        total_extracted += static_cast<unsigned long long>(extracted);
    }
    entities[index] = entity;
    if (total_extracted > 0) {
        atomicAdd(&game_extracted[game_index], total_extracted);
    }
}

__global__ void batched_inspiration_kernel(EntityFrameEntry *entities,
                                           std::size_t total_entities,
                                           std::size_t entities_per_game,
                                           const Entity::id_type *cell_entity,
                                           std::size_t cells_per_game,
                                           int width,
                                           int height,
                                           InspirationConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }
    auto entity = entities[index];
    if (!entity.alive || !config.enabled) {
        entities[index].is_inspired = false;
        return;
    }

    const auto game_index = index / entities_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    unsigned long opponents = 0;
    for (int dx = -config.radius; dx <= config.radius; ++dx) {
        for (int dy = -config.radius; dy <= config.radius; ++dy) {
            const int cur_x = (entity.location.x + dx + width) % width;
            const int cur_y = (entity.location.y + dy + height) % height;
            if (wrapped_distance(width, height, entity.location.x, entity.location.y, cur_x, cur_y) > config.radius) {
                continue;
            }
            const auto cell_index = cell_offset + static_cast<std::size_t>(cur_y * width + cur_x);
            const auto other_id = cell_entity[cell_index];
            if (other_id.value == Entity::None.value) {
                continue;
            }
            for (std::size_t other_index = 0; other_index < entities_per_game; ++other_index) {
                const auto other = entities[entity_offset + other_index];
                if (other.id.value == other_id.value && other.alive && other.owner.value != entity.owner.value) {
                    ++opponents;
                    break;
                }
            }
        }
    }
    entities[index].is_inspired = opponents >= config.ship_count_threshold;
}

__global__ void summarize_batched_inspiration_kernel(const EntityFrameEntry *entities,
                                                     std::size_t total_entities,
                                                     std::size_t entities_per_game,
                                                     unsigned long long *game_inspired) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }
    const auto entity = entities[index];
    if (!entity.alive || !entity.is_inspired) {
        return;
    }
    const auto game_index = index / entities_per_game;
    atomicAdd(&game_inspired[game_index], 1ULL);
}

__global__ void batched_spawn_apply_kernel(PlayerFrameEntry *players,
                                           std::size_t total_players,
                                           std::size_t players_per_game,
                                           EntityFrameEntry *entities,
                                           std::size_t entity_capacity_per_game,
                                           Entity::id_type *cell_entity,
                                           std::size_t cells_per_game,
                                           unsigned long long *game_spawned,
                                           int width,
                                           int current_turn,
                                           SpawnApplyConfigDevice config) {
    const auto player_global_index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (player_global_index >= total_players) {
        return;
    }

    const auto game_index = player_global_index / players_per_game;
    auto player = players[player_global_index];
    const auto cell_index = game_index * cells_per_game + static_cast<std::size_t>(player.factory.y * width + player.factory.x);
    if (cell_entity[cell_index].value != Entity::None.value) {
        players[player_global_index] = player;
        return;
    }

    const auto entity_offset = game_index * entity_capacity_per_game;
    unsigned long long spawned = 0;
    for (unsigned long iteration = 0; iteration < config.iterations; ++iteration) {
        const auto cost = scaled_spawn_cost_device(config.base_cost,
                                                   player.ship_count,
                                                   config.growth,
                                                   config.quad_threshold,
                                                   config.quad_growth);
        if (player.energy < cost) {
            break;
        }
        const auto spawn_slot = player.ship_begin + player.ship_count;
        if (spawn_slot >= entity_capacity_per_game) {
            break;
        }
        const auto entity_index = entity_offset + spawn_slot;
        auto entity = entities[entity_index];
        if (entity.alive) {
            break;
        }
        player.energy -= cost;
        entity.id.value = static_cast<int>(entity_index + 1);
        entity.owner = player.id;
        entity.location = player.factory;
        entity.energy = 0;
        entity.lifetime_deposited = 0;
        entity.enemy_halite_taken = 0;
        entity.enemy_hp_dealt = 0;
        entity.hp = config.initial_hp;
        entity.alive = true;
        entity.was_captured = false;
        entity.is_inspired = false;
        entity.is_defending = false;
        entity.protection_turns = 0;
        entities[entity_index] = entity;
        cell_entity[cell_index] = entity.id;
        // The benchmark keeps factories clear between iterations so every round can exercise spawn.
        cell_entity[cell_index] = Entity::None;
        ++player.ship_count;
        ++spawned;
    }
    players[player_global_index] = player;
    if (spawned > 0) {
        atomicAdd(&game_spawned[game_index], spawned);
    }
    (void)current_turn;
}

__global__ void batched_movement_decision_kernel(const FlatCommand *commands,
                                                 std::size_t total_commands,
                                                 std::size_t commands_per_game,
                                                 const EntityFrameEntry *entities,
                                                 std::size_t entities_per_game,
                                                 const energy_type *cell_energy,
                                                 std::size_t cells_per_game,
                                                 int width,
                                                 int height,
                                                 MovementConfigDevice config,
                                                 MovementFrameDecision *decisions,
                                                 unsigned long long *game_movable,
                                                 unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_commands) {
        return;
    }
    const auto game_index = index / commands_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    auto decision = decisions[index];
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        const auto command = commands[index];
        decision.player = command.player;
        decision.entity = command.entity;
        decision.from = command.target_location;
        decision.to = command.target_location;
        decision.required = 0;
        decision.current_energy = 0;
        decision.command = command.type == FlatCommandType::Move && command.direction != Direction::Still;
        decision.entity_missing = false;
        decision.insufficient_energy = false;
        if (!decision.command) {
            continue;
        }

        int entity_index = -1;
        for (std::size_t local = 0; local < entities_per_game; ++local) {
            const auto entity = entities[entity_offset + local];
            if (entity.id.value == command.entity.value &&
                entity.owner.value == command.player.value &&
                entity.alive) {
                entity_index = static_cast<int>(entity_offset + local);
                break;
            }
        }
        if (entity_index < 0) {
            decision.entity_missing = true;
            continue;
        }
        const auto entity = entities[entity_index];
        decision.from = entity.location;
        decision.to = move_location_device(entity.location, command.direction, width, height);
        const auto source_index = cell_offset + static_cast<std::size_t>(entity.location.y * width + entity.location.x);
        const auto cost_ratio = entity.is_inspired ? config.inspired_move_cost_ratio : config.move_cost_ratio;
        decision.required = cost_ratio == 0 ? 0 : cell_energy[source_index] / cost_ratio;
        decision.current_energy = entity.energy;
        decision.insufficient_energy = entity.energy < decision.required;
    }
    decisions[index] = decision;
    if (!decision.insufficient_energy) {
        atomicAdd(&game_movable[game_index], 1ULL);
    }
}

__global__ void batched_movement_apply_no_collision_kernel(const FlatCommand *commands,
                                                           const int *command_entity_slots,
                                                           std::size_t total_commands,
                                                           std::size_t commands_per_game,
                                                           EntityFrameEntry *entities,
                                                           std::size_t entities_per_game,
                                                           energy_type *cell_energy,
                                                           Entity::id_type *cell_entity,
                                                           std::size_t cells_per_game,
                                                           int width,
                                                           int height,
                                                           MovementConfigDevice config,
                                                           unsigned long long *game_applied,
                                                           unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_commands) {
        return;
    }
    const auto game_index = index / commands_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto command = commands[index];
    if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
        return;
    }

    const auto entity_slot = command_entity_slots[index];
    if (entity_slot < 0 || static_cast<std::size_t>(entity_slot) >= entities_per_game) {
        return;
    }
    const auto entity_index = entity_offset + static_cast<std::size_t>(entity_slot);

    auto entity = entities[entity_index];
    if (entity.id.value != command.entity.value ||
        entity.owner.value != command.player.value ||
        !entity.alive) {
        return;
    }
    unsigned long long applied = 0;
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        const auto source_local = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
        const auto source_index = cell_offset + source_local;
        const auto cost_ratio = entity.is_inspired ? config.inspired_move_cost_ratio : config.move_cost_ratio;
        const auto required = cost_ratio == 0 ? 0 : cell_energy[source_index] / cost_ratio;
        if (entity.energy < required) {
            break;
        }
        const auto destination = move_location_device(entity.location, command.direction, width, height);
        entity.energy -= required;
        entity.location = destination;
        ++applied;
    }
    entities[entity_index] = entity;
    if (applied > 0) {
        atomicAdd(&game_applied[game_index], applied);
    }
}

__global__ void clear_batched_cell_entities_kernel(Entity::id_type *cell_entity,
                                                   std::size_t total_cells) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_cells) {
        return;
    }
    cell_entity[index].value = Entity::None.value;
}

__global__ void scatter_batched_entities_to_cells_kernel(const EntityFrameEntry *entities,
                                                         std::size_t total_entities,
                                                         std::size_t entities_per_game,
                                                         Entity::id_type *cell_entity,
                                                         std::size_t cells_per_game,
                                                         int width) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }
    const auto entity = entities[index];
    if (!entity.alive || entity.id.value == Entity::None.value) {
        return;
    }
    const auto game_index = index / entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto cell = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    if (cell >= cells_per_game) {
        return;
    }
    cell_entity[cell_offset + cell].value = entity.id.value;
}

__global__ void batched_destination_count_kernel(const FlatCommand *commands,
                                                 const int *command_entity_slots,
                                                 std::size_t total_commands,
                                                 std::size_t commands_per_game,
                                                 const EntityFrameEntry *entities,
                                                 std::size_t entities_per_game,
                                                 const energy_type *cell_energy,
                                                 std::size_t cells_per_game,
                                                 int width,
                                                 int height,
                                                 MovementConfigDevice config,
                                                 unsigned int *destination_counts,
                                                 unsigned long long *game_moved,
                                                 unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_commands) {
        return;
    }
    const auto game_index = index / commands_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto command = commands[index];
    if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
        return;
    }
    const auto entity_slot = command_entity_slots[index];
    if (entity_slot < 0 || static_cast<std::size_t>(entity_slot) >= entities_per_game) {
        return;
    }
    const auto entity_index = entity_offset + static_cast<std::size_t>(entity_slot);
    const auto entity = entities[entity_index];
    if (entity.id.value != command.entity.value ||
        entity.owner.value != command.player.value ||
        !entity.alive) {
        return;
    }
    const auto source_local = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    const auto source_index = cell_offset + source_local;
    const auto cost_ratio = entity.is_inspired ? config.inspired_move_cost_ratio : config.move_cost_ratio;
    const auto required = cost_ratio == 0 ? 0 : cell_energy[source_index] / cost_ratio;
    if (entity.energy < required) {
        return;
    }
    const auto destination = move_location_device(entity.location, command.direction, width, height);
    const auto destination_index = cell_offset + static_cast<std::size_t>(destination.y * width + destination.x);
    atomicAdd(&destination_counts[destination_index], static_cast<unsigned int>(iterations));
    atomicAdd(&game_moved[game_index], static_cast<unsigned long long>(iterations));
}

__global__ void summarize_batched_destination_counts_kernel(const unsigned int *destination_counts,
                                                            std::size_t total_cells,
                                                            std::size_t cells_per_game,
                                                            unsigned long long *game_conflict_cells,
                                                            unsigned long long *game_max_arrivals,
                                                            unsigned long long *game_total_arrivals) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_cells) {
        return;
    }
    const auto count = destination_counts[index];
    if (count == 0) {
        return;
    }
    const auto game_index = index / cells_per_game;
    atomicAdd(&game_total_arrivals[game_index], static_cast<unsigned long long>(count));
    atomicMax(&game_max_arrivals[game_index], static_cast<unsigned long long>(count));
    if (count > 1U) {
        atomicAdd(&game_conflict_cells[game_index], 1ULL);
    }
}

__global__ void batched_collision_move_stage_kernel(const FlatCommand *commands,
                                                    const int *command_entity_slots,
                                                    std::size_t total_commands,
                                                    std::size_t commands_per_game,
                                                    EntityFrameEntry *entities,
                                                    std::size_t entities_per_game,
                                                    const energy_type *cell_energy,
                                                    Entity::id_type *cell_entity,
                                                    std::size_t cells_per_game,
                                                    int width,
                                                    int height,
                                                    CollisionConfigDevice config,
                                                    unsigned int *destination_counts,
                                                    unsigned long long *game_moved) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_commands) {
        return;
    }
    const auto game_index = index / commands_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto command = commands[index];
    if (command.type != FlatCommandType::Move || command.direction == Direction::Still) {
        return;
    }
    const auto entity_slot = command_entity_slots[index];
    if (entity_slot < 0 || static_cast<std::size_t>(entity_slot) >= entities_per_game) {
        return;
    }
    const auto entity_index = entity_offset + static_cast<std::size_t>(entity_slot);
    auto entity = entities[entity_index];
    if (entity.id.value != command.entity.value ||
        entity.owner.value != command.player.value ||
        !entity.alive) {
        return;
    }
    const auto source_local = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    const auto source_index = cell_offset + source_local;
    const auto cost_ratio = entity.is_inspired ? config.inspired_move_cost_ratio : config.move_cost_ratio;
    const auto required = cost_ratio == 0 ? 0 : cell_energy[source_index] / cost_ratio;
    if (entity.energy < required) {
        return;
    }
    const auto destination = move_location_device(entity.location, command.direction, width, height);
    const auto destination_index = cell_offset + static_cast<std::size_t>(destination.y * width + destination.x);
    entity.energy -= required;
    entity.location = destination;
    entities[entity_index] = entity;
    cell_entity[source_index].value = Entity::None.value;
    atomicAdd(&destination_counts[destination_index], 1U);
    atomicAdd(&game_moved[game_index], 1ULL);
}

__global__ void batched_collision_resolve_stage_kernel(EntityFrameEntry *entities,
                                                       std::size_t total_entities,
                                                       std::size_t entities_per_game,
                                                       energy_type *cell_energy,
                                                       const unsigned int *destination_counts,
                                                       std::size_t cells_per_game,
                                                       int width,
                                                       CollisionConfigDevice config,
                                                       unsigned long long *game_damaged,
                                                       unsigned long long *game_deaths,
                                                       unsigned long long *game_dropped) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }
    auto entity = entities[index];
    if (!entity.alive || entity.id.value == Entity::None.value) {
        return;
    }
    const auto game_index = index / entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    const auto cell = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    const auto count = destination_counts[cell_offset + cell];
    if (count <= 1U) {
        return;
    }
    atomicAdd(&game_damaged[game_index], 1ULL);
    entity.hp -= config.collision_hp_damage;
    if (entity.hp <= 0) {
        entity.alive = false;
        atomic_add_energy(&cell_energy[cell_offset + cell], entity.energy);
        atomicAdd(&game_deaths[game_index], 1ULL);
        atomicAdd(&game_dropped[game_index], static_cast<unsigned long long>(entity.energy));
        entity.energy = 0;
    }
    entities[index] = entity;
}

__global__ void summarize_batched_collision_cells_kernel(const unsigned int *destination_counts,
                                                         std::size_t total_cells,
                                                         std::size_t cells_per_game,
                                                         unsigned long long *game_collision_cells) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_cells) {
        return;
    }
    if (destination_counts[index] > 1U) {
        const auto game_index = index / cells_per_game;
        atomicAdd(&game_collision_cells[game_index], 1ULL);
    }
}

__global__ void batched_validation_decision_kernel(const FlatCommand *commands,
                                                   std::size_t total_commands,
                                                   std::size_t commands_per_game,
                                                   const PlayerFrameEntry *players,
                                                   std::size_t players_per_game,
                                                   const EntityFrameEntry *entities,
                                                   std::size_t entities_per_game,
                                                   const energy_type *cell_energy,
                                                   std::size_t cells_per_game,
                                                   int width,
                                                   ValidationConfigDevice config,
                                                   ValidationFrameDecision *decisions,
                                                   unsigned long long *game_included,
                                                   unsigned long long *game_expenses,
                                                   unsigned long long *game_occurrences,
                                                   unsigned long iterations) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_commands) {
        return;
    }
    const auto game_index = index / commands_per_game;
    const auto player_offset = game_index * players_per_game;
    const auto entity_offset = game_index * entities_per_game;
    const auto cell_offset = game_index * cells_per_game;
    auto decision = decisions[index];
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        const auto command = commands[index];
        decision.player = command.player;
        decision.entity = command.entity;
        decision.type = command.type;
        decision.expense = 0;
        decision.command = true;
        decision.player_exists = false;
        decision.ownership_ok = false;
        decision.combat_enabled = config.combat_enabled;
        decision.occurrence_command = false;
        decision.expense_command = false;
        decision.include_in_batch = false;

        int player_index = -1;
        for (std::size_t local = 0; local < players_per_game; ++local) {
            if (players[player_offset + local].id.value == command.player.value) {
                player_index = static_cast<int>(player_offset + local);
                break;
            }
        }
        if (player_index < 0) {
            continue;
        }
        decision.player_exists = true;
        const auto player = players[player_index];

        int entity_index = -1;
        if (command.type != FlatCommandType::Spawn) {
            for (std::size_t local = 0; local < entities_per_game; ++local) {
                const auto entity = entities[entity_offset + local];
                if (entity.id.value == command.entity.value &&
                    entity.owner.value == command.player.value &&
                    entity.alive) {
                    entity_index = static_cast<int>(entity_offset + local);
                    break;
                }
            }
        }

        if (command.type == FlatCommandType::Move) {
            decision.ownership_ok = entity_index >= 0;
            decision.occurrence_command = decision.ownership_ok;
            decision.include_in_batch = decision.ownership_ok;
            continue;
        }

        if (command.type == FlatCommandType::Construct) {
            decision.ownership_ok = entity_index >= 0;
            decision.occurrence_command = decision.ownership_ok;
            decision.expense_command = decision.ownership_ok;
            decision.include_in_batch = decision.ownership_ok;
            if (entity_index >= 0) {
                const auto entity = entities[entity_index];
                const auto cost = scaled_dropoff_cost_device(config.dropoff_cost, player.dropoff_count, config.dropoff_growth);
                const auto location_index = cell_offset + static_cast<std::size_t>(entity.location.y * width + entity.location.x);
                const auto credit = cell_energy[location_index] + entity.energy;
                decision.expense = credit >= cost ? 0 : cost - credit;
            }
            continue;
        }

        if (command.type == FlatCommandType::Spawn) {
            decision.ownership_ok = true;
            decision.expense_command = true;
            decision.include_in_batch = true;
            decision.expense = scaled_spawn_cost_device(config.spawn_cost,
                                                        player.ship_count,
                                                        config.spawn_growth,
                                                        config.spawn_quad_threshold,
                                                        config.spawn_quad_growth);
            continue;
        }

        if (command.type == FlatCommandType::AttackShip || command.type == FlatCommandType::AttackStructure ||
            command.type == FlatCommandType::Defend || command.type == FlatCommandType::Heal) {
            decision.ownership_ok = entity_index >= 0;
            decision.occurrence_command = config.combat_enabled && decision.ownership_ok;
            decision.include_in_batch = decision.occurrence_command;
            continue;
        }
    }
    decisions[index] = decision;
    if (decision.include_in_batch) {
        atomicAdd(&game_included[game_index], 1ULL);
    }
    if (decision.expense_command) {
        atomicAdd(&game_expenses[game_index], 1ULL);
    }
    if (decision.occurrence_command) {
        atomicAdd(&game_occurrences[game_index], 1ULL);
    }
}

__device__ energy_type max_energy_type_device(energy_type a, energy_type b) {
    return a > b ? a : b;
}

__device__ energy_type min_energy_type_device(energy_type a, energy_type b) {
    return a < b ? a : b;
}

__global__ void batched_over_ship_tax_kernel(PlayerFrameEntry *players,
                                             std::size_t total_players,
                                             OverShipTaxConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_players) {
        return;
    }
    auto player = players[index];
    const auto n = player.ship_count;

    if (config.income_on && n > 0 && !player.factory_destroyed) {
        const energy_type income = static_cast<energy_type>(n) * config.ship_income_per_turn;
        player.factory_halite += income;
    }

    if (config.tax_on && n > config.over_ship_tax_threshold) {
        const auto excess = n - config.over_ship_tax_threshold;
        const energy_type tax = static_cast<energy_type>(excess) * config.over_ship_tax_per_turn;
        player.energy = max_energy_type_device(0, player.energy - tax);
        if (!player.factory_destroyed) {
            player.factory_halite = max_energy_type_device(1, player.factory_halite - tax);
        }
    }

    if (config.quad_on) {
        const long long target = static_cast<long long>(config.ship_count_target);
        const long long actual = static_cast<long long>(n);
        if (actual > target) {
            const long long excess = actual - target;
            const long long penalty_ll = static_cast<long long>(config.ship_count_deviation_penalty) * excess * excess;
            const energy_type penalty = static_cast<energy_type>(penalty_ll);
            if (penalty > 0) {
                player.energy = max_energy_type_device(0, player.energy - penalty);
                if (!player.factory_destroyed) {
                    player.factory_halite = max_energy_type_device(1, player.factory_halite - penalty);
                }
            }
        }
    }

    players[index] = player;
}

__global__ void batched_halite_rebalance_kernel(PlayerFrameEntry *players,
                                                std::size_t total_games,
                                                std::size_t players_per_game,
                                                DropoffFrameEntry *dropoffs,
                                                std::size_t dropoffs_per_game,
                                                const unsigned long *turn_numbers,
                                                HaliteRebalanceConfigDevice config) {
    const auto game_index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (game_index >= total_games) {
        return;
    }
    if (!config.enabled || config.period == 0 || config.fraction <= 0.0) {
        return;
    }
    const auto turn_number = turn_numbers[game_index];
    if (turn_number == 0 || turn_number % config.period != 0 || players_per_game == 0) {
        return;
    }

    const auto player_offset = game_index * players_per_game;
    const auto dropoff_offset = game_index * dropoffs_per_game;
    std::size_t leader_index = 0;
    std::size_t trailer_index = 0;
    energy_type max_score = 0;
    energy_type min_score = 0;
    bool seen_any = false;

    for (std::size_t local_player = 0; local_player < players_per_game; ++local_player) {
        const auto player = players[player_offset + local_player];
        if (player.id.value == Player::None.value) {
            continue;
        }
        energy_type score = player.factory_halite;
        const auto dropoff_end = player.dropoff_begin + player.dropoff_count;
        for (std::size_t dropoff_index = player.dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
            score += dropoffs[dropoff_offset + dropoff_index].halite_pool;
        }

        if (!seen_any || score > max_score) {
            max_score = score;
            leader_index = local_player;
        }
        if (!seen_any || score < min_score) {
            min_score = score;
            trailer_index = local_player;
        }
        seen_any = true;
    }

    if (!seen_any || leader_index == trailer_index || max_score <= min_score) {
        return;
    }

    if (config.min_gap_frac > 0.0 && max_score + min_score > 0) {
        const double gap_frac = static_cast<double>(max_score - min_score) /
                                static_cast<double>(max_score + min_score);
        if (gap_frac < config.min_gap_frac) {
            return;
        }
    }

    const auto gap = max_score - min_score;
    const auto transfer = static_cast<energy_type>(static_cast<double>(gap) * config.fraction * 0.5);
    if (transfer <= 0) {
        return;
    }

    auto leader = players[player_offset + leader_index];
    auto trailer = players[player_offset + trailer_index];
    energy_type remaining = transfer;

    if (!leader.factory_destroyed) {
        const auto drained = min_energy_type_device(leader.factory_halite, remaining);
        leader.factory_halite -= drained;
        remaining -= drained;
    }

    const auto leader_dropoff_end = leader.dropoff_begin + leader.dropoff_count;
    for (std::size_t dropoff_index = leader.dropoff_begin; dropoff_index < leader_dropoff_end && remaining > 0; ++dropoff_index) {
        auto dropoff = dropoffs[dropoff_offset + dropoff_index];
        if (dropoff.destroyed) {
            continue;
        }
        const auto drained = min_energy_type_device(dropoff.halite_pool, remaining);
        dropoff.halite_pool -= drained;
        remaining -= drained;
        dropoffs[dropoff_offset + dropoff_index] = dropoff;
    }

    const auto actually_drained = transfer - remaining;
    if (actually_drained <= 0) {
        return;
    }

    if (!trailer.factory_destroyed) {
        trailer.factory_halite += actually_drained;
    } else {
        const auto trailer_dropoff_end = trailer.dropoff_begin + trailer.dropoff_count;
        for (std::size_t dropoff_index = trailer.dropoff_begin; dropoff_index < trailer_dropoff_end; ++dropoff_index) {
            auto dropoff = dropoffs[dropoff_offset + dropoff_index];
            if (dropoff.destroyed) {
                continue;
            }
            dropoff.halite_pool += actually_drained;
            dropoffs[dropoff_offset + dropoff_index] = dropoff;
            break;
        }
    }

    leader.energy = max_energy_type_device(static_cast<energy_type>(0), leader.energy - actually_drained);
    trailer.energy += actually_drained;

    players[player_offset + leader_index] = leader;
    players[player_offset + trailer_index] = trailer;
}

__global__ void refill_mining_inputs_kernel(EntityFrameEntry *entities,
                                            std::size_t total_entities,
                                            std::size_t entities_per_game,
                                            energy_type *cell_energy,
                                            const Player::id_type *cell_owner,
                                            std::size_t cells_per_game,
                                            int width,
                                            energy_type refill_cell_energy) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= total_entities) {
        return;
    }
    auto entity = entities[index];
    if (!entity.alive) {
        return;
    }
    const auto game_index = index / entities_per_game;
    const auto cell_index = game_index * cells_per_game + static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    if (refill_cell_energy <= 0 || cell_owner[cell_index].value != Player::None.value) {
        return;
    }
    entity.energy = 0;
    entities[index] = entity;
    cell_energy[cell_index] = refill_cell_energy;
}

} // namespace

BatchedRegenStats run_cuda_batched_regen_iterations(std::vector<StateFrame *> frames,
                                                    const GameConfig &config,
                                                    unsigned long iterations) {
    BatchedRegenStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.cells_per_game = cells_per_game;
    if (cells_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_regen received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->cell_initial_energy.size() != cells_per_game ||
            frame->cell_owner.size() != cells_per_game) {
            throw std::invalid_argument("all frames must have the same map cell count");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    std::vector<energy_type> host_energy(total_cells);
    std::vector<energy_type> host_initial(total_cells);
    std::vector<Player::id_type> host_owner(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        const auto offset = game * cells_per_game;
        std::copy(frames[game]->cell_energy.begin(), frames[game]->cell_energy.end(), host_energy.begin() + offset);
        std::copy(frames[game]->cell_initial_energy.begin(), frames[game]->cell_initial_energy.end(), host_initial.begin() + offset);
        std::copy(frames[game]->cell_owner.begin(), frames[game]->cell_owner.end(), host_owner.begin() + offset);
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<energy_type> device_initial(total_cells);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<unsigned long long> device_deltas(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy batched energy H2D");
    check_cuda(cudaMemcpy(device_initial.data(), host_initial.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy batched initial H2D");
    check_cuda(cudaMemcpy(device_owner.data(), host_owner.data(), total_cells * sizeof(Player::id_type), cudaMemcpyHostToDevice), "cudaMemcpy batched owner H2D");
    check_cuda(cudaMemset(device_deltas.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset batched deltas");

    RegenConfigDevice device_config;
    device_config.enabled = config.ruleset.economy.cell_regen_enabled;
    device_config.rate = config.ruleset.economy.cell_regen_rate;
    device_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    batched_regen_kernel<<<blocks, threads_per_block>>>(device_energy.data(),
                                                        device_initial.data(),
                                                        device_owner.data(),
                                                        device_deltas.data(),
                                                        total_cells,
                                                        cells_per_game,
                                                        device_config,
                                                        iterations);
    check_cuda(cudaGetLastError(), "batched_regen_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_regen_kernel sync");

    std::vector<unsigned long long> host_deltas(frames.size());
    check_cuda(cudaMemcpy(host_energy.data(), device_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyDeviceToHost), "cudaMemcpy batched energy D2H");
    check_cuda(cudaMemcpy(host_deltas.data(), device_deltas.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy batched deltas D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        const auto offset = game * cells_per_game;
        std::copy(host_energy.begin() + offset, host_energy.begin() + offset + cells_per_game, frames[game]->cell_energy.begin());
        frames[game]->map_total_energy += host_deltas[game];
        stats.total_delta += host_deltas[game];
    }
    return stats;
}

BatchedRegenStats run_cuda_batched_regen(std::vector<StateFrame *> frames, const GameConfig &config) {
    return run_cuda_batched_regen_iterations(std::move(frames), config, 1);
}

BatchedDumpStats run_cuda_batched_dump(std::vector<StateFrame *> frames,
                                       const GameConfig &config,
                                       unsigned long iterations,
                                       energy_type refill_cargo) {
    BatchedDumpStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto entities_per_game = frames.front()->entities.size();
    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    const auto cells_per_game = frames.front()->cell_owner.size();
    stats.entities_per_game = entities_per_game;
    if (entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_dump received a null frame");
        }
        if (frame->entities.size() != entities_per_game ||
            frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game ||
            frame->cell_owner.size() != cells_per_game) {
            throw std::invalid_argument("all frames must have the same entity/player/dropoff/cell counts");
        }
    }

    const auto total_entities = entities_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    std::vector<Player::id_type> host_owner(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(),
                      frames[game]->dropoffs.end(),
                      host_dropoffs.begin() + game * dropoffs_per_game);
        }
        std::copy(frames[game]->cell_owner.begin(),
                  frames[game]->cell_owner.end(),
                  host_owner.begin() + game * cells_per_game);
    }

    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<unsigned long long> device_deposits(frames.size());
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched dump entities H2D");
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched dump players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(),
                              host_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy batched dump dropoffs H2D");
    }
    check_cuda(cudaMemcpy(device_owner.data(),
                          host_owner.data(),
                          total_cells * sizeof(Player::id_type),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched dump owners H2D");
    check_cuda(cudaMemset(device_deposits.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset dump deposits");

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    batched_dump_kernel<<<blocks, threads_per_block>>>(device_entities.data(),
                                                       total_entities,
                                                       entities_per_game,
                                                       device_players.data(),
                                                       players_per_game,
                                                       device_dropoffs.data(),
                                                       dropoffs_per_game,
                                                       device_owner.data(),
                                                       cells_per_game,
                                                       device_deposits.data(),
                                                       frames.front()->width,
                                                       config.ruleset.combat.initial_hp,
                                                       iterations,
                                                       refill_cargo,
                                                       false);
    check_cuda(cudaGetLastError(), "batched_dump_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_dump_kernel sync");

    std::vector<unsigned long long> host_deposits(frames.size());
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched dump entities D2H");
    check_cuda(cudaMemcpy(host_players.data(),
                          device_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched dump players D2H");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(host_dropoffs.data(),
                              device_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy batched dump dropoffs D2H");
    }
    check_cuda(cudaMemcpy(host_deposits.data(),
                          device_deposits.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched dump deposits D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        std::copy(host_players.begin() + game * players_per_game,
                  host_players.begin() + (game + 1) * players_per_game,
                  frames[game]->players.begin());
        if (dropoffs_per_game > 0) {
            std::copy(host_dropoffs.begin() + game * dropoffs_per_game,
                      host_dropoffs.begin() + (game + 1) * dropoffs_per_game,
                      frames[game]->dropoffs.begin());
        }
        stats.total_deposited += host_deposits[game];
    }
    return stats;
}

BatchedInspirationStats run_cuda_batched_inspiration_resident(std::vector<StateFrame *> frames,
                                                              const GameConfig &config,
                                                              unsigned long turns) {
    BatchedInspirationStats stats;
    stats.games = frames.size();
    if (frames.empty() || turns == 0) {
        return stats;
    }
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_entity.size();
    stats.entities_per_game = entities_per_game;
    if (entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_inspiration_resident received a null frame");
        }
        if (frame->entities.size() != entities_per_game ||
            frame->cell_entity.size() != cells_per_game ||
            frame->width != frames.front()->width ||
            frame->height != frames.front()->height) {
            throw std::invalid_argument("all frames must have the same inspiration batch layout");
        }
    }

    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
    }

    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<unsigned long long> device_inspired(frames.size());
    check_cuda(cudaMemcpy(device_entities.data(), host_entities.data(), total_entities * sizeof(EntityFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy resident inspiration entities H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(), host_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyHostToDevice), "cudaMemcpy resident inspiration cell entity H2D");
    check_cuda(cudaMemset(device_inspired.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset resident inspiration inspired");

    InspirationConfigDevice inspiration_config;
    inspiration_config.enabled = config.ruleset.inspiration.enabled;
    inspiration_config.radius = config.ruleset.inspiration.radius;
    inspiration_config.ship_count_threshold = config.ruleset.inspiration.ship_count;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    for (unsigned long turn = 0; turn < turns; ++turn) {
        batched_inspiration_kernel<<<blocks, threads_per_block>>>(device_entities.data(),
                                                                  total_entities,
                                                                  entities_per_game,
                                                                  device_cell_entity.data(),
                                                                  cells_per_game,
                                                                  frames.front()->width,
                                                                  frames.front()->height,
                                                                  inspiration_config);
        check_cuda(cudaGetLastError(), "resident batched_inspiration_kernel launch");
    }
    summarize_batched_inspiration_kernel<<<blocks, threads_per_block>>>(device_entities.data(),
                                                                        total_entities,
                                                                        entities_per_game,
                                                                        device_inspired.data());
    check_cuda(cudaGetLastError(), "summarize resident batched_inspiration_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "resident inspiration sync");

    std::vector<unsigned long long> host_inspired(frames.size());
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy resident inspiration entities D2H");
    check_cuda(cudaMemcpy(host_inspired.data(), device_inspired.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy resident inspiration inspired D2H");
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        stats.inspired_entities += host_inspired[game];
    }
    return stats;
}

BatchedMiningStats run_cuda_batched_mining(std::vector<StateFrame *> frames,
                                           const GameConfig &config,
                                           unsigned long iterations) {
    BatchedMiningStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    const auto entities_per_game = frames.front()->entities.size();
    stats.cells_per_game = cells_per_game;
    stats.entities_per_game = entities_per_game;
    if (cells_per_game == 0 || entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_mining received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->entities.size() != entities_per_game) {
            throw std::invalid_argument("all frames must have the same mining batch layout");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    std::vector<energy_type> host_energy(total_cells);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<unsigned long long> device_extracted(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy batched mining energy H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched mining entities H2D");
    check_cuda(cudaMemset(device_extracted.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset batched mining extracted");

    MiningConfigDevice mining_config;
    mining_config.max_energy = config.ruleset.economy.max_energy;
    mining_config.extract_ratio = config.ruleset.economy.extract_ratio;
    mining_config.inspired_extract_ratio = config.ruleset.inspiration.extract_ratio;
    mining_config.inspired_bonus_multiplier = config.ruleset.inspiration.bonus_multiplier;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    batched_mining_kernel<<<blocks, threads_per_block>>>(device_entities.data(),
                                                         total_entities,
                                                         entities_per_game,
                                                         device_energy.data(),
                                                         cells_per_game,
                                                         device_extracted.data(),
                                                         frames.front()->width,
                                                         mining_config,
                                                         iterations);
    check_cuda(cudaGetLastError(), "batched_mining_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_mining_kernel sync");

    std::vector<unsigned long long> host_extracted(frames.size());
    check_cuda(cudaMemcpy(host_energy.data(), device_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyDeviceToHost), "cudaMemcpy batched mining energy D2H");
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched mining entities D2H");
    check_cuda(cudaMemcpy(host_extracted.data(),
                          device_extracted.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched mining extracted D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_energy.begin() + game * cells_per_game,
                  host_energy.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_energy.begin());
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        frames[game]->map_total_energy -= host_extracted[game];
        stats.total_extracted += host_extracted[game];
    }
    return stats;
}

BatchedSpawnStats run_cuda_batched_spawn_apply(std::vector<StateFrame *> frames,
                                               const GameConfig &config,
                                               unsigned long iterations) {
    BatchedSpawnStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto players_per_game = frames.front()->players.size();
    const auto entity_capacity_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_entity.size();
    stats.players_per_game = players_per_game;
    stats.entity_capacity_per_game = entity_capacity_per_game;
    if (players_per_game == 0 || entity_capacity_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_spawn_apply received a null frame");
        }
        if (frame->players.size() != players_per_game ||
            frame->entities.size() != entity_capacity_per_game ||
            frame->cell_entity.size() != cells_per_game) {
            throw std::invalid_argument("all frames must have the same spawn batch layout");
        }
    }

    const auto total_players = players_per_game * frames.size();
    const auto total_entities = entity_capacity_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entity_capacity_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
    }

    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<unsigned long long> device_spawned(frames.size());
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched spawn players H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched spawn entities H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(),
                          host_cell_entity.data(),
                          total_cells * sizeof(Entity::id_type),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched spawn cell entity H2D");
    check_cuda(cudaMemset(device_spawned.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset batched spawn spawned");

    SpawnApplyConfigDevice spawn_config;
    spawn_config.base_cost = config.ruleset.economy.new_entity_energy_cost;
    spawn_config.growth = config.ruleset.economy.spawn_cost_growth;
    spawn_config.quad_threshold = config.ruleset.economy.spawn_quad_threshold;
    spawn_config.quad_growth = config.ruleset.economy.spawn_quad_growth;
    spawn_config.initial_hp = config.ruleset.combat.initial_hp;
    spawn_config.iterations = iterations;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_players + threads_per_block - 1) / threads_per_block);
    batched_spawn_apply_kernel<<<blocks, threads_per_block>>>(device_players.data(),
                                                              total_players,
                                                              players_per_game,
                                                              device_entities.data(),
                                                              entity_capacity_per_game,
                                                              device_cell_entity.data(),
                                                              cells_per_game,
                                                              device_spawned.data(),
                                                              frames.front()->width,
                                                              static_cast<int>(frames.front()->turn_number),
                                                              spawn_config);
    check_cuda(cudaGetLastError(), "batched_spawn_apply_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_spawn_apply_kernel sync");

    std::vector<unsigned long long> host_spawned(frames.size());
    check_cuda(cudaMemcpy(host_players.data(),
                          device_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched spawn players D2H");
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched spawn entities D2H");
    check_cuda(cudaMemcpy(host_cell_entity.data(),
                          device_cell_entity.data(),
                          total_cells * sizeof(Entity::id_type),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched spawn cell entity D2H");
    check_cuda(cudaMemcpy(host_spawned.data(),
                          device_spawned.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched spawn spawned D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_players.begin() + game * players_per_game,
                  host_players.begin() + (game + 1) * players_per_game,
                  frames[game]->players.begin());
        std::copy(host_entities.begin() + game * entity_capacity_per_game,
                  host_entities.begin() + (game + 1) * entity_capacity_per_game,
                  frames[game]->entities.begin());
        std::copy(host_cell_entity.begin() + game * cells_per_game,
                  host_cell_entity.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_entity.begin());
        stats.total_spawned += host_spawned[game];
    }
    return stats;
}

BatchedMovementStats run_cuda_batched_movement_decisions(std::vector<StateFrame *> frames,
                                                         const GameConfig &config,
                                                         const std::vector<CommandFrame> &commands,
                                                         std::vector<MovementFrameDecision> &decisions_out,
                                                         unsigned long iterations) {
    BatchedMovementStats stats;
    stats.games = frames.size();
    decisions_out.clear();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    if (commands.size() != frames.size()) {
        throw std::invalid_argument("movement command frame count must match frames");
    }
    const auto commands_per_game = commands.front().commands.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.commands_per_game = commands_per_game;
    if (commands_per_game == 0 || entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        if (frames[game] == nullptr) {
            throw std::invalid_argument("run_cuda_batched_movement_decisions received a null frame");
        }
        if (commands[game].commands.size() != commands_per_game ||
            frames[game]->entities.size() != entities_per_game ||
            frames[game]->cell_energy.size() != cells_per_game) {
            throw std::invalid_argument("all movement frames must have the same batch layout");
        }
    }

    const auto total_commands = commands_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<FlatCommand> host_commands(total_commands);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<energy_type> host_energy(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(commands[game].commands.begin(),
                  commands[game].commands.end(),
                  host_commands.begin() + game * commands_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
    }
    decisions_out.assign(total_commands,
                         MovementFrameDecision{Player::None,
                                               Entity::None,
                                               Location{0, 0},
                                               Location{0, 0},
                                               0,
                                               0,
                                               false,
                                               false,
                                               false});

    DeviceVector<FlatCommand> device_commands(total_commands);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<MovementFrameDecision> device_decisions(total_commands);
    DeviceVector<unsigned long long> device_movable(frames.size());
    check_cuda(cudaMemcpy(device_commands.data(),
                          host_commands.data(),
                          total_commands * sizeof(FlatCommand),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched movement commands H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched movement entities H2D");
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy batched movement energy H2D");
    check_cuda(cudaMemcpy(device_decisions.data(),
                          decisions_out.data(),
                          total_commands * sizeof(MovementFrameDecision),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched movement decisions H2D");
    check_cuda(cudaMemset(device_movable.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset batched movement movable");

    MovementConfigDevice movement_config;
    movement_config.move_cost_ratio = config.ruleset.economy.move_cost_ratio;
    movement_config.inspired_move_cost_ratio = config.ruleset.inspiration.move_cost_ratio;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_commands + threads_per_block - 1) / threads_per_block);
    batched_movement_decision_kernel<<<blocks, threads_per_block>>>(device_commands.data(),
                                                                    total_commands,
                                                                    commands_per_game,
                                                                    device_entities.data(),
                                                                    entities_per_game,
                                                                    device_energy.data(),
                                                                    cells_per_game,
                                                                    frames.front()->width,
                                                                    frames.front()->height,
                                                                    movement_config,
                                                                    device_decisions.data(),
                                                                    device_movable.data(),
                                                                    iterations);
    check_cuda(cudaGetLastError(), "batched_movement_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_movement_decision_kernel sync");

    std::vector<unsigned long long> host_movable(frames.size());
    check_cuda(cudaMemcpy(decisions_out.data(),
                          device_decisions.data(),
                          total_commands * sizeof(MovementFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched movement decisions D2H");
    check_cuda(cudaMemcpy(host_movable.data(),
                          device_movable.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched movement movable D2H");
    for (auto count : host_movable) {
        stats.movable_commands += count;
    }
    return stats;
}

BatchedMovementApplyStats run_cuda_batched_movement_apply_no_collision(std::vector<StateFrame *> frames,
                                                                       const GameConfig &config,
                                                                       const std::vector<CommandFrame> &commands,
                                                                       unsigned long iterations) {
    BatchedMovementApplyStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    if (commands.size() != frames.size()) {
        throw std::invalid_argument("movement apply command frame count must match frames");
    }
    const auto commands_per_game = commands.front().commands.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.commands_per_game = commands_per_game;
    if (commands_per_game == 0 || entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        if (frames[game] == nullptr) {
            throw std::invalid_argument("run_cuda_batched_movement_apply_no_collision received a null frame");
        }
        if (commands[game].commands.size() != commands_per_game ||
            frames[game]->entities.size() != entities_per_game ||
            frames[game]->cell_energy.size() != cells_per_game ||
            frames[game]->cell_entity.size() != cells_per_game) {
            throw std::invalid_argument("all movement apply frames must have the same batch layout");
        }
    }

    const auto total_commands = commands_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<FlatCommand> host_commands(total_commands);
    std::vector<int> host_command_entity_slots(total_commands, -1);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<energy_type> host_energy(total_cells);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(commands[game].commands.begin(),
                  commands[game].commands.end(),
                  host_commands.begin() + game * commands_per_game);
        for (std::size_t command = 0; command < commands_per_game; ++command) {
            const auto &flat = commands[game].commands[command];
            const auto slot = frames[game]->entity_slot_by_id.find(flat.entity);
            if (slot != frames[game]->entity_slot_by_id.end() &&
                slot->second < entities_per_game) {
                host_command_entity_slots[game * commands_per_game + command] = static_cast<int>(slot->second);
            }
        }
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
    }

    DeviceVector<FlatCommand> device_commands(total_commands);
    DeviceVector<int> device_command_entity_slots(total_commands);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<unsigned long long> device_applied(frames.size());
    check_cuda(cudaMemcpy(device_commands.data(),
                          host_commands.data(),
                          total_commands * sizeof(FlatCommand),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy movement apply commands H2D");
    check_cuda(cudaMemcpy(device_command_entity_slots.data(),
                          host_command_entity_slots.data(),
                          total_commands * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy movement apply command slots H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy movement apply entities H2D");
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy movement apply energy H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(),
                          host_cell_entity.data(),
                          total_cells * sizeof(Entity::id_type),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy movement apply cell entity H2D");
    check_cuda(cudaMemset(device_applied.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset movement apply applied");

    MovementConfigDevice movement_config;
    movement_config.move_cost_ratio = config.ruleset.economy.move_cost_ratio;
    movement_config.inspired_move_cost_ratio = config.ruleset.inspiration.move_cost_ratio;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_commands + threads_per_block - 1) / threads_per_block);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        batched_movement_apply_no_collision_kernel<<<blocks, threads_per_block>>>(device_commands.data(),
                                                                                  device_command_entity_slots.data(),
                                                                                  total_commands,
                                                                                  commands_per_game,
                                                                                  device_entities.data(),
                                                                                  entities_per_game,
                                                                                  device_energy.data(),
                                                                                  device_cell_entity.data(),
                                                                                  cells_per_game,
                                                                                  frames.front()->width,
                                                                                  frames.front()->height,
                                                                                  movement_config,
                                                                                  device_applied.data(),
                                                                                  1);
        check_cuda(cudaGetLastError(), "batched_movement_apply_no_collision_kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "batched_movement_apply_no_collision_kernel sync");

    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    clear_batched_cell_entities_kernel<<<cell_blocks, threads_per_block>>>(device_cell_entity.data(), total_cells);
    check_cuda(cudaGetLastError(), "clear_batched_cell_entities_kernel launch");
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    scatter_batched_entities_to_cells_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                                   total_entities,
                                                                                   entities_per_game,
                                                                                   device_cell_entity.data(),
                                                                                   cells_per_game,
                                                                                   frames.front()->width);
    check_cuda(cudaGetLastError(), "scatter_batched_entities_to_cells_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "movement apply occupancy rebuild sync");

    std::vector<unsigned long long> host_applied(frames.size());
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy movement apply entities D2H");
    check_cuda(cudaMemcpy(host_cell_entity.data(),
                          device_cell_entity.data(),
                          total_cells * sizeof(Entity::id_type),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy movement apply cell entity D2H");
    check_cuda(cudaMemcpy(host_applied.data(),
                          device_applied.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy movement apply applied D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        std::copy(host_cell_entity.begin() + game * cells_per_game,
                  host_cell_entity.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_entity.begin());
        stats.applied_commands += host_applied[game];
    }
    return stats;
}

BatchedDestinationStats run_cuda_batched_destination_counts(std::vector<StateFrame *> frames,
                                                           const GameConfig &config,
                                                           const std::vector<CommandFrame> &commands,
                                                           unsigned long iterations) {
    BatchedDestinationStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    if (commands.size() != frames.size()) {
        throw std::invalid_argument("destination command frame count must match frames");
    }
    const auto commands_per_game = commands.front().commands.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.commands_per_game = commands_per_game;
    if (commands_per_game == 0 || entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        if (frames[game] == nullptr) {
            throw std::invalid_argument("run_cuda_batched_destination_counts received a null frame");
        }
        if (commands[game].commands.size() != commands_per_game ||
            frames[game]->entities.size() != entities_per_game ||
            frames[game]->cell_energy.size() != cells_per_game) {
            throw std::invalid_argument("all destination frames must have the same batch layout");
        }
    }

    const auto total_commands = commands_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<FlatCommand> host_commands(total_commands);
    std::vector<int> host_command_entity_slots(total_commands, -1);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<energy_type> host_energy(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(commands[game].commands.begin(),
                  commands[game].commands.end(),
                  host_commands.begin() + game * commands_per_game);
        for (std::size_t command = 0; command < commands_per_game; ++command) {
            const auto &flat = commands[game].commands[command];
            const auto slot = frames[game]->entity_slot_by_id.find(flat.entity);
            if (slot != frames[game]->entity_slot_by_id.end() &&
                slot->second < entities_per_game) {
                host_command_entity_slots[game * commands_per_game + command] = static_cast<int>(slot->second);
            }
        }
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
    }

    DeviceVector<FlatCommand> device_commands(total_commands);
    DeviceVector<int> device_command_entity_slots(total_commands);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<unsigned int> device_destination_counts(total_cells);
    DeviceVector<unsigned long long> device_moved(frames.size());
    DeviceVector<unsigned long long> device_conflict_cells(frames.size());
    DeviceVector<unsigned long long> device_max_arrivals(frames.size());
    DeviceVector<unsigned long long> device_total_arrivals(frames.size());
    check_cuda(cudaMemcpy(device_commands.data(),
                          host_commands.data(),
                          total_commands * sizeof(FlatCommand),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy destination commands H2D");
    check_cuda(cudaMemcpy(device_command_entity_slots.data(),
                          host_command_entity_slots.data(),
                          total_commands * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy destination command slots H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy destination entities H2D");
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy destination energy H2D");
    check_cuda(cudaMemset(device_destination_counts.data(), 0, total_cells * sizeof(unsigned int)), "cudaMemset destination counts");
    check_cuda(cudaMemset(device_moved.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset destination moved");
    check_cuda(cudaMemset(device_conflict_cells.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset destination conflicts");
    check_cuda(cudaMemset(device_max_arrivals.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset destination max");
    check_cuda(cudaMemset(device_total_arrivals.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset destination arrivals");

    MovementConfigDevice movement_config;
    movement_config.move_cost_ratio = config.ruleset.economy.move_cost_ratio;
    movement_config.inspired_move_cost_ratio = config.ruleset.inspiration.move_cost_ratio;

    constexpr int threads_per_block = 256;
    const auto command_blocks = static_cast<int>((total_commands + threads_per_block - 1) / threads_per_block);
    batched_destination_count_kernel<<<command_blocks, threads_per_block>>>(device_commands.data(),
                                                                            device_command_entity_slots.data(),
                                                                            total_commands,
                                                                            commands_per_game,
                                                                            device_entities.data(),
                                                                            entities_per_game,
                                                                            device_energy.data(),
                                                                            cells_per_game,
                                                                            frames.front()->width,
                                                                            frames.front()->height,
                                                                            movement_config,
                                                                            device_destination_counts.data(),
                                                                            device_moved.data(),
                                                                            iterations);
    check_cuda(cudaGetLastError(), "batched_destination_count_kernel launch");
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    summarize_batched_destination_counts_kernel<<<cell_blocks, threads_per_block>>>(device_destination_counts.data(),
                                                                                   total_cells,
                                                                                   cells_per_game,
                                                                                   device_conflict_cells.data(),
                                                                                   device_max_arrivals.data(),
                                                                                   device_total_arrivals.data());
    check_cuda(cudaGetLastError(), "summarize_batched_destination_counts_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "destination counts sync");

    std::vector<unsigned long long> host_moved(frames.size());
    std::vector<unsigned long long> host_conflict_cells(frames.size());
    std::vector<unsigned long long> host_max_arrivals(frames.size());
    std::vector<unsigned long long> host_total_arrivals(frames.size());
    check_cuda(cudaMemcpy(host_moved.data(),
                          device_moved.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy destination moved D2H");
    check_cuda(cudaMemcpy(host_conflict_cells.data(),
                          device_conflict_cells.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy destination conflicts D2H");
    check_cuda(cudaMemcpy(host_max_arrivals.data(),
                          device_max_arrivals.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy destination max D2H");
    check_cuda(cudaMemcpy(host_total_arrivals.data(),
                          device_total_arrivals.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy destination arrivals D2H");
    for (std::size_t game = 0; game < frames.size(); ++game) {
        stats.moved_commands += host_moved[game];
        stats.conflict_cells += host_conflict_cells[game];
        stats.total_arrivals += host_total_arrivals[game];
        stats.max_arrivals = std::max(stats.max_arrivals, host_max_arrivals[game]);
    }
    return stats;
}

BatchedMovementCollisionStats run_cuda_batched_movement_collision_apply(std::vector<StateFrame *> frames,
                                                                        const GameConfig &config,
                                                                        const std::vector<CommandFrame> &commands,
                                                                        unsigned long iterations) {
    BatchedMovementCollisionStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    if (commands.size() != frames.size()) {
        throw std::invalid_argument("collision command frame count must match frames");
    }
    const auto commands_per_game = commands.front().commands.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.commands_per_game = commands_per_game;
    if (commands_per_game == 0 || entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        if (frames[game] == nullptr) {
            throw std::invalid_argument("run_cuda_batched_movement_collision_apply received a null frame");
        }
        if (commands[game].commands.size() != commands_per_game ||
            frames[game]->entities.size() != entities_per_game ||
            frames[game]->cell_energy.size() != cells_per_game ||
            frames[game]->cell_entity.size() != cells_per_game) {
            throw std::invalid_argument("all collision frames must have the same batch layout");
        }
    }

    const auto total_commands = commands_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<FlatCommand> host_commands(total_commands);
    std::vector<int> host_command_entity_slots(total_commands, -1);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<energy_type> host_energy(total_cells);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(commands[game].commands.begin(),
                  commands[game].commands.end(),
                  host_commands.begin() + game * commands_per_game);
        for (std::size_t command = 0; command < commands_per_game; ++command) {
            const auto &flat = commands[game].commands[command];
            const auto slot = frames[game]->entity_slot_by_id.find(flat.entity);
            if (slot != frames[game]->entity_slot_by_id.end() &&
                slot->second < entities_per_game) {
                host_command_entity_slots[game * commands_per_game + command] = static_cast<int>(slot->second);
            }
        }
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
    }

    DeviceVector<FlatCommand> device_commands(total_commands);
    DeviceVector<int> device_command_entity_slots(total_commands);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<unsigned int> device_destination_counts(total_cells);
    DeviceVector<unsigned long long> device_moved(frames.size());
    DeviceVector<unsigned long long> device_collision_cells(frames.size());
    DeviceVector<unsigned long long> device_damaged(frames.size());
    DeviceVector<unsigned long long> device_deaths(frames.size());
    DeviceVector<unsigned long long> device_dropped(frames.size());
    check_cuda(cudaMemcpy(device_commands.data(), host_commands.data(), total_commands * sizeof(FlatCommand), cudaMemcpyHostToDevice), "cudaMemcpy collision commands H2D");
    check_cuda(cudaMemcpy(device_command_entity_slots.data(), host_command_entity_slots.data(), total_commands * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy collision command slots H2D");
    check_cuda(cudaMemcpy(device_entities.data(), host_entities.data(), total_entities * sizeof(EntityFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy collision entities H2D");
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy collision energy H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(), host_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyHostToDevice), "cudaMemcpy collision cell entity H2D");
    check_cuda(cudaMemset(device_destination_counts.data(), 0, total_cells * sizeof(unsigned int)), "cudaMemset collision destination counts");
    check_cuda(cudaMemset(device_moved.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset collision moved");
    check_cuda(cudaMemset(device_collision_cells.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset collision cells");
    check_cuda(cudaMemset(device_damaged.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset collision damaged");
    check_cuda(cudaMemset(device_deaths.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset collision deaths");
    check_cuda(cudaMemset(device_dropped.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset collision dropped");

    CollisionConfigDevice collision_config;
    collision_config.move_cost_ratio = config.ruleset.economy.move_cost_ratio;
    collision_config.inspired_move_cost_ratio = config.ruleset.inspiration.move_cost_ratio;
    collision_config.collision_hp_damage = config.ruleset.combat.collision_hp_damage;

    constexpr int threads_per_block = 256;
    const auto command_blocks = static_cast<int>((total_commands + threads_per_block - 1) / threads_per_block);
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        check_cuda(cudaMemset(device_destination_counts.data(), 0, total_cells * sizeof(unsigned int)), "cudaMemset collision destination counts iteration");
        batched_collision_move_stage_kernel<<<command_blocks, threads_per_block>>>(device_commands.data(),
                                                                                   device_command_entity_slots.data(),
                                                                                   total_commands,
                                                                                   commands_per_game,
                                                                                   device_entities.data(),
                                                                                   entities_per_game,
                                                                                   device_energy.data(),
                                                                                   device_cell_entity.data(),
                                                                                   cells_per_game,
                                                                                   frames.front()->width,
                                                                                   frames.front()->height,
                                                                                   collision_config,
                                                                                   device_destination_counts.data(),
                                                                                   device_moved.data());
        check_cuda(cudaGetLastError(), "batched_collision_move_stage_kernel launch");
        summarize_batched_collision_cells_kernel<<<cell_blocks, threads_per_block>>>(device_destination_counts.data(),
                                                                                    total_cells,
                                                                                    cells_per_game,
                                                                                    device_collision_cells.data());
        check_cuda(cudaGetLastError(), "summarize_batched_collision_cells_kernel launch");
        batched_collision_resolve_stage_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                                    total_entities,
                                                                                    entities_per_game,
                                                                                    device_energy.data(),
                                                                                    device_destination_counts.data(),
                                                                                    cells_per_game,
                                                                                    frames.front()->width,
                                                                                    collision_config,
                                                                                    device_damaged.data(),
                                                                                    device_deaths.data(),
                                                                                    device_dropped.data());
        check_cuda(cudaGetLastError(), "batched_collision_resolve_stage_kernel launch");
        clear_batched_cell_entities_kernel<<<cell_blocks, threads_per_block>>>(device_cell_entity.data(), total_cells);
        check_cuda(cudaGetLastError(), "clear_batched_cell_entities_kernel collision launch");
        scatter_batched_entities_to_cells_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                                       total_entities,
                                                                                       entities_per_game,
                                                                                       device_cell_entity.data(),
                                                                                       cells_per_game,
                                                                                       frames.front()->width);
        check_cuda(cudaGetLastError(), "scatter_batched_entities_to_cells_kernel collision launch");
    }
    check_cuda(cudaDeviceSynchronize(), "collision apply sync");

    std::vector<unsigned long long> host_moved(frames.size());
    std::vector<unsigned long long> host_collision_cells(frames.size());
    std::vector<unsigned long long> host_damaged(frames.size());
    std::vector<unsigned long long> host_deaths(frames.size());
    std::vector<unsigned long long> host_dropped(frames.size());
    check_cuda(cudaMemcpy(host_entities.data(), device_entities.data(), total_entities * sizeof(EntityFrameEntry), cudaMemcpyDeviceToHost), "cudaMemcpy collision entities D2H");
    check_cuda(cudaMemcpy(host_energy.data(), device_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyDeviceToHost), "cudaMemcpy collision energy D2H");
    check_cuda(cudaMemcpy(host_cell_entity.data(), device_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyDeviceToHost), "cudaMemcpy collision cell entity D2H");
    check_cuda(cudaMemcpy(host_moved.data(), device_moved.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy collision moved D2H");
    check_cuda(cudaMemcpy(host_collision_cells.data(), device_collision_cells.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy collision cells D2H");
    check_cuda(cudaMemcpy(host_damaged.data(), device_damaged.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy collision damaged D2H");
    check_cuda(cudaMemcpy(host_deaths.data(), device_deaths.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy collision deaths D2H");
    check_cuda(cudaMemcpy(host_dropped.data(), device_dropped.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy collision dropped D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        std::copy(host_energy.begin() + game * cells_per_game,
                  host_energy.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_energy.begin());
        std::copy(host_cell_entity.begin() + game * cells_per_game,
                  host_cell_entity.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_entity.begin());
        stats.moved_commands += host_moved[game];
        stats.collision_cells += host_collision_cells[game];
        stats.damaged_entities += host_damaged[game];
        stats.deaths += host_deaths[game];
        stats.dropped_energy += host_dropped[game];
    }
    return stats;
}

BatchedValidationStats run_cuda_batched_validation_decisions(std::vector<StateFrame *> frames,
                                                             const GameConfig &config,
                                                             const std::vector<CommandFrame> &commands,
                                                             std::vector<ValidationFrameDecision> &decisions_out,
                                                             unsigned long iterations) {
    BatchedValidationStats stats;
    stats.games = frames.size();
    decisions_out.clear();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    if (commands.size() != frames.size()) {
        throw std::invalid_argument("validation command frame count must match frames");
    }
    const auto commands_per_game = commands.front().commands.size();
    const auto players_per_game = frames.front()->players.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto cells_per_game = frames.front()->cell_energy.size();
    stats.commands_per_game = commands_per_game;
    if (commands_per_game == 0 || players_per_game == 0 || entities_per_game == 0 || cells_per_game == 0) {
        return stats;
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        if (frames[game] == nullptr) {
            throw std::invalid_argument("run_cuda_batched_validation_decisions received a null frame");
        }
        if (commands[game].commands.size() != commands_per_game ||
            frames[game]->players.size() != players_per_game ||
            frames[game]->entities.size() != entities_per_game ||
            frames[game]->cell_energy.size() != cells_per_game) {
            throw std::invalid_argument("all validation frames must have the same batch layout");
        }
    }

    const auto total_commands = commands_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_cells = cells_per_game * frames.size();
    std::vector<FlatCommand> host_commands(total_commands);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<energy_type> host_energy(total_cells);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(commands[game].commands.begin(),
                  commands[game].commands.end(),
                  host_commands.begin() + game * commands_per_game);
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
    }
    decisions_out.assign(total_commands,
                         ValidationFrameDecision{Player::None,
                                                 Entity::None,
                                                 FlatCommandType::Move,
                                                 0,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 false});

    DeviceVector<FlatCommand> device_commands(total_commands);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<ValidationFrameDecision> device_decisions(total_commands);
    DeviceVector<unsigned long long> device_included(frames.size());
    DeviceVector<unsigned long long> device_expenses(frames.size());
    DeviceVector<unsigned long long> device_occurrences(frames.size());
    check_cuda(cudaMemcpy(device_commands.data(),
                          host_commands.data(),
                          total_commands * sizeof(FlatCommand),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched validation commands H2D");
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched validation players H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched validation entities H2D");
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy batched validation energy H2D");
    check_cuda(cudaMemcpy(device_decisions.data(),
                          decisions_out.data(),
                          total_commands * sizeof(ValidationFrameDecision),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy batched validation decisions H2D");
    check_cuda(cudaMemset(device_included.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset validation included");
    check_cuda(cudaMemset(device_expenses.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset validation expenses");
    check_cuda(cudaMemset(device_occurrences.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset validation occurrences");

    ValidationConfigDevice validation_config;
    validation_config.dropoff_cost = config.ruleset.economy.dropoff_cost;
    validation_config.dropoff_growth = config.ruleset.economy.dropoff_cost_growth;
    validation_config.spawn_cost = config.ruleset.economy.new_entity_energy_cost;
    validation_config.spawn_growth = config.ruleset.economy.spawn_cost_growth;
    validation_config.spawn_quad_threshold = config.ruleset.economy.spawn_quad_threshold;
    validation_config.spawn_quad_growth = config.ruleset.economy.spawn_quad_growth;
    validation_config.combat_enabled = config.ruleset.combat.enable_combat_commands;

    constexpr int threads_per_block = 256;
    const auto blocks = static_cast<int>((total_commands + threads_per_block - 1) / threads_per_block);
    batched_validation_decision_kernel<<<blocks, threads_per_block>>>(device_commands.data(),
                                                                      total_commands,
                                                                      commands_per_game,
                                                                      device_players.data(),
                                                                      players_per_game,
                                                                      device_entities.data(),
                                                                      entities_per_game,
                                                                      device_energy.data(),
                                                                      cells_per_game,
                                                                      frames.front()->width,
                                                                      validation_config,
                                                                      device_decisions.data(),
                                                                      device_included.data(),
                                                                      device_expenses.data(),
                                                                      device_occurrences.data(),
                                                                      iterations);
    check_cuda(cudaGetLastError(), "batched_validation_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched_validation_decision_kernel sync");

    std::vector<unsigned long long> host_included(frames.size());
    std::vector<unsigned long long> host_expenses(frames.size());
    std::vector<unsigned long long> host_occurrences(frames.size());
    check_cuda(cudaMemcpy(decisions_out.data(),
                          device_decisions.data(),
                          total_commands * sizeof(ValidationFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched validation decisions D2H");
    check_cuda(cudaMemcpy(host_included.data(),
                          device_included.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched validation included D2H");
    check_cuda(cudaMemcpy(host_expenses.data(),
                          device_expenses.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched validation expenses D2H");
    check_cuda(cudaMemcpy(host_occurrences.data(),
                          device_occurrences.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy batched validation occurrences D2H");
    for (std::size_t game = 0; game < frames.size(); ++game) {
        stats.included_commands += host_included[game];
        stats.expense_commands += host_expenses[game];
        stats.occurrence_commands += host_occurrences[game];
    }
    return stats;
}

BatchedEndEconomyStats run_cuda_batched_end_economy(std::vector<StateFrame *> frames,
                                                    const GameConfig &config) {
    BatchedEndEconomyStats stats;
    stats.games = frames.size();
    if (frames.empty()) {
        return stats;
    }

    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    stats.players_per_game = players_per_game;
    stats.dropoffs_per_game = dropoffs_per_game;
    if (players_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_end_economy received a null frame");
        }
        if (frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game) {
            throw std::invalid_argument("all end-economy frames must have the same player/dropoff layout");
        }
    }

    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    std::vector<unsigned long> host_turns(frames.size());
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(),
                      frames[game]->dropoffs.end(),
                      host_dropoffs.begin() + game * dropoffs_per_game);
        }
        host_turns[game] = frames[game]->turn_number;
    }

    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<unsigned long> device_turns(frames.size());
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy end economy players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(),
                              host_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy end economy dropoffs H2D");
    }
    check_cuda(cudaMemcpy(device_turns.data(),
                          host_turns.data(),
                          frames.size() * sizeof(unsigned long),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy end economy turns H2D");

    OverShipTaxConfigDevice tax_config;
    tax_config.tax_on = config.ruleset.economy.over_ship_tax_per_turn > 0;
    tax_config.income_on = config.ruleset.economy.ship_income_per_turn > 0;
    tax_config.quad_on = config.ruleset.economy.ship_count_deviation_penalty > 0;
    tax_config.over_ship_tax_threshold = config.ruleset.economy.over_ship_tax_threshold;
    tax_config.over_ship_tax_per_turn = config.ruleset.economy.over_ship_tax_per_turn;
    tax_config.ship_income_per_turn = config.ruleset.economy.ship_income_per_turn;
    tax_config.ship_count_target = config.ruleset.economy.ship_count_target;
    tax_config.ship_count_deviation_penalty = config.ruleset.economy.ship_count_deviation_penalty;

    HaliteRebalanceConfigDevice rebalance_config;
    rebalance_config.enabled = config.ruleset.economy.halite_rebalance_enabled;
    rebalance_config.period = config.ruleset.economy.halite_rebalance_period;
    rebalance_config.fraction = config.ruleset.economy.halite_rebalance_fraction;
    rebalance_config.min_gap_frac = config.ruleset.economy.halite_rebalance_min_gap_frac;

    constexpr int threads_per_block = 256;
    const auto player_blocks = static_cast<int>((total_players + threads_per_block - 1) / threads_per_block);
    if (tax_config.tax_on || tax_config.income_on || tax_config.quad_on) {
        batched_over_ship_tax_kernel<<<player_blocks, threads_per_block>>>(device_players.data(),
                                                                           total_players,
                                                                           tax_config);
        check_cuda(cudaGetLastError(), "batched_over_ship_tax_kernel launch");
    }
    const auto game_blocks = static_cast<int>((frames.size() + threads_per_block - 1) / threads_per_block);
    batched_halite_rebalance_kernel<<<game_blocks, threads_per_block>>>(device_players.data(),
                                                                        frames.size(),
                                                                        players_per_game,
                                                                        device_dropoffs.data(),
                                                                        dropoffs_per_game,
                                                                        device_turns.data(),
                                                                        rebalance_config);
    check_cuda(cudaGetLastError(), "batched_halite_rebalance_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "batched end economy sync");

    check_cuda(cudaMemcpy(host_players.data(),
                          device_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy end economy players D2H");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(host_dropoffs.data(),
                              device_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy end economy dropoffs D2H");
    }
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_players.begin() + game * players_per_game,
                  host_players.begin() + (game + 1) * players_per_game,
                  frames[game]->players.begin());
        if (dropoffs_per_game > 0) {
            std::copy(host_dropoffs.begin() + game * dropoffs_per_game,
                      host_dropoffs.begin() + (game + 1) * dropoffs_per_game,
                      frames[game]->dropoffs.begin());
        }
    }
    return stats;
}

BatchedMapPipelineStats run_cuda_batched_dump_regen_pipeline(std::vector<StateFrame *> frames,
                                                            const GameConfig &config,
                                                            unsigned long iterations,
                                                            energy_type dump_refill_cargo) {
    BatchedMapPipelineStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    stats.cells_per_game = cells_per_game;
    stats.entities_per_game = entities_per_game;
    if (cells_per_game == 0 || entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_dump_regen_pipeline received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->cell_initial_energy.size() != cells_per_game ||
            frame->cell_owner.size() != cells_per_game ||
            frame->entities.size() != entities_per_game ||
            frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game) {
            throw std::invalid_argument("all frames must have the same batch layout");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();

    std::vector<energy_type> host_energy(total_cells);
    std::vector<energy_type> host_initial(total_cells);
    std::vector<Player::id_type> host_owner(total_cells);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_initial_energy.begin(),
                  frames[game]->cell_initial_energy.end(),
                  host_initial.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_owner.begin(),
                  frames[game]->cell_owner.end(),
                  host_owner.begin() + game * cells_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(),
                      frames[game]->dropoffs.end(),
                      host_dropoffs.begin() + game * dropoffs_per_game);
        }
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<energy_type> device_initial(total_cells);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<unsigned long long> device_regen_deltas(frames.size());
    DeviceVector<unsigned long long> device_dump_deposits(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy pipeline energy H2D");
    check_cuda(cudaMemcpy(device_initial.data(), host_initial.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy pipeline initial H2D");
    check_cuda(cudaMemcpy(device_owner.data(), host_owner.data(), total_cells * sizeof(Player::id_type), cudaMemcpyHostToDevice), "cudaMemcpy pipeline owner H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy pipeline entities H2D");
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy pipeline players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(),
                              host_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy pipeline dropoffs H2D");
    }
    check_cuda(cudaMemset(device_regen_deltas.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset pipeline regen deltas");
    check_cuda(cudaMemset(device_dump_deposits.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset pipeline dump deposits");

    RegenConfigDevice regen_config;
    regen_config.enabled = config.ruleset.economy.cell_regen_enabled;
    regen_config.rate = config.ruleset.economy.cell_regen_rate;
    regen_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;

    constexpr int threads_per_block = 256;
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        batched_dump_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                  total_entities,
                                                                  entities_per_game,
                                                                  device_players.data(),
                                                                  players_per_game,
                                                                  device_dropoffs.data(),
                                                                  dropoffs_per_game,
                                                                  device_owner.data(),
                                                                  cells_per_game,
                                                                  device_dump_deposits.data(),
                                                                  frames.front()->width,
                                                                  config.ruleset.combat.initial_hp,
                                                                  1,
                                                                  dump_refill_cargo,
                                                                  iteration > 0);
        check_cuda(cudaGetLastError(), "pipeline batched_dump_kernel launch");
        batched_regen_kernel<<<cell_blocks, threads_per_block>>>(device_energy.data(),
                                                                 device_initial.data(),
                                                                 device_owner.data(),
                                                                 device_regen_deltas.data(),
                                                                 total_cells,
                                                                 cells_per_game,
                                                                 regen_config,
                                                                 1);
        check_cuda(cudaGetLastError(), "pipeline batched_regen_kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "pipeline kernels sync");

    std::vector<unsigned long long> host_regen_deltas(frames.size());
    std::vector<unsigned long long> host_dump_deposits(frames.size());
    check_cuda(cudaMemcpy(host_energy.data(), device_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyDeviceToHost), "cudaMemcpy pipeline energy D2H");
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy pipeline entities D2H");
    check_cuda(cudaMemcpy(host_players.data(),
                          device_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy pipeline players D2H");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(host_dropoffs.data(),
                              device_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy pipeline dropoffs D2H");
    }
    check_cuda(cudaMemcpy(host_regen_deltas.data(),
                          device_regen_deltas.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy pipeline regen deltas D2H");
    check_cuda(cudaMemcpy(host_dump_deposits.data(),
                          device_dump_deposits.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy pipeline dump deposits D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_energy.begin() + game * cells_per_game,
                  host_energy.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_energy.begin());
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        std::copy(host_players.begin() + game * players_per_game,
                  host_players.begin() + (game + 1) * players_per_game,
                  frames[game]->players.begin());
        if (dropoffs_per_game > 0) {
            std::copy(host_dropoffs.begin() + game * dropoffs_per_game,
                      host_dropoffs.begin() + (game + 1) * dropoffs_per_game,
                      frames[game]->dropoffs.begin());
        }
        frames[game]->map_total_energy += host_regen_deltas[game];
        stats.total_regen_delta += host_regen_deltas[game];
        stats.total_deposited += host_dump_deposits[game];
    }
    return stats;
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline(std::vector<StateFrame *> frames,
                                                          const GameConfig &config,
                                                          unsigned long iterations,
                                                          energy_type dump_refill_cargo,
                                                          energy_type mining_refill_cell_energy) {
    BatchedMapPipelineStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    stats.cells_per_game = cells_per_game;
    stats.entities_per_game = entities_per_game;
    if (cells_per_game == 0 || entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_economy_pipeline received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->cell_initial_energy.size() != cells_per_game ||
            frame->cell_owner.size() != cells_per_game ||
            frame->entities.size() != entities_per_game ||
            frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game) {
            throw std::invalid_argument("all frames must have the same economy batch layout");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();
    std::vector<energy_type> host_energy(total_cells);
    std::vector<energy_type> host_initial(total_cells);
    std::vector<Player::id_type> host_owner(total_cells);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_initial_energy.begin(),
                  frames[game]->cell_initial_energy.end(),
                  host_initial.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_owner.begin(),
                  frames[game]->cell_owner.end(),
                  host_owner.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(),
                      frames[game]->dropoffs.end(),
                      host_dropoffs.begin() + game * dropoffs_per_game);
        }
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<energy_type> device_initial(total_cells);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<unsigned long long> device_regen_deltas(frames.size());
    DeviceVector<unsigned long long> device_dump_deposits(frames.size());
    DeviceVector<unsigned long long> device_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy economy energy H2D");
    check_cuda(cudaMemcpy(device_initial.data(), host_initial.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy economy initial H2D");
    check_cuda(cudaMemcpy(device_owner.data(), host_owner.data(), total_cells * sizeof(Player::id_type), cudaMemcpyHostToDevice), "cudaMemcpy economy owner H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(), host_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyHostToDevice), "cudaMemcpy economy cell entity H2D");
    check_cuda(cudaMemcpy(device_entities.data(),
                          host_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy economy entities H2D");
    check_cuda(cudaMemcpy(device_players.data(),
                          host_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy economy players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(),
                              host_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy economy dropoffs H2D");
    }
    check_cuda(cudaMemset(device_regen_deltas.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset economy regen deltas");
    check_cuda(cudaMemset(device_dump_deposits.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset economy dump deposits");
    check_cuda(cudaMemset(device_mining_extracted.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset economy mining extracted");

    RegenConfigDevice regen_config;
    regen_config.enabled = config.ruleset.economy.cell_regen_enabled;
    regen_config.rate = config.ruleset.economy.cell_regen_rate;
    regen_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;
    MiningConfigDevice mining_config;
    mining_config.max_energy = config.ruleset.economy.max_energy;
    mining_config.extract_ratio = config.ruleset.economy.extract_ratio;
    mining_config.inspired_extract_ratio = config.ruleset.inspiration.extract_ratio;
    mining_config.inspired_bonus_multiplier = config.ruleset.inspiration.bonus_multiplier;
    InspirationConfigDevice inspiration_config;
    inspiration_config.enabled = config.ruleset.inspiration.enabled;
    inspiration_config.radius = config.ruleset.inspiration.radius;
    inspiration_config.ship_count_threshold = config.ruleset.inspiration.ship_count;

    constexpr int threads_per_block = 256;
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        batched_dump_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                  total_entities,
                                                                  entities_per_game,
                                                                  device_players.data(),
                                                                  players_per_game,
                                                                  device_dropoffs.data(),
                                                                  dropoffs_per_game,
                                                                  device_owner.data(),
                                                                  cells_per_game,
                                                                  device_dump_deposits.data(),
                                                                  frames.front()->width,
                                                                  config.ruleset.combat.initial_hp,
                                                                  1,
                                                                  dump_refill_cargo,
                                                                  iteration > 0);
        check_cuda(cudaGetLastError(), "economy batched_dump_kernel launch");
        if (iteration > 0 && mining_refill_cell_energy > 0) {
            refill_mining_inputs_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                              total_entities,
                                                                              entities_per_game,
                                                                              device_energy.data(),
                                                                              device_owner.data(),
                                                                              cells_per_game,
                                                                              frames.front()->width,
                                                                              mining_refill_cell_energy);
            check_cuda(cudaGetLastError(), "economy refill_mining_inputs_kernel launch");
        }
        batched_inspiration_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                         total_entities,
                                                                         entities_per_game,
                                                                         device_cell_entity.data(),
                                                                         cells_per_game,
                                                                         frames.front()->width,
                                                                         frames.front()->height,
                                                                         inspiration_config);
        check_cuda(cudaGetLastError(), "economy batched_inspiration_kernel launch");
        batched_mining_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                    total_entities,
                                                                    entities_per_game,
                                                                    device_energy.data(),
                                                                    cells_per_game,
                                                                    device_mining_extracted.data(),
                                                                    frames.front()->width,
                                                                    mining_config,
                                                                    1);
        check_cuda(cudaGetLastError(), "economy batched_mining_kernel launch");
        batched_regen_kernel<<<cell_blocks, threads_per_block>>>(device_energy.data(),
                                                                 device_initial.data(),
                                                                 device_owner.data(),
                                                                 device_regen_deltas.data(),
                                                                 total_cells,
                                                                 cells_per_game,
                                                                 regen_config,
                                                                 1);
        check_cuda(cudaGetLastError(), "economy batched_regen_kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "economy kernels sync");

    std::vector<unsigned long long> host_regen_deltas(frames.size());
    std::vector<unsigned long long> host_dump_deposits(frames.size());
    std::vector<unsigned long long> host_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(host_energy.data(), device_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyDeviceToHost), "cudaMemcpy economy energy D2H");
    check_cuda(cudaMemcpy(host_entities.data(),
                          device_entities.data(),
                          total_entities * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy economy entities D2H");
    check_cuda(cudaMemcpy(host_players.data(),
                          device_players.data(),
                          total_players * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy economy players D2H");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(host_dropoffs.data(),
                              device_dropoffs.data(),
                              total_dropoffs * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy economy dropoffs D2H");
    }
    check_cuda(cudaMemcpy(host_regen_deltas.data(),
                          device_regen_deltas.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy economy regen deltas D2H");
    check_cuda(cudaMemcpy(host_dump_deposits.data(),
                          device_dump_deposits.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy economy dump deposits D2H");
    check_cuda(cudaMemcpy(host_mining_extracted.data(),
                          device_mining_extracted.data(),
                          frames.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy economy mining extracted D2H");

    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(host_energy.begin() + game * cells_per_game,
                  host_energy.begin() + (game + 1) * cells_per_game,
                  frames[game]->cell_energy.begin());
        std::copy(host_entities.begin() + game * entities_per_game,
                  host_entities.begin() + (game + 1) * entities_per_game,
                  frames[game]->entities.begin());
        std::copy(host_players.begin() + game * players_per_game,
                  host_players.begin() + (game + 1) * players_per_game,
                  frames[game]->players.begin());
        if (dropoffs_per_game > 0) {
            std::copy(host_dropoffs.begin() + game * dropoffs_per_game,
                      host_dropoffs.begin() + (game + 1) * dropoffs_per_game,
                      frames[game]->dropoffs.begin());
        }
        frames[game]->map_total_energy += host_regen_deltas[game];
        frames[game]->map_total_energy -= host_mining_extracted[game];
        stats.total_regen_delta += host_regen_deltas[game];
        stats.total_deposited += host_dump_deposits[game];
        stats.total_mining_extracted += host_mining_extracted[game];
    }
    return stats;
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline_resident(std::vector<StateFrame *> frames,
                                                                   const GameConfig &config,
                                                                   unsigned long iterations,
                                                                   energy_type dump_refill_cargo,
                                                                   energy_type mining_refill_cell_energy) {
    BatchedMapPipelineStats stats;
    stats.games = frames.size();
    if (frames.empty() || iterations == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    stats.cells_per_game = cells_per_game;
    stats.entities_per_game = entities_per_game;
    if (cells_per_game == 0 || entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_economy_pipeline_resident received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->cell_initial_energy.size() != cells_per_game ||
            frame->cell_owner.size() != cells_per_game ||
            frame->entities.size() != entities_per_game ||
            frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game) {
            throw std::invalid_argument("all frames must have the same resident economy batch layout");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();
    std::vector<energy_type> host_energy(total_cells);
    std::vector<energy_type> host_initial(total_cells);
    std::vector<Player::id_type> host_owner(total_cells);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->cell_energy.begin(),
                  frames[game]->cell_energy.end(),
                  host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_initial_energy.begin(),
                  frames[game]->cell_initial_energy.end(),
                  host_initial.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_owner.begin(),
                  frames[game]->cell_owner.end(),
                  host_owner.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_entity.begin(),
                  frames[game]->cell_entity.end(),
                  host_cell_entity.begin() + game * cells_per_game);
        std::copy(frames[game]->entities.begin(),
                  frames[game]->entities.end(),
                  host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->players.begin(),
                  frames[game]->players.end(),
                  host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(),
                      frames[game]->dropoffs.end(),
                      host_dropoffs.begin() + game * dropoffs_per_game);
        }
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<energy_type> device_initial(total_cells);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<unsigned long long> device_regen_deltas(frames.size());
    DeviceVector<unsigned long long> device_dump_deposits(frames.size());
    DeviceVector<unsigned long long> device_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy resident economy energy H2D");
    check_cuda(cudaMemcpy(device_initial.data(), host_initial.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy resident economy initial H2D");
    check_cuda(cudaMemcpy(device_owner.data(), host_owner.data(), total_cells * sizeof(Player::id_type), cudaMemcpyHostToDevice), "cudaMemcpy resident economy owner H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(), host_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyHostToDevice), "cudaMemcpy resident economy cell entity H2D");
    check_cuda(cudaMemcpy(device_entities.data(), host_entities.data(), total_entities * sizeof(EntityFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy resident economy entities H2D");
    check_cuda(cudaMemcpy(device_players.data(), host_players.data(), total_players * sizeof(PlayerFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy resident economy players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(), host_dropoffs.data(), total_dropoffs * sizeof(DropoffFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy resident economy dropoffs H2D");
    }
    check_cuda(cudaMemset(device_regen_deltas.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset resident economy regen deltas");
    check_cuda(cudaMemset(device_dump_deposits.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset resident economy dump deposits");
    check_cuda(cudaMemset(device_mining_extracted.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset resident economy mining extracted");

    RegenConfigDevice regen_config;
    regen_config.enabled = config.ruleset.economy.cell_regen_enabled;
    regen_config.rate = config.ruleset.economy.cell_regen_rate;
    regen_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;
    MiningConfigDevice mining_config;
    mining_config.max_energy = config.ruleset.economy.max_energy;
    mining_config.extract_ratio = config.ruleset.economy.extract_ratio;
    mining_config.inspired_extract_ratio = config.ruleset.inspiration.extract_ratio;
    mining_config.inspired_bonus_multiplier = config.ruleset.inspiration.bonus_multiplier;
    InspirationConfigDevice inspiration_config;
    inspiration_config.enabled = config.ruleset.inspiration.enabled;
    inspiration_config.radius = config.ruleset.inspiration.radius;
    inspiration_config.ship_count_threshold = config.ruleset.inspiration.ship_count;

    constexpr int threads_per_block = 256;
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    for (unsigned long iteration = 0; iteration < iterations; ++iteration) {
        batched_dump_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                  total_entities,
                                                                  entities_per_game,
                                                                  device_players.data(),
                                                                  players_per_game,
                                                                  device_dropoffs.data(),
                                                                  dropoffs_per_game,
                                                                  device_owner.data(),
                                                                  cells_per_game,
                                                                  device_dump_deposits.data(),
                                                                  frames.front()->width,
                                                                  config.ruleset.combat.initial_hp,
                                                                  1,
                                                                  dump_refill_cargo,
                                                                  iteration > 0);
        check_cuda(cudaGetLastError(), "resident economy batched_dump_kernel launch");
        if (iteration > 0 && mining_refill_cell_energy > 0) {
            refill_mining_inputs_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                              total_entities,
                                                                              entities_per_game,
                                                                              device_energy.data(),
                                                                              device_owner.data(),
                                                                              cells_per_game,
                                                                              frames.front()->width,
                                                                              mining_refill_cell_energy);
            check_cuda(cudaGetLastError(), "resident economy refill_mining_inputs_kernel launch");
        }
        batched_inspiration_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                         total_entities,
                                                                         entities_per_game,
                                                                         device_cell_entity.data(),
                                                                         cells_per_game,
                                                                         frames.front()->width,
                                                                         frames.front()->height,
                                                                         inspiration_config);
        check_cuda(cudaGetLastError(), "resident economy batched_inspiration_kernel launch");
        batched_mining_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                    total_entities,
                                                                    entities_per_game,
                                                                    device_energy.data(),
                                                                    cells_per_game,
                                                                    device_mining_extracted.data(),
                                                                    frames.front()->width,
                                                                    mining_config,
                                                                    1);
        check_cuda(cudaGetLastError(), "resident economy batched_mining_kernel launch");
        batched_regen_kernel<<<cell_blocks, threads_per_block>>>(device_energy.data(),
                                                                 device_initial.data(),
                                                                 device_owner.data(),
                                                                 device_regen_deltas.data(),
                                                                 total_cells,
                                                                 cells_per_game,
                                                                 regen_config,
                                                                 1);
        check_cuda(cudaGetLastError(), "resident economy batched_regen_kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "resident economy kernels sync");

    std::vector<unsigned long long> host_regen_deltas(frames.size());
    std::vector<unsigned long long> host_dump_deposits(frames.size());
    std::vector<unsigned long long> host_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(host_regen_deltas.data(), device_regen_deltas.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy resident economy regen deltas D2H");
    check_cuda(cudaMemcpy(host_dump_deposits.data(), device_dump_deposits.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy resident economy dump deposits D2H");
    check_cuda(cudaMemcpy(host_mining_extracted.data(), device_mining_extracted.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy resident economy mining extracted D2H");
    for (std::size_t game = 0; game < frames.size(); ++game) {
        stats.total_regen_delta += host_regen_deltas[game];
        stats.total_deposited += host_dump_deposits[game];
        stats.total_mining_extracted += host_mining_extracted[game];
    }
    return stats;
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline_device_resident(std::vector<StateFrame *> frames,
                                                                          const GameConfig &config,
                                                                          unsigned long turns,
                                                                          unsigned long iterations_per_turn,
                                                                          energy_type dump_refill_cargo,
                                                                          energy_type mining_refill_cell_energy) {
    BatchedMapPipelineStats stats;
    stats.games = frames.size();
    if (frames.empty() || turns == 0 || iterations_per_turn == 0) {
        return stats;
    }
    const auto cells_per_game = frames.front()->cell_energy.size();
    const auto entities_per_game = frames.front()->entities.size();
    const auto players_per_game = frames.front()->players.size();
    const auto dropoffs_per_game = frames.front()->dropoffs.size();
    stats.cells_per_game = cells_per_game;
    stats.entities_per_game = entities_per_game;
    if (cells_per_game == 0 || entities_per_game == 0) {
        return stats;
    }
    for (const auto *frame : frames) {
        if (frame == nullptr) {
            throw std::invalid_argument("run_cuda_batched_economy_pipeline_device_resident received a null frame");
        }
        if (frame->cell_energy.size() != cells_per_game ||
            frame->cell_initial_energy.size() != cells_per_game ||
            frame->cell_owner.size() != cells_per_game ||
            frame->cell_entity.size() != cells_per_game ||
            frame->entities.size() != entities_per_game ||
            frame->players.size() != players_per_game ||
            frame->dropoffs.size() != dropoffs_per_game) {
            throw std::invalid_argument("all frames must have the same device-resident economy batch layout");
        }
    }

    const auto total_cells = cells_per_game * frames.size();
    const auto total_entities = entities_per_game * frames.size();
    const auto total_players = players_per_game * frames.size();
    const auto total_dropoffs = dropoffs_per_game * frames.size();
    std::vector<energy_type> host_energy(total_cells);
    std::vector<energy_type> host_initial(total_cells);
    std::vector<Player::id_type> host_owner(total_cells);
    std::vector<Entity::id_type> host_cell_entity(total_cells);
    std::vector<EntityFrameEntry> host_entities(total_entities);
    std::vector<PlayerFrameEntry> host_players(total_players);
    std::vector<DropoffFrameEntry> host_dropoffs(total_dropoffs);
    for (std::size_t game = 0; game < frames.size(); ++game) {
        std::copy(frames[game]->cell_energy.begin(), frames[game]->cell_energy.end(), host_energy.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_initial_energy.begin(), frames[game]->cell_initial_energy.end(), host_initial.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_owner.begin(), frames[game]->cell_owner.end(), host_owner.begin() + game * cells_per_game);
        std::copy(frames[game]->cell_entity.begin(), frames[game]->cell_entity.end(), host_cell_entity.begin() + game * cells_per_game);
        std::copy(frames[game]->entities.begin(), frames[game]->entities.end(), host_entities.begin() + game * entities_per_game);
        std::copy(frames[game]->players.begin(), frames[game]->players.end(), host_players.begin() + game * players_per_game);
        if (dropoffs_per_game > 0) {
            std::copy(frames[game]->dropoffs.begin(), frames[game]->dropoffs.end(), host_dropoffs.begin() + game * dropoffs_per_game);
        }
    }

    DeviceVector<energy_type> device_energy(total_cells);
    DeviceVector<energy_type> device_initial(total_cells);
    DeviceVector<Player::id_type> device_owner(total_cells);
    DeviceVector<Entity::id_type> device_cell_entity(total_cells);
    DeviceVector<EntityFrameEntry> device_entities(total_entities);
    DeviceVector<PlayerFrameEntry> device_players(total_players);
    DeviceVector<DropoffFrameEntry> device_dropoffs(total_dropoffs);
    DeviceVector<unsigned long long> device_regen_deltas(frames.size());
    DeviceVector<unsigned long long> device_dump_deposits(frames.size());
    DeviceVector<unsigned long long> device_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(device_energy.data(), host_energy.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy energy H2D");
    check_cuda(cudaMemcpy(device_initial.data(), host_initial.data(), total_cells * sizeof(energy_type), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy initial H2D");
    check_cuda(cudaMemcpy(device_owner.data(), host_owner.data(), total_cells * sizeof(Player::id_type), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy owner H2D");
    check_cuda(cudaMemcpy(device_cell_entity.data(), host_cell_entity.data(), total_cells * sizeof(Entity::id_type), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy cell entity H2D");
    check_cuda(cudaMemcpy(device_entities.data(), host_entities.data(), total_entities * sizeof(EntityFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy entities H2D");
    check_cuda(cudaMemcpy(device_players.data(), host_players.data(), total_players * sizeof(PlayerFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy players H2D");
    if (total_dropoffs > 0) {
        check_cuda(cudaMemcpy(device_dropoffs.data(), host_dropoffs.data(), total_dropoffs * sizeof(DropoffFrameEntry), cudaMemcpyHostToDevice), "cudaMemcpy device resident economy dropoffs H2D");
    }
    check_cuda(cudaMemset(device_regen_deltas.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset device resident economy regen deltas");
    check_cuda(cudaMemset(device_dump_deposits.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset device resident economy dump deposits");
    check_cuda(cudaMemset(device_mining_extracted.data(), 0, frames.size() * sizeof(unsigned long long)), "cudaMemset device resident economy mining extracted");

    RegenConfigDevice regen_config;
    regen_config.enabled = config.ruleset.economy.cell_regen_enabled;
    regen_config.rate = config.ruleset.economy.cell_regen_rate;
    regen_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;
    MiningConfigDevice mining_config;
    mining_config.max_energy = config.ruleset.economy.max_energy;
    mining_config.extract_ratio = config.ruleset.economy.extract_ratio;
    mining_config.inspired_extract_ratio = config.ruleset.inspiration.extract_ratio;
    mining_config.inspired_bonus_multiplier = config.ruleset.inspiration.bonus_multiplier;
    InspirationConfigDevice inspiration_config;
    inspiration_config.enabled = config.ruleset.inspiration.enabled;
    inspiration_config.radius = config.ruleset.inspiration.radius;
    inspiration_config.ship_count_threshold = config.ruleset.inspiration.ship_count;

    constexpr int threads_per_block = 256;
    const auto entity_blocks = static_cast<int>((total_entities + threads_per_block - 1) / threads_per_block);
    const auto cell_blocks = static_cast<int>((total_cells + threads_per_block - 1) / threads_per_block);
    for (unsigned long turn = 0; turn < turns; ++turn) {
        for (unsigned long iteration = 0; iteration < iterations_per_turn; ++iteration) {
            batched_dump_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                      total_entities,
                                                                      entities_per_game,
                                                                      device_players.data(),
                                                                      players_per_game,
                                                                      device_dropoffs.data(),
                                                                      dropoffs_per_game,
                                                                      device_owner.data(),
                                                                      cells_per_game,
                                                                      device_dump_deposits.data(),
                                                                      frames.front()->width,
                                                                      config.ruleset.combat.initial_hp,
                                                                      1,
                                                                      dump_refill_cargo,
                                                                      turn > 0 || iteration > 0);
            check_cuda(cudaGetLastError(), "device resident economy batched_dump_kernel launch");
            if ((turn > 0 || iteration > 0) && mining_refill_cell_energy > 0) {
                refill_mining_inputs_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                                  total_entities,
                                                                                  entities_per_game,
                                                                                  device_energy.data(),
                                                                                  device_owner.data(),
                                                                                  cells_per_game,
                                                                                  frames.front()->width,
                                                                                  mining_refill_cell_energy);
                check_cuda(cudaGetLastError(), "device resident economy refill_mining_inputs_kernel launch");
            }
            batched_inspiration_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                             total_entities,
                                                                             entities_per_game,
                                                                             device_cell_entity.data(),
                                                                             cells_per_game,
                                                                             frames.front()->width,
                                                                             frames.front()->height,
                                                                             inspiration_config);
            check_cuda(cudaGetLastError(), "device resident economy batched_inspiration_kernel launch");
            batched_mining_kernel<<<entity_blocks, threads_per_block>>>(device_entities.data(),
                                                                        total_entities,
                                                                        entities_per_game,
                                                                        device_energy.data(),
                                                                        cells_per_game,
                                                                        device_mining_extracted.data(),
                                                                        frames.front()->width,
                                                                        mining_config,
                                                                        1);
            check_cuda(cudaGetLastError(), "device resident economy batched_mining_kernel launch");
            batched_regen_kernel<<<cell_blocks, threads_per_block>>>(device_energy.data(),
                                                                     device_initial.data(),
                                                                     device_owner.data(),
                                                                     device_regen_deltas.data(),
                                                                     total_cells,
                                                                     cells_per_game,
                                                                     regen_config,
                                                                     1);
            check_cuda(cudaGetLastError(), "device resident economy batched_regen_kernel launch");
        }
    }
    check_cuda(cudaDeviceSynchronize(), "device resident economy kernels sync");

    std::vector<unsigned long long> host_regen_deltas(frames.size());
    std::vector<unsigned long long> host_dump_deposits(frames.size());
    std::vector<unsigned long long> host_mining_extracted(frames.size());
    check_cuda(cudaMemcpy(host_regen_deltas.data(), device_regen_deltas.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy device resident economy regen deltas D2H");
    check_cuda(cudaMemcpy(host_dump_deposits.data(), device_dump_deposits.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy device resident economy dump deposits D2H");
    check_cuda(cudaMemcpy(host_mining_extracted.data(), device_mining_extracted.data(), frames.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy device resident economy mining extracted D2H");
    for (std::size_t game = 0; game < frames.size(); ++game) {
        stats.total_regen_delta += host_regen_deltas[game];
        stats.total_deposited += host_dump_deposits[game];
        stats.total_mining_extracted += host_mining_extracted[game];
    }
    return stats;
}

} // namespace hlt::gpu
