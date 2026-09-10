#include "CudaMining.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct MiningConfigDevice {
    energy_type max_energy{};
    unsigned long extract_ratio{};
    unsigned long inspired_extract_ratio{};
    double inspired_bonus_multiplier{};
    bool defend_allows_mining{};
    int mining_interference_range{};
    double mining_interference_ratio{};
    double mining_interference_cargo_loss_ratio{};
};

struct PlunderConfigDevice {
    int range{};
    energy_type halite_per_turn{};
    energy_type max_energy{};
};

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
    T *ptr{};
    std::size_t count{};

public:
    explicit DeviceBuffer(const std::vector<T> &host) : count(host.size()) {
        if (count > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), "cudaMalloc");
            check_cuda(cudaMemcpy(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        }
    }

    ~DeviceBuffer() {
        cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *data() { return ptr; }
    const T *data() const { return ptr; }
};

__device__ int wrapped_distance(int width, int height, int ax, int ay, int bx, int by) {
    int dx = ax > bx ? ax - bx : bx - ax;
    int dy = ay > by ? ay - by : by - ay;
    const int wx = width - dx;
    const int wy = height - dy;
    return (dx < wx ? dx : wx) + (dy < wy ? dy : wy);
}

__device__ bool contains_changed_entity(const Entity::id_type *changed_entities,
                                        std::size_t changed_count,
                                        Entity::id_type entity_id) {
    for (std::size_t i = 0; i < changed_count; ++i) {
        if (changed_entities[i].value == entity_id.value) {
            return true;
        }
    }
    return false;
}

__global__ void mining_kernel(EntityFrameEntry *entities,
                              std::size_t entity_count,
                              energy_type *cell_energy,
                              MiningEffectFrameEntry *effects,
                              const Entity::id_type *changed_entities,
                              std::size_t changed_count,
                              int width,
                              int height,
                              MiningConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive || entity.energy >= config.max_energy) {
        return;
    }

    const bool defend_mines = config.defend_allows_mining && entity.is_defending;
    const bool already_acted = contains_changed_entity(changed_entities, changed_count, entity.id);
    if (already_acted && !defend_mines) {
        return;
    }

    const auto cell_index = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    const auto ratio = entity.is_inspired ? config.inspired_extract_ratio : config.extract_ratio;
    energy_type extracted = static_cast<energy_type>(ceil(static_cast<double>(cell_energy[cell_index]) / static_cast<double>(ratio)));
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

    const bool interference_active = config.mining_interference_range > 0 &&
                                     (config.mining_interference_ratio > 0.0 ||
                                      config.mining_interference_cargo_loss_ratio > 0.0);
    if (interference_active && !entity.is_defending) {
        bool interfered = false;
        for (std::size_t other_index = 0; other_index < entity_count; ++other_index) {
            const auto other = entities[other_index];
            if (!other.alive || other.owner.value == entity.owner.value || other.is_defending) {
                continue;
            }
            if (wrapped_distance(width,
                                 height,
                                 entity.location.x,
                                 entity.location.y,
                                 other.location.x,
                                 other.location.y) <= config.mining_interference_range) {
                interfered = true;
                break;
            }
        }
        if (interfered) {
            if (gained > 0 && config.mining_interference_ratio > 0.0) {
                gained = static_cast<energy_type>(static_cast<double>(gained) * (1.0 - config.mining_interference_ratio));
                extracted = static_cast<energy_type>(static_cast<double>(extracted) * (1.0 - config.mining_interference_ratio));
            }
            if (config.mining_interference_cargo_loss_ratio > 0.0 && entity.energy > 0) {
                energy_type loss = static_cast<energy_type>(static_cast<double>(entity.energy) * config.mining_interference_cargo_loss_ratio);
                if (loss == 0) {
                    loss = 1;
                }
                gained -= loss;
            }
        }
    }

    if (config.max_energy - entity.energy < gained) {
        gained = config.max_energy - entity.energy;
    }

    entity.energy += gained;
    entities[index] = entity;

    effects[index] = MiningEffectFrameEntry{entity.id,
                                            entity.owner,
                                            entity.location,
                                            extracted,
                                            gained,
                                            entity.was_captured};
}

__global__ void apply_mining_effects_kernel(energy_type *cell_energy,
                                            unsigned long long *map_total_energy,
                                            const MiningEffectFrameEntry *effects,
                                            std::size_t effect_count,
                                            int width) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    unsigned long long extracted_total = 0;
    for (std::size_t index = 0; index < effect_count; ++index) {
        const auto effect = effects[index];
        if (effect.entity.value == Entity::None.value || effect.extracted <= 0) {
            continue;
        }
        const auto cell_index = static_cast<std::size_t>(effect.location.y * width + effect.location.x);
        cell_energy[cell_index] -= effect.extracted;
        extracted_total += static_cast<unsigned long long>(effect.extracted);
    }
    *map_total_energy -= extracted_total;
}

__global__ void plunder_kernel(EntityFrameEntry *entities,
                               std::size_t entity_count,
                               const PlayerFrameEntry *players,
                               std::size_t player_count,
                               const DropoffFrameEntry *dropoffs,
                               int width,
                               int height,
                               PlunderConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count || config.range <= 0 || config.halite_per_turn <= 0) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive || entity.energy >= config.max_energy) {
        return;
    }

    for (std::size_t player_index = 0; player_index < player_count; ++player_index) {
        const auto player = players[player_index];
        if (player.id.value == entity.owner.value) {
            continue;
        }
        if (wrapped_distance(width,
                             height,
                             entity.location.x,
                             entity.location.y,
                             player.factory.x,
                             player.factory.y) <= config.range) {
            const auto candidate = entity.energy + config.halite_per_turn;
            entities[index].energy = candidate < config.max_energy ? candidate : config.max_energy;
            return;
        }

        const auto dropoff_end = player.dropoff_begin + player.dropoff_count;
        for (std::size_t dropoff_index = player.dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
            const auto dropoff = dropoffs[dropoff_index];
            if (wrapped_distance(width,
                                 height,
                                 entity.location.x,
                                 entity.location.y,
                                 dropoff.location.x,
                                 dropoff.location.y) <= config.range) {
                const auto candidate = entity.energy + config.halite_per_turn;
                entities[index].energy = candidate < config.max_energy ? candidate : config.max_energy;
                return;
            }
        }
    }
}

void run_cuda_mining_impl(StateFrame &state_frame,
                          const TurnFrame &turn_frame,
                          const GameConfig &config,
                          std::vector<MiningEffectFrameEntry> *effects,
                          bool run_plunder) {
    if (state_frame.entities.empty()) {
        if (effects != nullptr) {
            effects->clear();
        }
        return;
    }

    DeviceBuffer<EntityFrameEntry> entities(state_frame.entities);
    DeviceBuffer<energy_type> cell_energy(state_frame.cell_energy);
    DeviceBuffer<Entity::id_type> changed_entities(turn_frame.changed_entities);

    std::vector<MiningEffectFrameEntry> zero_effects(state_frame.entities.size());
    DeviceBuffer<MiningEffectFrameEntry> mining_effects(zero_effects);

    unsigned long long *device_map_total_energy{};
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_map_total_energy), sizeof(unsigned long long)), "cudaMalloc map total energy");
    check_cuda(cudaMemcpy(device_map_total_energy,
                          &state_frame.map_total_energy,
                          sizeof(unsigned long long),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy map total energy H2D");

    MiningConfigDevice device_config;
    device_config.max_energy = config.ruleset.economy.max_energy;
    device_config.extract_ratio = config.ruleset.economy.extract_ratio;
    device_config.inspired_extract_ratio = config.ruleset.inspiration.extract_ratio;
    device_config.inspired_bonus_multiplier = config.ruleset.inspiration.bonus_multiplier;
    device_config.defend_allows_mining = config.ruleset.combat.defend_allows_mining;
    device_config.mining_interference_range = config.ruleset.economy.mining_interference_range;
    device_config.mining_interference_ratio = config.ruleset.economy.mining_interference_ratio;
    device_config.mining_interference_cargo_loss_ratio = config.ruleset.economy.mining_interference_cargo_loss_ratio;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.entities.size() + threads_per_block - 1) / threads_per_block);
    mining_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                                 state_frame.entities.size(),
                                                 cell_energy.data(),
                                                 mining_effects.data(),
                                                 changed_entities.data(),
                                                 turn_frame.changed_entities.size(),
                                                 state_frame.width,
                                                 state_frame.height,
                                                 device_config);
    check_cuda(cudaGetLastError(), "mining_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "mining_kernel sync");

    apply_mining_effects_kernel<<<1, 1>>>(cell_energy.data(),
                                          device_map_total_energy,
                                          mining_effects.data(),
                                          state_frame.entities.size(),
                                          state_frame.width);
    check_cuda(cudaGetLastError(), "apply_mining_effects_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "apply_mining_effects_kernel sync");

    if (run_plunder) {
        PlunderConfigDevice plunder_config;
        plunder_config.range = config.ruleset.economy.plunder_range;
        plunder_config.halite_per_turn = config.ruleset.economy.plunder_halite_per_turn;
        plunder_config.max_energy = config.ruleset.economy.max_energy;
        if (plunder_config.range > 0 && plunder_config.halite_per_turn > 0) {
            DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
            DeviceBuffer<DropoffFrameEntry> dropoffs(state_frame.dropoffs);
            plunder_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                                          state_frame.entities.size(),
                                                          players.data(),
                                                          state_frame.players.size(),
                                                          dropoffs.data(),
                                                          state_frame.width,
                                                          state_frame.height,
                                                          plunder_config);
            check_cuda(cudaGetLastError(), "plunder_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "plunder_kernel sync");
        }
    }

    check_cuda(cudaMemcpy(state_frame.entities.data(),
                          entities.data(),
                          state_frame.entities.size() * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy mining entities D2H");
    check_cuda(cudaMemcpy(state_frame.cell_energy.data(),
                          cell_energy.data(),
                          state_frame.cell_energy.size() * sizeof(energy_type),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy mining cells D2H");
    check_cuda(cudaMemcpy(&state_frame.map_total_energy,
                          device_map_total_energy,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy map total energy D2H");
    check_cuda(cudaFree(device_map_total_energy), "cudaFree map total energy");
    if (effects != nullptr) {
        effects->resize(state_frame.entities.size());
        check_cuda(cudaMemcpy(effects->data(),
                              mining_effects.data(),
                              state_frame.entities.size() * sizeof(MiningEffectFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy mining effects D2H");
    }
}

} // namespace

void run_cuda_mining(StateFrame &state_frame,
                     const TurnFrame &turn_frame,
                     const GameConfig &config,
                     std::vector<MiningEffectFrameEntry> *effects) {
    run_cuda_mining_impl(state_frame, turn_frame, config, effects, false);
}

void run_cuda_mining_and_plunder(StateFrame &state_frame,
                                 const TurnFrame &turn_frame,
                                 const GameConfig &config,
                                 std::vector<MiningEffectFrameEntry> *effects) {
    run_cuda_mining_impl(state_frame, turn_frame, config, effects, true);
}

} // namespace hlt::gpu
