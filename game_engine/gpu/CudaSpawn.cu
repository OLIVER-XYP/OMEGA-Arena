#include "CudaSpawn.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct SpawnConfigDevice {
    energy_type base_cost{};
    double growth{};
    unsigned long quad_threshold{};
    double quad_growth{};
    bool emergency_enabled{};
    unsigned long emergency_period{};
    unsigned long emergency_count{};
    energy_type emergency_min_energy{};
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
    return static_cast<energy_type>(static_cast<double>(base) * factor);
}

__device__ int find_player_index(const PlayerFrameEntry *players,
                                 std::size_t player_count,
                                 Player::id_type player_id) {
    for (std::size_t index = 0; index < player_count; ++index) {
        if (players[index].id.value == player_id.value) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

__device__ int find_entity_index(const EntityFrameEntry *entities,
                                 std::size_t entity_count,
                                 Entity::id_type entity_id) {
    for (std::size_t index = 0; index < entity_count; ++index) {
        if (entities[index].id.value == entity_id.value && entities[index].alive) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

__device__ std::size_t location_index(Location location, int width) {
    return static_cast<std::size_t>(location.y * width + location.x);
}

__device__ Location neighbor_at(Location location, int width, int height, int offset) {
    Location result = location;
    switch (offset) {
    case 0:
        result.x = (location.x + 1) % width;
        break;
    case 1:
        result.x = (location.x - 1 + width) % width;
        break;
    case 2:
        result.y = (location.y + 1) % height;
        break;
    default:
        result.y = (location.y - 1 + height) % height;
        break;
    }
    return result;
}

bool emergency_spawn_possible(const StateFrame &state_frame, const GameConfig &config) {
    const auto &economy = config.ruleset.economy;
    return economy.emergency_spawn_enabled &&
           economy.emergency_spawn_period > 0 &&
           economy.emergency_spawn_count > 0 &&
           state_frame.turn_number > 0 &&
           state_frame.turn_number % economy.emergency_spawn_period == 0;
}

__global__ void spawn_decision_kernel(const FlatCommand *commands,
                                      std::size_t command_count,
                                      const PlayerFrameEntry *players,
                                      std::size_t player_count,
                                      const EntityFrameEntry *entities,
                                      std::size_t entity_count,
                                      const Entity::id_type *cell_entity,
                                      const Player::id_type *cell_owner,
                                      int width,
                                      SpawnConfigDevice config,
                                      SpawnFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count) {
        return;
    }

    const auto command = commands[index];
    SpawnFrameDecision decision{command.player,
                                command.target_location,
                                Entity::None,
                                Player::None,
                                0,
                                command.type == FlatCommandType::Spawn,
                                false,
                                false};
    if (!decision.command) {
        decisions[index] = decision;
        return;
    }

    const auto player_index = find_player_index(players, player_count, command.player);
    if (player_index < 0) {
        decisions[index] = decision;
        return;
    }

    const auto player = players[player_index];
    decision.factory = player.factory;
    decision.cost = scaled_spawn_cost_device(config.base_cost,
                                             player.ship_count,
                                             config.growth,
                                             config.quad_threshold,
                                             config.quad_growth);
    const auto factory_index = location_index(player.factory, width);
    decision.existing_entity = cell_entity[factory_index];
    decision.cell_owner = cell_owner[factory_index];
    decision.factory_occupied = decision.existing_entity.value != Entity::None.value;
    if (decision.factory_occupied) {
        const auto entity_index = find_entity_index(entities, entity_count, decision.existing_entity);
        decision.self_collision = entity_index >= 0 && entities[entity_index].owner.value == decision.cell_owner.value;
    }
    decisions[index] = decision;
}

__global__ void emergency_spawn_decision_kernel(const PlayerFrameEntry *players,
                                                std::size_t player_count,
                                                const Entity::id_type *cell_entity,
                                                int width,
                                                int height,
                                                unsigned long turn_number,
                                                SpawnConfigDevice config,
                                                EmergencySpawnFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= player_count) {
        return;
    }

    const auto player = players[index];
    EmergencySpawnFrameDecision decision{player.id,
                                        0,
                                        {player.factory, player.factory, player.factory, player.factory, player.factory}};
    for (int i = 0; i < 5; ++i) {
        decision.locations[i] = player.factory;
    }

    if (!config.emergency_enabled ||
        config.emergency_period == 0 ||
        config.emergency_count == 0 ||
        turn_number == 0 ||
        turn_number % config.emergency_period != 0 ||
        player.ship_count > 0 ||
        player.energy >= config.emergency_min_energy) {
        decisions[index] = decision;
        return;
    }

    const auto max_count = config.emergency_count < 5 ? config.emergency_count : 5;
    if (cell_entity[location_index(player.factory, width)].value == Entity::None.value && decision.count < max_count) {
        decision.locations[decision.count++] = player.factory;
    }

    for (int neighbor_index = 0; neighbor_index < 4 && decision.count < max_count; ++neighbor_index) {
        const auto location = neighbor_at(player.factory, width, height, neighbor_index);
        if (cell_entity[location_index(location, width)].value == Entity::None.value) {
            decision.locations[decision.count++] = location;
        }
    }
    decisions[index] = decision;
}

} // namespace

SpawnFrameDecisions run_cuda_spawn_decisions(const StateFrame &state_frame,
                                             const CommandFrame &commands,
                                             const GameConfig &config) {
    SpawnFrameDecisions output;
    output.spawns.reserve(commands.commands.size());
    for (std::size_t index = 0; index < commands.commands.size(); ++index) {
        output.spawns.push_back(SpawnFrameDecision{Player::None,
                                                  Location{0, 0},
                                                  Entity::None,
                                                  Player::None,
                                                  0,
                                                  false,
                                                  false,
                                                  false});
    }
    output.emergency_spawns.reserve(state_frame.players.size());
    for (std::size_t index = 0; index < state_frame.players.size(); ++index) {
        output.emergency_spawns.push_back(EmergencySpawnFrameDecision{Player::None,
                                                                     0,
                                                                     {Location{0, 0},
                                                                      Location{0, 0},
                                                                      Location{0, 0},
                                                                      Location{0, 0},
                                                                      Location{0, 0}}});
    }

    SpawnConfigDevice device_config;
    device_config.base_cost = config.ruleset.economy.new_entity_energy_cost;
    device_config.growth = config.ruleset.economy.spawn_cost_growth;
    device_config.quad_threshold = config.ruleset.economy.spawn_quad_threshold;
    device_config.quad_growth = config.ruleset.economy.spawn_quad_growth;
    device_config.emergency_enabled = config.ruleset.economy.emergency_spawn_enabled;
    device_config.emergency_period = config.ruleset.economy.emergency_spawn_period;
    device_config.emergency_count = config.ruleset.economy.emergency_spawn_count;
    device_config.emergency_min_energy = config.ruleset.economy.new_entity_energy_cost;

    DeviceBuffer<PlayerFrameEntry> player_buffer(state_frame.players);
    DeviceBuffer<EntityFrameEntry> entity_buffer(state_frame.entities);
    DeviceBuffer<Entity::id_type> cell_entity_buffer(state_frame.cell_entity);
    DeviceBuffer<Player::id_type> cell_owner_buffer(state_frame.cell_owner);

    constexpr int threads_per_block = 128;
    if (!commands.commands.empty()) {
        DeviceBuffer<FlatCommand> command_buffer(commands.commands);
        DeviceBuffer<SpawnFrameDecision> spawn_buffer(output.spawns);
        const auto blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
        spawn_decision_kernel<<<blocks, threads_per_block>>>(command_buffer.data(),
                                                             commands.commands.size(),
                                                             player_buffer.data(),
                                                             state_frame.players.size(),
                                                             entity_buffer.data(),
                                                             state_frame.entities.size(),
                                                             cell_entity_buffer.data(),
                                                             cell_owner_buffer.data(),
                                                             state_frame.width,
                                                             device_config,
                                                             spawn_buffer.data());
        check_cuda(cudaGetLastError(), "spawn_decision_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "spawn_decision_kernel sync");
        check_cuda(cudaMemcpy(output.spawns.data(),
                              spawn_buffer.data(),
                              output.spawns.size() * sizeof(SpawnFrameDecision),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy spawn decisions D2H");
    }

    if (emergency_spawn_possible(state_frame, config) && !state_frame.players.empty()) {
        DeviceBuffer<EmergencySpawnFrameDecision> emergency_buffer(output.emergency_spawns);
        const auto blocks = static_cast<int>((state_frame.players.size() + threads_per_block - 1) / threads_per_block);
        emergency_spawn_decision_kernel<<<blocks, threads_per_block>>>(player_buffer.data(),
                                                                       state_frame.players.size(),
                                                                       cell_entity_buffer.data(),
                                                                       state_frame.width,
                                                                       state_frame.height,
                                                                       state_frame.turn_number,
                                                                       device_config,
                                                                       emergency_buffer.data());
        check_cuda(cudaGetLastError(), "emergency_spawn_decision_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "emergency_spawn_decision_kernel sync");
        check_cuda(cudaMemcpy(output.emergency_spawns.data(),
                              emergency_buffer.data(),
                              output.emergency_spawns.size() * sizeof(EmergencySpawnFrameDecision),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy emergency spawn decisions D2H");
    }

    return output;
}

} // namespace hlt::gpu
