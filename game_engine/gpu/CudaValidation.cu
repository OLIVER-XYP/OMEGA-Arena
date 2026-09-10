#include "CudaValidation.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct ValidationConfigDevice {
    energy_type dropoff_cost{};
    double dropoff_growth{};
    energy_type spawn_cost{};
    double spawn_growth{};
    unsigned long spawn_quad_threshold{};
    double spawn_quad_growth{};
    bool combat_enabled{};
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

__device__ energy_type scaled_dropoff_cost_device(energy_type base, std::size_t num_dropoffs, double growth) {
    if (growth <= 0.0) {
        return base;
    }
    return static_cast<energy_type>(static_cast<double>(base) * (1.0 + growth * static_cast<double>(num_dropoffs)));
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

__device__ int find_owned_entity_index(const EntityFrameEntry *entities,
                                       std::size_t entity_count,
                                       Entity::id_type entity_id,
                                       Player::id_type player_id) {
    for (std::size_t index = 0; index < entity_count; ++index) {
        if (entities[index].id.value == entity_id.value &&
            entities[index].owner.value == player_id.value &&
            entities[index].alive) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

__global__ void validation_decision_kernel(const FlatCommand *commands,
                                           std::size_t command_count,
                                           const PlayerFrameEntry *players,
                                           std::size_t player_count,
                                           const EntityFrameEntry *entities,
                                           std::size_t entity_count,
                                           const energy_type *cell_energy,
                                           int width,
                                           ValidationConfigDevice config,
                                           ValidationFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count) {
        return;
    }

    const auto command = commands[index];
    ValidationFrameDecision decision{command.player,
                                     command.entity,
                                     command.type,
                                     0,
                                     true,
                                     false,
                                     false,
                                     config.combat_enabled,
                                     false,
                                     false,
                                     false};
    const auto player_index = find_player_index(players, player_count, command.player);
    if (player_index < 0) {
        decisions[index] = decision;
        return;
    }
    decision.player_exists = true;
    const auto player = players[player_index];

    if (command.type == FlatCommandType::Move) {
        const auto entity_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
        decision.ownership_ok = entity_index >= 0;
        decision.occurrence_command = decision.ownership_ok;
        decision.include_in_batch = decision.ownership_ok;
        decisions[index] = decision;
        return;
    }

    if (command.type == FlatCommandType::Construct) {
        const auto entity_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
        decision.ownership_ok = entity_index >= 0;
        decision.occurrence_command = decision.ownership_ok;
        decision.expense_command = decision.ownership_ok;
        decision.include_in_batch = decision.ownership_ok;
        if (entity_index >= 0) {
            const auto entity = entities[entity_index];
            energy_type cost = scaled_dropoff_cost_device(config.dropoff_cost, player.dropoff_count, config.dropoff_growth);
            const auto location_index = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
            const auto credit = cell_energy[location_index] + entity.energy;
            decision.expense = credit >= cost ? 0 : cost - credit;
        }
        decisions[index] = decision;
        return;
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
        decisions[index] = decision;
        return;
    }

    if (command.type == FlatCommandType::AttackShip || command.type == FlatCommandType::AttackStructure) {
        const auto entity_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
        decision.ownership_ok = entity_index >= 0;
        decision.occurrence_command = config.combat_enabled && decision.ownership_ok;
        decision.include_in_batch = decision.occurrence_command;
        decisions[index] = decision;
        return;
    }

    if (command.type == FlatCommandType::Defend || command.type == FlatCommandType::Heal) {
        const auto entity_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
        decision.ownership_ok = entity_index >= 0;
        decision.occurrence_command = config.combat_enabled && decision.ownership_ok;
        decision.include_in_batch = decision.occurrence_command;
        decisions[index] = decision;
        return;
    }

    decisions[index] = decision;
}

} // namespace

std::vector<ValidationFrameDecision> run_cuda_validation_decisions(const StateFrame &state_frame,
                                                                   const CommandFrame &commands,
                                                                   const GameConfig &config) {
    std::vector<ValidationFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (std::size_t index = 0; index < commands.commands.size(); ++index) {
        decisions.push_back(ValidationFrameDecision{Player::None,
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
    }
    if (commands.commands.empty()) {
        return decisions;
    }

    DeviceBuffer<FlatCommand> command_buffer(commands.commands);
    DeviceBuffer<PlayerFrameEntry> player_buffer(state_frame.players);
    DeviceBuffer<EntityFrameEntry> entity_buffer(state_frame.entities);
    DeviceBuffer<energy_type> cell_energy_buffer(state_frame.cell_energy);
    DeviceBuffer<ValidationFrameDecision> decision_buffer(decisions);

    ValidationConfigDevice device_config;
    device_config.dropoff_cost = config.ruleset.economy.dropoff_cost;
    device_config.dropoff_growth = config.ruleset.economy.dropoff_cost_growth;
    device_config.spawn_cost = config.ruleset.economy.new_entity_energy_cost;
    device_config.spawn_growth = config.ruleset.economy.spawn_cost_growth;
    device_config.spawn_quad_threshold = config.ruleset.economy.spawn_quad_threshold;
    device_config.spawn_quad_growth = config.ruleset.economy.spawn_quad_growth;
    device_config.combat_enabled = config.ruleset.combat.enable_combat_commands;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
    validation_decision_kernel<<<blocks, threads_per_block>>>(command_buffer.data(),
                                                              commands.commands.size(),
                                                              player_buffer.data(),
                                                              state_frame.players.size(),
                                                              entity_buffer.data(),
                                                              state_frame.entities.size(),
                                                              cell_energy_buffer.data(),
                                                              state_frame.width,
                                                              device_config,
                                                              decision_buffer.data());
    check_cuda(cudaGetLastError(), "validation_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "validation_decision_kernel sync");
    check_cuda(cudaMemcpy(decisions.data(),
                          decision_buffer.data(),
                          decisions.size() * sizeof(ValidationFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy validation decisions D2H");
    return decisions;
}

} // namespace hlt::gpu
