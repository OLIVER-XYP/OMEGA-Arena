#include "CudaConstruction.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct ConstructionConfigDevice {
    energy_type base_cost{};
    double growth{};
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

__global__ void construction_decision_kernel(const FlatCommand *commands,
                                             std::size_t command_count,
                                             const PlayerFrameEntry *players,
                                             std::size_t player_count,
                                             const EntityFrameEntry *entities,
                                             std::size_t entity_count,
                                             const Player::id_type *cell_owner,
                                             int width,
                                             ConstructionConfigDevice config,
                                             ConstructionFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count) {
        return;
    }

    const auto command = commands[index];
    ConstructionFrameDecision decision{command.player,
                                       command.entity,
                                       command.target_location,
                                       0,
                                       Player::None,
                                       command.type == FlatCommandType::Construct,
                                       false,
                                       false};
    if (!decision.command) {
        decisions[index] = decision;
        return;
    }

    const auto player_index = find_player_index(players, player_count, command.player);
    if (player_index < 0) {
        decision.entity_missing = true;
        decisions[index] = decision;
        return;
    }

    decision.cost = scaled_dropoff_cost_device(config.base_cost, players[player_index].dropoff_count, config.growth);

    const auto entity_index = find_entity_index(entities, entity_count, command.entity, command.player);
    if (entity_index < 0) {
        decision.entity_missing = true;
        decisions[index] = decision;
        return;
    }

    const auto entity = entities[entity_index];
    decision.location = entity.location;
    const auto cell_index_value = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    decision.cell_owner = cell_owner[cell_index_value];
    if (decision.cell_owner.value != Player::None.value) {
        decision.cell_owned = true;
    }
    decisions[index] = decision;
}

} // namespace

std::vector<ConstructionFrameDecision> run_cuda_construction_decisions(const StateFrame &state_frame,
                                                                       const CommandFrame &commands,
                                                                       const GameConfig &config) {
    std::vector<ConstructionFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (std::size_t index = 0; index < commands.commands.size(); ++index) {
        decisions.push_back(ConstructionFrameDecision{Player::None,
                                                     Entity::None,
                                                     Location{0, 0},
                                                     0,
                                                     Player::None,
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
    DeviceBuffer<Player::id_type> owner_buffer(state_frame.cell_owner);
    DeviceBuffer<ConstructionFrameDecision> decision_buffer(decisions);

    ConstructionConfigDevice device_config;
    device_config.base_cost = config.ruleset.economy.dropoff_cost;
    device_config.growth = config.ruleset.economy.dropoff_cost_growth;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
    construction_decision_kernel<<<blocks, threads_per_block>>>(command_buffer.data(),
                                                                commands.commands.size(),
                                                                player_buffer.data(),
                                                                state_frame.players.size(),
                                                                entity_buffer.data(),
                                                                state_frame.entities.size(),
                                                                owner_buffer.data(),
                                                                state_frame.width,
                                                                device_config,
                                                                decision_buffer.data());
    check_cuda(cudaGetLastError(), "construction_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "construction_decision_kernel sync");
    check_cuda(cudaMemcpy(decisions.data(),
                          decision_buffer.data(),
                          decisions.size() * sizeof(ConstructionFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy construction decisions D2H");
    return decisions;
}

} // namespace hlt::gpu
