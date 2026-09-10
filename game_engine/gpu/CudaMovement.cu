#include "CudaMovement.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct MovementConfigDevice {
    energy_type move_cost_ratio{};
    energy_type inspired_move_cost_ratio{};
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

__global__ void movement_decision_kernel(const FlatCommand *commands,
                                         std::size_t command_count,
                                         const PlayerFrameEntry *players,
                                         std::size_t player_count,
                                         const EntityFrameEntry *entities,
                                         std::size_t entity_count,
                                         const energy_type *cell_energy,
                                         int width,
                                         int height,
                                         MovementConfigDevice config,
                                         MovementFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count) {
        return;
    }

    const auto command = commands[index];
    MovementFrameDecision decision{command.player,
                                   command.entity,
                                   command.target_location,
                                   command.target_location,
                                   0,
                                   0,
                                   command.type == FlatCommandType::Move && command.direction != Direction::Still,
                                   false,
                                   false};
    if (!decision.command) {
        decisions[index] = decision;
        return;
    }

    if (find_player_index(players, player_count, command.player) < 0) {
        decision.entity_missing = true;
        decisions[index] = decision;
        return;
    }

    const auto entity_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
    if (entity_index < 0) {
        decision.entity_missing = true;
        decisions[index] = decision;
        return;
    }

    const auto entity = entities[entity_index];
    decision.from = entity.location;
    decision.to = move_location_device(entity.location, command.direction, width, height);
    const auto source_index = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    const auto cost_ratio = entity.is_inspired ? config.inspired_move_cost_ratio : config.move_cost_ratio;
    decision.required = cost_ratio == 0 ? 0 : cell_energy[source_index] / cost_ratio;
    decision.current_energy = entity.energy;
    decision.insufficient_energy = entity.energy < decision.required;
    decisions[index] = decision;
}

} // namespace

std::vector<MovementFrameDecision> run_cuda_movement_decisions(const StateFrame &state_frame,
                                                               const CommandFrame &commands,
                                                               const GameConfig &config) {
    std::vector<MovementFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (std::size_t index = 0; index < commands.commands.size(); ++index) {
        decisions.push_back(MovementFrameDecision{Player::None,
                                                 Entity::None,
                                                 Location{0, 0},
                                                 Location{0, 0},
                                                 0,
                                                 0,
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
    DeviceBuffer<MovementFrameDecision> decision_buffer(decisions);

    MovementConfigDevice device_config;
    device_config.move_cost_ratio = config.ruleset.economy.move_cost_ratio;
    device_config.inspired_move_cost_ratio = config.ruleset.inspiration.move_cost_ratio;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
    movement_decision_kernel<<<blocks, threads_per_block>>>(command_buffer.data(),
                                                            commands.commands.size(),
                                                            player_buffer.data(),
                                                            state_frame.players.size(),
                                                            entity_buffer.data(),
                                                            state_frame.entities.size(),
                                                            cell_energy_buffer.data(),
                                                            state_frame.width,
                                                            state_frame.height,
                                                            device_config,
                                                            decision_buffer.data());
    check_cuda(cudaGetLastError(), "movement_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "movement_decision_kernel sync");
    check_cuda(cudaMemcpy(decisions.data(),
                          decision_buffer.data(),
                          decisions.size() * sizeof(MovementFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy movement decisions D2H");
    return decisions;
}

} // namespace hlt::gpu
