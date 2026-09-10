#include "CudaCombat.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct CombatConfigDevice {
    int attack_range{};
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

__device__ int wrapped_distance(int width, int height, Location from, Location to) {
    int dx = from.x > to.x ? from.x - to.x : to.x - from.x;
    int dy = from.y > to.y ? from.y - to.y : to.y - from.y;
    const int wx = width - dx;
    const int wy = height - dy;
    return (dx < wx ? dx : wx) + (dy < wy ? dy : wy);
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

__device__ int find_enemy_entity_index(const EntityFrameEntry *entities,
                                       std::size_t entity_count,
                                       Entity::id_type entity_id,
                                       Player::id_type player_id) {
    for (std::size_t index = 0; index < entity_count; ++index) {
        if (entities[index].id.value == entity_id.value &&
            entities[index].owner.value != player_id.value &&
            entities[index].alive) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

__global__ void combat_decision_kernel(const FlatCommand *commands,
                                       std::size_t command_count,
                                       const EntityFrameEntry *entities,
                                       std::size_t entity_count,
                                       const Player::id_type *cell_owner,
                                       int width,
                                       int height,
                                       CombatConfigDevice config,
                                       CombatFrameDecision *decisions) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count) {
        return;
    }

    const auto command = commands[index];
    CombatFrameDecision decision{command.player,
                                 command.entity,
                                 command.target_location,
                                 command.target,
                                 command.target_player,
                                 command.target_location,
                                 command.type == FlatCommandType::AttackShip || command.type == FlatCommandType::AttackStructure,
                                 command.type == FlatCommandType::AttackStructure,
                                 false,
                                 false,
                                 false};
    if (!decision.command) {
        decisions[index] = decision;
        return;
    }

    const auto attacker_index = find_owned_entity_index(entities, entity_count, command.entity, command.player);
    if (attacker_index < 0) {
        decisions[index] = decision;
        return;
    }
    const auto attacker = entities[attacker_index];
    decision.attacker_location = attacker.location;

    if (decision.structure_target) {
        if (command.target_player.value == Player::None.value || command.target_player.value == command.player.value) {
            decision.invalid = true;
            decisions[index] = decision;
            return;
        }
        const auto target_index = static_cast<std::size_t>(command.target_location.y * width + command.target_location.x);
        if (cell_owner[target_index].value != command.target_player.value ||
            wrapped_distance(width, height, attacker.location, command.target_location) != 1) {
            decision.invalid = true;
            decisions[index] = decision;
            return;
        }
        decision.valid_structure = true;
        decisions[index] = decision;
        return;
    }

    const auto target_index = find_enemy_entity_index(entities, entity_count, command.target, command.player);
    if (target_index < 0) {
        decision.invalid = true;
        decisions[index] = decision;
        return;
    }
    const auto target = entities[target_index];
    decision.target_owner = target.owner;
    decision.target_location = target.location;
    if (wrapped_distance(width, height, attacker.location, target.location) > config.attack_range) {
        decision.invalid = true;
        decisions[index] = decision;
        return;
    }
    decision.valid_ship = true;
    decisions[index] = decision;
}

} // namespace

std::vector<CombatFrameDecision> run_cuda_combat_decisions(const StateFrame &state_frame,
                                                           const CommandFrame &commands,
                                                           const GameConfig &config) {
    std::vector<CombatFrameDecision> decisions;
    decisions.reserve(commands.commands.size());
    for (std::size_t index = 0; index < commands.commands.size(); ++index) {
        decisions.push_back(CombatFrameDecision{Player::None,
                                               Entity::None,
                                               Location{0, 0},
                                               Entity::None,
                                               Player::None,
                                               Location{0, 0},
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
    DeviceBuffer<EntityFrameEntry> entity_buffer(state_frame.entities);
    DeviceBuffer<Player::id_type> cell_owner_buffer(state_frame.cell_owner);
    DeviceBuffer<CombatFrameDecision> decision_buffer(decisions);

    CombatConfigDevice device_config;
    device_config.attack_range = config.ruleset.combat.attack_range;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
    combat_decision_kernel<<<blocks, threads_per_block>>>(command_buffer.data(),
                                                          commands.commands.size(),
                                                          entity_buffer.data(),
                                                          state_frame.entities.size(),
                                                          cell_owner_buffer.data(),
                                                          state_frame.width,
                                                          state_frame.height,
                                                          device_config,
                                                          decision_buffer.data());
    check_cuda(cudaGetLastError(), "combat_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "combat_decision_kernel sync");
    check_cuda(cudaMemcpy(decisions.data(),
                          decision_buffer.data(),
                          decisions.size() * sizeof(CombatFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy combat decisions D2H");
    return decisions;
}

} // namespace hlt::gpu
