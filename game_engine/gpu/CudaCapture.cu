#include "CudaCapture.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct CaptureConfigDevice {
    bool enabled{};
    dimension_type radius{};
    unsigned long ships_above_for_capture{};
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

__device__ Player::id_type find_entity_owner(const EntityFrameEntry *entities,
                                             std::size_t entity_count,
                                             Entity::id_type entity_id) {
    for (std::size_t index = 0; index < entity_count; ++index) {
        if (entities[index].id.value == entity_id.value && entities[index].alive) {
            return entities[index].owner;
        }
    }
    return Player::None;
}

__global__ void capture_decision_kernel(const EntityFrameEntry *entities,
                                        std::size_t entity_count,
                                        const Entity::id_type *cell_entity,
                                        const PlayerFrameEntry *players,
                                        std::size_t player_count,
                                        CaptureFrameDecision *decisions,
                                        int width,
                                        int height,
                                        CaptureConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count) {
        return;
    }

    const auto entity = entities[index];
    CaptureFrameDecision decision{entity.location, entity.id, entity.owner, Player::None, false};
    if (!config.enabled || !entity.alive) {
        decisions[index] = decision;
        return;
    }

    const auto own_player_index = find_player_index(players, player_count, entity.owner);
    if (own_player_index < 0) {
        decisions[index] = decision;
        return;
    }

    unsigned long own_count = 0;
    unsigned long max_opponent_count = 0;
    Player::id_type max_opponent = Player::None;

    for (std::size_t player_index = 0; player_index < player_count; ++player_index) {
        unsigned long count = 0;
        const auto player_id = players[player_index].id;
        for (int dy = -static_cast<int>(config.radius); dy <= static_cast<int>(config.radius); ++dy) {
            for (int dx = -static_cast<int>(config.radius); dx <= static_cast<int>(config.radius); ++dx) {
                const int cur_x = (entity.location.x + dx + width) % width;
                const int cur_y = (entity.location.y + dy + height) % height;
                if (wrapped_distance(width, height, entity.location.x, entity.location.y, cur_x, cur_y) > config.radius) {
                    continue;
                }
                const auto cell_index_value = static_cast<std::size_t>(cur_y * width + cur_x);
                const auto occupant = cell_entity[cell_index_value];
                if (occupant.value == Entity::None.value) {
                    continue;
                }
                const auto owner = find_entity_owner(entities, entity_count, occupant);
                if (owner.value == player_id.value) {
                    ++count;
                }
            }
        }

        if (player_index == static_cast<std::size_t>(own_player_index)) {
            own_count = count;
        } else if (count > max_opponent_count) {
            max_opponent_count = count;
            max_opponent = player_id;
        }
    }

    if (own_count + config.ships_above_for_capture <= max_opponent_count) {
        decision.should_capture = true;
        decision.new_owner = max_opponent;
    }
    decisions[index] = decision;
}

} // namespace

std::vector<CaptureFrameDecision> run_cuda_capture_decisions(const StateFrame &state_frame, const GameConfig &config) {
    std::vector<CaptureFrameDecision> decisions;
    decisions.reserve(state_frame.entities.size());
    for (std::size_t index = 0; index < state_frame.entities.size(); ++index) {
        decisions.push_back(CaptureFrameDecision{Location{0, 0}, Entity::None, Player::None, Player::None, false});
    }
    if (!config.ruleset.capture.enabled || state_frame.entities.empty()) {
        return decisions;
    }

    DeviceBuffer<EntityFrameEntry> entities(state_frame.entities);
    DeviceBuffer<Entity::id_type> cell_entity(state_frame.cell_entity);
    DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
    DeviceBuffer<CaptureFrameDecision> device_decisions(decisions);

    CaptureConfigDevice device_config;
    device_config.enabled = config.ruleset.capture.enabled;
    device_config.radius = config.ruleset.capture.radius;
    device_config.ships_above_for_capture = config.ruleset.capture.ships_above_for_capture;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.entities.size() + threads_per_block - 1) / threads_per_block);
    capture_decision_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                                           state_frame.entities.size(),
                                                           cell_entity.data(),
                                                           players.data(),
                                                           state_frame.players.size(),
                                                           device_decisions.data(),
                                                           state_frame.width,
                                                           state_frame.height,
                                                           device_config);
    check_cuda(cudaGetLastError(), "capture_decision_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "capture_decision_kernel sync");
    check_cuda(cudaMemcpy(decisions.data(),
                          device_decisions.data(),
                          decisions.size() * sizeof(CaptureFrameDecision),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy capture decisions D2H");
    return decisions;
}

} // namespace hlt::gpu
