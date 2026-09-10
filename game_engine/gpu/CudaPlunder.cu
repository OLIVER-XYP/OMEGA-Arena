#include "CudaPlunder.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

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

} // namespace

void run_cuda_plunder(StateFrame &state_frame, const GameConfig &config) {
    if (state_frame.entities.empty()) {
        return;
    }

    PlunderConfigDevice device_config;
    device_config.range = config.ruleset.economy.plunder_range;
    device_config.halite_per_turn = config.ruleset.economy.plunder_halite_per_turn;
    device_config.max_energy = config.ruleset.economy.max_energy;
    if (device_config.range <= 0 || device_config.halite_per_turn <= 0) {
        return;
    }

    DeviceBuffer<EntityFrameEntry> entities(state_frame.entities);
    DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
    DeviceBuffer<DropoffFrameEntry> dropoffs(state_frame.dropoffs);

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.entities.size() + threads_per_block - 1) / threads_per_block);
    plunder_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                                  state_frame.entities.size(),
                                                  players.data(),
                                                  state_frame.players.size(),
                                                  dropoffs.data(),
                                                  state_frame.width,
                                                  state_frame.height,
                                                  device_config);
    check_cuda(cudaGetLastError(), "plunder_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "plunder_kernel sync");
    check_cuda(cudaMemcpy(state_frame.entities.data(),
                          entities.data(),
                          state_frame.entities.size() * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy plunder entities D2H");
}

} // namespace hlt::gpu
