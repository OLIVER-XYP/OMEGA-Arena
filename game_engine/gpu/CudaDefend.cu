#include "CudaDefend.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

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
};

__global__ void defend_reset_kernel(EntityFrameEntry *entities, std::size_t entity_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count) {
        return;
    }
    if (entities[index].protection_turns > 0) {
        entities[index].protection_turns -= 1;
        entities[index].is_defending = true;
    } else {
        entities[index].is_defending = false;
    }
}

__global__ void defend_command_kernel(EntityFrameEntry *entities,
                                      std::size_t entity_count,
                                      const FlatCommand *commands,
                                      std::size_t command_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= command_count || commands[index].type != FlatCommandType::Defend) {
        return;
    }
    const auto entity_id = commands[index].entity;
    const auto player_id = commands[index].player;
    for (std::size_t entity_index = 0; entity_index < entity_count; ++entity_index) {
        if (entities[entity_index].id.value == entity_id.value &&
            entities[entity_index].owner.value == player_id.value &&
            entities[entity_index].alive) {
            entities[entity_index].is_defending = true;
            return;
        }
    }
}

} // namespace

void run_cuda_defend(StateFrame &state_frame, const CommandFrame &commands, const GameConfig &config) {
    (void)config;
    if (state_frame.entities.empty()) {
        return;
    }
    DeviceBuffer<EntityFrameEntry> entities(state_frame.entities);
    DeviceBuffer<FlatCommand> command_buffer(commands.commands);

    constexpr int threads_per_block = 128;
    const auto entity_blocks = static_cast<int>((state_frame.entities.size() + threads_per_block - 1) / threads_per_block);
    defend_reset_kernel<<<entity_blocks, threads_per_block>>>(entities.data(), state_frame.entities.size());
    check_cuda(cudaGetLastError(), "defend_reset_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "defend_reset_kernel sync");

    if (!commands.commands.empty()) {
        const auto command_blocks = static_cast<int>((commands.commands.size() + threads_per_block - 1) / threads_per_block);
        defend_command_kernel<<<command_blocks, threads_per_block>>>(entities.data(),
                                                                     state_frame.entities.size(),
                                                                     command_buffer.data(),
                                                                     commands.commands.size());
        check_cuda(cudaGetLastError(), "defend_command_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "defend_command_kernel sync");
    }

    check_cuda(cudaMemcpy(state_frame.entities.data(),
                          entities.data(),
                          state_frame.entities.size() * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy defend entities D2H");
}

} // namespace hlt::gpu
