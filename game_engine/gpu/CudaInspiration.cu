#include "CudaInspiration.hpp"

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
    const T *data() const { return ptr; }
};

__device__ int wrapped_distance(int width, int height, int ax, int ay, int bx, int by) {
    int dx = ax > bx ? ax - bx : bx - ax;
    int dy = ay > by ? ay - by : by - ay;
    const int wx = width - dx;
    const int wy = height - dy;
    return (dx < wx ? dx : wx) + (dy < wy ? dy : wy);
}

__global__ void inspiration_kernel(EntityFrameEntry *entities,
                                   std::size_t entity_count,
                                   const Entity::id_type *cell_entity,
                                   const EntityFrameEntry *entity_lookup,
                                   int width,
                                   int height,
                                   int radius,
                                   unsigned long ship_count_threshold) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive) {
        return;
    }

    unsigned long opponents = 0;
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            const int cur_x = (entity.location.x + dx + width) % width;
            const int cur_y = (entity.location.y + dy + height) % height;
            if (wrapped_distance(width, height, entity.location.x, entity.location.y, cur_x, cur_y) > radius) {
                continue;
            }
            const auto cell_index = static_cast<std::size_t>(cur_y * width + cur_x);
            const auto other_id = cell_entity[cell_index];
            if (other_id.value == Entity::None.value) {
                continue;
            }
            for (std::size_t other_index = 0; other_index < entity_count; ++other_index) {
                const auto other = entity_lookup[other_index];
                if (other.id.value == other_id.value && other.alive && other.owner.value != entity.owner.value) {
                    ++opponents;
                    break;
                }
            }
        }
    }

    entities[index].is_inspired = opponents >= ship_count_threshold;
}

} // namespace

void run_cuda_inspiration(StateFrame &frame, const InspirationConfig &config) {
    if (!config.enabled || frame.entities.empty()) {
        return;
    }

    DeviceBuffer<EntityFrameEntry> entities(frame.entities);
    DeviceBuffer<EntityFrameEntry> entity_lookup(frame.entities);
    DeviceBuffer<Entity::id_type> cell_entity(frame.cell_entity);

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((frame.entities.size() + threads_per_block - 1) / threads_per_block);
    inspiration_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                                      frame.entities.size(),
                                                      cell_entity.data(),
                                                      entity_lookup.data(),
                                                      frame.width,
                                                      frame.height,
                                                      config.radius,
                                                      config.ship_count);
    check_cuda(cudaGetLastError(), "inspiration_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "inspiration_kernel sync");
    check_cuda(cudaMemcpy(frame.entities.data(),
                          entities.data(),
                          frame.entities.size() * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy inspiration D2H");
}

} // namespace hlt::gpu
