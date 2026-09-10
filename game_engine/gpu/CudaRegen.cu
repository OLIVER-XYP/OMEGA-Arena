#include "CudaRegen.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct RegenConfigDevice {
    bool enabled{};
    double rate{};
    double cap_fraction{};
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

__global__ void regen_kernel(energy_type *cell_energy,
                             const energy_type *cell_initial_energy,
                             const Player::id_type *cell_owner,
                             std::size_t cell_count,
                             unsigned long long *map_total_energy_delta,
                             RegenConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= cell_count || !config.enabled || config.rate <= 0.0 || config.cap_fraction <= 0.0) {
        return;
    }
    if (cell_owner[index].value != Player::None.value || cell_initial_energy[index] <= 0) {
        return;
    }
    const auto cap = static_cast<energy_type>(config.cap_fraction * static_cast<double>(cell_initial_energy[index]));
    if (cell_energy[index] >= cap) {
        return;
    }
    const auto regen = static_cast<energy_type>(ceil(config.rate * static_cast<double>(cell_initial_energy[index])));
    if (regen <= 0) {
        return;
    }
    const auto candidate = cell_energy[index] + regen;
    const auto new_energy = candidate < cap ? candidate : cap;
    const auto delta = new_energy - cell_energy[index];
    if (delta <= 0) {
        return;
    }
    cell_energy[index] = new_energy;
    atomicAdd(map_total_energy_delta, static_cast<unsigned long long>(delta));
}

} // namespace

void run_cuda_regen(StateFrame &state_frame, const GameConfig &config) {
    if (state_frame.cell_energy.empty()) {
        return;
    }

    DeviceBuffer<energy_type> cell_energy(state_frame.cell_energy);
    DeviceBuffer<energy_type> cell_initial_energy(state_frame.cell_initial_energy);
    DeviceBuffer<Player::id_type> cell_owner(state_frame.cell_owner);

    unsigned long long *device_delta{};
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_delta), sizeof(unsigned long long)), "cudaMalloc regen delta");
    check_cuda(cudaMemset(device_delta, 0, sizeof(unsigned long long)), "cudaMemset regen delta");

    RegenConfigDevice device_config;
    device_config.enabled = config.ruleset.economy.cell_regen_enabled;
    device_config.rate = config.ruleset.economy.cell_regen_rate;
    device_config.cap_fraction = config.ruleset.economy.cell_regen_cap_fraction;

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.cell_energy.size() + threads_per_block - 1) / threads_per_block);
    regen_kernel<<<blocks, threads_per_block>>>(cell_energy.data(),
                                                cell_initial_energy.data(),
                                                cell_owner.data(),
                                                state_frame.cell_energy.size(),
                                                device_delta,
                                                device_config);
    check_cuda(cudaGetLastError(), "regen_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "regen_kernel sync");

    unsigned long long delta{};
    check_cuda(cudaMemcpy(&delta, device_delta, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy regen delta D2H");
    check_cuda(cudaFree(device_delta), "cudaFree regen delta");
    check_cuda(cudaMemcpy(state_frame.cell_energy.data(),
                          cell_energy.data(),
                          state_frame.cell_energy.size() * sizeof(energy_type),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy regen cells D2H");
    state_frame.map_total_energy += delta;
}

} // namespace hlt::gpu
