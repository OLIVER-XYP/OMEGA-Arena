#include "CudaOverShipTax.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct OverShipTaxConfigDevice {
    bool tax_on{};
    bool income_on{};
    bool quad_on{};
    unsigned long over_ship_tax_threshold{};
    energy_type over_ship_tax_per_turn{};
    energy_type ship_income_per_turn{};
    unsigned long ship_count_target{};
    energy_type ship_count_deviation_penalty{};
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
};

__device__ energy_type max_energy_type(energy_type a, energy_type b) {
    return a > b ? a : b;
}

__global__ void over_ship_tax_kernel(PlayerFrameEntry *players,
                                     std::size_t player_count,
                                     OverShipTaxConfigDevice config) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= player_count) {
        return;
    }
    auto player = players[index];
    const auto n = player.ship_count;

    if (config.income_on && n > 0 && !player.factory_destroyed) {
        const energy_type income = static_cast<energy_type>(n) * config.ship_income_per_turn;
        player.factory_halite += income;
    }

    if (config.tax_on && n > config.over_ship_tax_threshold) {
        const auto excess = n - config.over_ship_tax_threshold;
        const energy_type tax = static_cast<energy_type>(excess) * config.over_ship_tax_per_turn;
        player.energy = max_energy_type(0, player.energy - tax);
        if (!player.factory_destroyed) {
            player.factory_halite = max_energy_type(1, player.factory_halite - tax);
        }
    }

    if (config.quad_on) {
        const long long target = static_cast<long long>(config.ship_count_target);
        const long long actual = static_cast<long long>(n);
        if (actual > target) {
            const long long excess = actual - target;
            const long long penalty_ll = static_cast<long long>(config.ship_count_deviation_penalty) * excess * excess;
            const energy_type penalty = static_cast<energy_type>(penalty_ll);
            if (penalty > 0) {
                player.energy = max_energy_type(0, player.energy - penalty);
                if (!player.factory_destroyed) {
                    player.factory_halite = max_energy_type(1, player.factory_halite - penalty);
                }
            }
        }
    }

    players[index] = player;
}

} // namespace

void run_cuda_over_ship_tax(StateFrame &state_frame, const GameConfig &config) {
    if (state_frame.players.empty()) {
        return;
    }

    OverShipTaxConfigDevice device_config;
    device_config.tax_on = config.ruleset.economy.over_ship_tax_per_turn > 0;
    device_config.income_on = config.ruleset.economy.ship_income_per_turn > 0;
    device_config.quad_on = config.ruleset.economy.ship_count_deviation_penalty > 0;
    if (!device_config.tax_on && !device_config.income_on && !device_config.quad_on) {
        return;
    }
    device_config.over_ship_tax_threshold = config.ruleset.economy.over_ship_tax_threshold;
    device_config.over_ship_tax_per_turn = config.ruleset.economy.over_ship_tax_per_turn;
    device_config.ship_income_per_turn = config.ruleset.economy.ship_income_per_turn;
    device_config.ship_count_target = config.ruleset.economy.ship_count_target;
    device_config.ship_count_deviation_penalty = config.ruleset.economy.ship_count_deviation_penalty;

    DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.players.size() + threads_per_block - 1) / threads_per_block);
    over_ship_tax_kernel<<<blocks, threads_per_block>>>(players.data(), state_frame.players.size(), device_config);
    check_cuda(cudaGetLastError(), "over_ship_tax_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "over_ship_tax_kernel sync");
    check_cuda(cudaMemcpy(state_frame.players.data(),
                          players.data(),
                          state_frame.players.size() * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy over_ship_tax players D2H");
}

} // namespace hlt::gpu
