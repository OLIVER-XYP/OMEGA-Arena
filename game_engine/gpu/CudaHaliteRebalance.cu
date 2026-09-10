#include "CudaHaliteRebalance.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hlt::gpu {

namespace {

struct HaliteRebalanceConfigDevice {
    bool enabled{};
    unsigned long period{};
    double fraction{};
    double min_gap_frac{};
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

__device__ energy_type min_energy_type(energy_type a, energy_type b) {
    return a < b ? a : b;
}

__device__ energy_type max_energy_type(energy_type a, energy_type b) {
    return a > b ? a : b;
}

__global__ void halite_rebalance_kernel(PlayerFrameEntry *players,
                                        std::size_t player_count,
                                        DropoffFrameEntry *dropoffs,
                                        unsigned long turn_number,
                                        HaliteRebalanceConfigDevice config) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (!config.enabled || config.period == 0 || config.fraction <= 0.0) {
        return;
    }
    if (turn_number == 0 || turn_number % config.period != 0 || player_count == 0) {
        return;
    }

    std::size_t leader_index = 0;
    std::size_t trailer_index = 0;
    energy_type max_score = 0;
    energy_type min_score = 0;
    bool seen_any = false;

    for (std::size_t player_index = 0; player_index < player_count; ++player_index) {
        const auto player = players[player_index];
        energy_type score = player.factory_halite;
        const auto dropoff_end = player.dropoff_begin + player.dropoff_count;
        for (std::size_t dropoff_index = player.dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
            score += dropoffs[dropoff_index].halite_pool;
        }

        if (!seen_any || score > max_score) {
            max_score = score;
            leader_index = player_index;
        }
        if (!seen_any || score < min_score) {
            min_score = score;
            trailer_index = player_index;
        }
        seen_any = true;
    }

    if (!seen_any || leader_index == trailer_index || max_score <= min_score) {
        return;
    }

    if (config.min_gap_frac > 0.0 && max_score + min_score > 0) {
        const double gap_frac = static_cast<double>(max_score - min_score) /
                                static_cast<double>(max_score + min_score);
        if (gap_frac < config.min_gap_frac) {
            return;
        }
    }

    const auto gap = max_score - min_score;
    const auto transfer = static_cast<energy_type>(static_cast<double>(gap) * config.fraction * 0.5);
    if (transfer <= 0) {
        return;
    }

    auto leader = players[leader_index];
    auto trailer = players[trailer_index];
    energy_type remaining = transfer;

    if (!leader.factory_destroyed) {
        const auto drained = min_energy_type(leader.factory_halite, remaining);
        leader.factory_halite -= drained;
        remaining -= drained;
    }

    const auto leader_dropoff_end = leader.dropoff_begin + leader.dropoff_count;
    for (std::size_t dropoff_index = leader.dropoff_begin; dropoff_index < leader_dropoff_end && remaining > 0; ++dropoff_index) {
        auto dropoff = dropoffs[dropoff_index];
        if (dropoff.destroyed) {
            continue;
        }
        const auto drained = min_energy_type(dropoff.halite_pool, remaining);
        dropoff.halite_pool -= drained;
        remaining -= drained;
        dropoffs[dropoff_index] = dropoff;
    }

    const auto actually_drained = transfer - remaining;
    if (actually_drained <= 0) {
        return;
    }

    if (!trailer.factory_destroyed) {
        trailer.factory_halite += actually_drained;
    } else {
        const auto trailer_dropoff_end = trailer.dropoff_begin + trailer.dropoff_count;
        for (std::size_t dropoff_index = trailer.dropoff_begin; dropoff_index < trailer_dropoff_end; ++dropoff_index) {
            auto dropoff = dropoffs[dropoff_index];
            if (dropoff.destroyed) {
                continue;
            }
            dropoff.halite_pool += actually_drained;
            dropoffs[dropoff_index] = dropoff;
            break;
        }
    }

    leader.energy = max_energy_type(static_cast<energy_type>(0), leader.energy - actually_drained);
    trailer.energy += actually_drained;

    players[leader_index] = leader;
    players[trailer_index] = trailer;
}

} // namespace

void run_cuda_halite_rebalance(StateFrame &state_frame, const GameConfig &config) {
    if (state_frame.players.empty()) {
        return;
    }

    HaliteRebalanceConfigDevice device_config;
    device_config.enabled = config.ruleset.economy.halite_rebalance_enabled;
    device_config.period = config.ruleset.economy.halite_rebalance_period;
    device_config.fraction = config.ruleset.economy.halite_rebalance_fraction;
    device_config.min_gap_frac = config.ruleset.economy.halite_rebalance_min_gap_frac;
    if (!device_config.enabled || device_config.period == 0 || device_config.fraction <= 0.0) {
        return;
    }
    if (state_frame.turn_number == 0 || state_frame.turn_number % device_config.period != 0) {
        return;
    }

    DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
    DeviceBuffer<DropoffFrameEntry> dropoffs(state_frame.dropoffs);

    halite_rebalance_kernel<<<1, 1>>>(players.data(),
                                      state_frame.players.size(),
                                      dropoffs.data(),
                                      state_frame.turn_number,
                                      device_config);
    check_cuda(cudaGetLastError(), "halite_rebalance_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "halite_rebalance_kernel sync");
    check_cuda(cudaMemcpy(state_frame.players.data(),
                          players.data(),
                          state_frame.players.size() * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy halite rebalance players D2H");
    if (!state_frame.dropoffs.empty()) {
        check_cuda(cudaMemcpy(state_frame.dropoffs.data(),
                              dropoffs.data(),
                              state_frame.dropoffs.size() * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy halite rebalance dropoffs D2H");
    }
}

} // namespace hlt::gpu
