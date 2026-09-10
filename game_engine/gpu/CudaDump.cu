#include "CudaDump.hpp"

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

__global__ void dump_kernel(EntityFrameEntry *entities,
                            std::size_t entity_count,
                            const PlayerFrameEntry *players,
                            std::size_t player_count,
                            const DropoffFrameEntry *dropoffs,
                            const Player::id_type *cell_owner,
                            energy_type *deposits,
                            int *deposit_player_indices,
                            int *deposit_dropoff_indices,
                            unsigned char *deposit_to_factory,
                            int width,
                            int max_hp) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entity_count) {
        return;
    }

    auto entity = entities[index];
    if (!entity.alive) {
        return;
    }

    const auto cell_index = static_cast<std::size_t>(entity.location.y * width + entity.location.x);
    if (cell_owner[cell_index].value != entity.owner.value) {
        return;
    }

    const auto deposited = entity.energy;
    const auto player_index = find_player_index(players, player_count, entity.owner);
    if (player_index < 0) {
        return;
    }

    entity.energy = 0;
    entity.lifetime_deposited += deposited;
    entity.hp = max_hp;
    deposits[index] = deposited;
    deposit_player_indices[index] = player_index;
    if (entity.location.x == players[player_index].factory.x &&
        entity.location.y == players[player_index].factory.y) {
        deposit_to_factory[index] = 1;
    } else {
        const auto dropoff_end = players[player_index].dropoff_begin + players[player_index].dropoff_count;
        for (std::size_t dropoff_index = players[player_index].dropoff_begin; dropoff_index < dropoff_end; ++dropoff_index) {
            if (dropoffs[dropoff_index].location.x == entity.location.x &&
                dropoffs[dropoff_index].location.y == entity.location.y) {
                deposit_dropoff_indices[index] = static_cast<int>(dropoff_index);
                break;
            }
        }
    }

    entities[index] = entity;
}

__global__ void apply_dump_deposits_kernel(PlayerFrameEntry *players,
                                           DropoffFrameEntry *dropoffs,
                                           const energy_type *deposits,
                                           const int *deposit_player_indices,
                                           const int *deposit_dropoff_indices,
                                           const unsigned char *deposit_to_factory,
                                           std::size_t entity_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (std::size_t index = 0; index < entity_count; ++index) {
        const auto deposited = deposits[index];
        if (deposited <= 0 || deposit_player_indices[index] < 0) {
            continue;
        }
        const auto player_index = static_cast<std::size_t>(deposit_player_indices[index]);
        auto player = players[player_index];
        player.energy += deposited;
        player.total_energy_deposited += deposited;

        if (deposit_to_factory[index] != 0) {
            player.factory_energy_deposited += deposited;
            if (!player.factory_destroyed) {
                player.factory_halite += deposited;
            }
        } else if (deposit_dropoff_indices[index] >= 0) {
            const auto dropoff_index = static_cast<std::size_t>(deposit_dropoff_indices[index]);
            auto dropoff = dropoffs[dropoff_index];
            dropoff.deposited_halite += deposited;
            if (!dropoff.destroyed) {
                dropoff.halite_pool += deposited;
            }
            dropoffs[dropoff_index] = dropoff;
        }

        players[player_index] = player;
    }
}

} // namespace

void run_cuda_dump(StateFrame &state_frame, const GameConfig &config) {
    if (state_frame.entities.empty()) {
        return;
    }

    DeviceBuffer<EntityFrameEntry> entities(state_frame.entities);
    DeviceBuffer<PlayerFrameEntry> players(state_frame.players);
    DeviceBuffer<DropoffFrameEntry> dropoffs(state_frame.dropoffs);
    DeviceBuffer<Player::id_type> cell_owner(state_frame.cell_owner);

    std::vector<energy_type> zero_deposits(state_frame.entities.size(), 0);
    std::vector<int> no_player(state_frame.entities.size(), -1);
    std::vector<int> no_dropoff(state_frame.entities.size(), -1);
    std::vector<unsigned char> no_factory(state_frame.entities.size(), 0);
    DeviceBuffer<energy_type> deposits(zero_deposits);
    DeviceBuffer<int> deposit_player_indices(no_player);
    DeviceBuffer<int> deposit_dropoff_indices(no_dropoff);
    DeviceBuffer<unsigned char> deposit_to_factory(no_factory);

    constexpr int threads_per_block = 128;
    const auto blocks = static_cast<int>((state_frame.entities.size() + threads_per_block - 1) / threads_per_block);
    dump_kernel<<<blocks, threads_per_block>>>(entities.data(),
                                               state_frame.entities.size(),
                                               players.data(),
                                               state_frame.players.size(),
                                               dropoffs.data(),
                                               cell_owner.data(),
                                               deposits.data(),
                                               deposit_player_indices.data(),
                                               deposit_dropoff_indices.data(),
                                               deposit_to_factory.data(),
                                               state_frame.width,
                                               config.ruleset.combat.initial_hp);
    check_cuda(cudaGetLastError(), "dump_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "dump_kernel sync");
    apply_dump_deposits_kernel<<<1, 1>>>(players.data(),
                                         dropoffs.data(),
                                         deposits.data(),
                                         deposit_player_indices.data(),
                                         deposit_dropoff_indices.data(),
                                         deposit_to_factory.data(),
                                         state_frame.entities.size());
    check_cuda(cudaGetLastError(), "apply_dump_deposits_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "apply_dump_deposits_kernel sync");
    check_cuda(cudaMemcpy(state_frame.entities.data(),
                          entities.data(),
                          state_frame.entities.size() * sizeof(EntityFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy dump entities D2H");
    check_cuda(cudaMemcpy(state_frame.players.data(),
                          players.data(),
                          state_frame.players.size() * sizeof(PlayerFrameEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy dump players D2H");
    if (!state_frame.dropoffs.empty()) {
        check_cuda(cudaMemcpy(state_frame.dropoffs.data(),
                              dropoffs.data(),
                              state_frame.dropoffs.size() * sizeof(DropoffFrameEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy dump dropoffs D2H");
    }
}

} // namespace hlt::gpu
