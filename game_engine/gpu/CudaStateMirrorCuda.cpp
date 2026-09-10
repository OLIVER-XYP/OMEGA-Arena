#include "CudaStateMirror.hpp"

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
void upload_vector(T *&device_ptr, std::size_t &capacity, const std::vector<T> &host) {
    if (host.size() > capacity) {
        if (device_ptr != nullptr) {
            check_cuda(cudaFree(device_ptr), "cudaFree");
        }
        capacity = host.size();
        if (capacity > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_ptr), capacity * sizeof(T)), "cudaMalloc");
        } else {
            device_ptr = nullptr;
        }
    }
    if (!host.empty()) {
        check_cuda(cudaMemcpy(device_ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
}

template <typename T>
void download_vector(std::vector<T> &host, const T *device_ptr) {
    if (!host.empty()) {
        check_cuda(cudaMemcpy(host.data(), device_ptr, host.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
}

} // namespace

class CudaStateMirror::Impl {
public:
    StateFrame host_frame;
    energy_type *cell_energy{};
    energy_type *cell_initial_energy{};
    Entity::id_type *cell_entity{};
    Player::id_type *cell_owner{};
    EntityFrameEntry *entities{};
    PlayerFrameEntry *players{};
    Entity::id_type *player_entity_ids{};
    DropoffFrameEntry *dropoffs{};
    std::size_t cell_energy_capacity{};
    std::size_t cell_initial_energy_capacity{};
    std::size_t cell_entity_capacity{};
    std::size_t cell_owner_capacity{};
    std::size_t entities_capacity{};
    std::size_t players_capacity{};
    std::size_t player_entity_ids_capacity{};
    std::size_t dropoffs_capacity{};

    ~Impl() {
        cudaFree(cell_energy);
        cudaFree(cell_initial_energy);
        cudaFree(cell_entity);
        cudaFree(cell_owner);
        cudaFree(entities);
        cudaFree(players);
        cudaFree(player_entity_ids);
        cudaFree(dropoffs);
    }
};

CudaStateMirror::CudaStateMirror() : impl(std::make_unique<Impl>()) {}

CudaStateMirror::~CudaStateMirror() = default;

void CudaStateMirror::upload(const StateFrame &frame) {
    impl->host_frame = frame;
    upload_vector(impl->cell_energy, impl->cell_energy_capacity, impl->host_frame.cell_energy);
    upload_vector(impl->cell_initial_energy, impl->cell_initial_energy_capacity, impl->host_frame.cell_initial_energy);
    upload_vector(impl->cell_entity, impl->cell_entity_capacity, impl->host_frame.cell_entity);
    upload_vector(impl->cell_owner, impl->cell_owner_capacity, impl->host_frame.cell_owner);
    upload_vector(impl->entities, impl->entities_capacity, impl->host_frame.entities);
    upload_vector(impl->players, impl->players_capacity, impl->host_frame.players);
    upload_vector(impl->player_entity_ids, impl->player_entity_ids_capacity, impl->host_frame.player_entity_ids);
    upload_vector(impl->dropoffs, impl->dropoffs_capacity, impl->host_frame.dropoffs);
}

StateFrame CudaStateMirror::download() const {
    auto frame = impl->host_frame;
    download_vector(frame.cell_energy, impl->cell_energy);
    download_vector(frame.cell_initial_energy, impl->cell_initial_energy);
    download_vector(frame.cell_entity, impl->cell_entity);
    download_vector(frame.cell_owner, impl->cell_owner);
    download_vector(frame.entities, impl->entities);
    download_vector(frame.players, impl->players);
    download_vector(frame.player_entity_ids, impl->player_entity_ids);
    download_vector(frame.dropoffs, impl->dropoffs);
    return frame;
}

bool CudaStateMirror::has_device_storage() const {
    return impl->cell_energy != nullptr || impl->entities != nullptr || impl->players != nullptr;
}

std::size_t CudaStateMirror::cell_count() const {
    return impl->host_frame.cell_energy.size();
}

std::size_t CudaStateMirror::entity_count() const {
    return impl->host_frame.entities.size();
}

std::size_t CudaStateMirror::player_count() const {
    return impl->host_frame.players.size();
}

} // namespace hlt::gpu
