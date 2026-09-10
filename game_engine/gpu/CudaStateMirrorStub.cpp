#include "CudaStateMirror.hpp"

namespace hlt::gpu {

class CudaStateMirror::Impl {
public:
    StateFrame host_frame;
};

CudaStateMirror::CudaStateMirror() : impl(std::make_unique<Impl>()) {}

CudaStateMirror::~CudaStateMirror() = default;

void CudaStateMirror::upload(const StateFrame &frame) {
    impl->host_frame = frame;
}

StateFrame CudaStateMirror::download() const {
    return impl->host_frame;
}

bool CudaStateMirror::has_device_storage() const {
    return false;
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
