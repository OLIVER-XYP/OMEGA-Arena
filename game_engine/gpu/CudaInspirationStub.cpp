#include "CudaInspiration.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_inspiration(StateFrame &frame, const InspirationConfig &config) {
    (void)frame;
    (void)config;
    throw std::runtime_error("CUDA inspiration kernel was not built");
}

} // namespace hlt::gpu
