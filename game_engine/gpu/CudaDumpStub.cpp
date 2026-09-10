#include "CudaDump.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_dump(StateFrame &state_frame, const GameConfig &config) {
    (void)state_frame;
    (void)config;
    throw std::runtime_error("CUDA dump kernel was not built");
}

} // namespace hlt::gpu
