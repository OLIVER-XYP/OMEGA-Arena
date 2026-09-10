#include "CudaCapture.hpp"

#include <stdexcept>

namespace hlt::gpu {

std::vector<CaptureFrameDecision> run_cuda_capture_decisions(const StateFrame &state_frame, const GameConfig &config) {
    (void)state_frame;
    (void)config;
    throw std::runtime_error("CUDA capture kernel was not built");
}

} // namespace hlt::gpu
