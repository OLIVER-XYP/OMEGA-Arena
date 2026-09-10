#ifndef CUDADEFEND_HPP
#define CUDADEFEND_HPP

#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_defend(StateFrame &state_frame, const CommandFrame &commands, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDADEFEND_HPP
