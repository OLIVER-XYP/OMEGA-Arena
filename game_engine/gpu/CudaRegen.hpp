#ifndef CUDAREGEN_HPP
#define CUDAREGEN_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_regen(StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAREGEN_HPP
