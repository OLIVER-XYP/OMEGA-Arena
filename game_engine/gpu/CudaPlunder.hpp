#ifndef CUDAPLUNDER_HPP
#define CUDAPLUNDER_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_plunder(StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAPLUNDER_HPP
