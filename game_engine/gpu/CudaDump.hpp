#ifndef CUDADUMP_HPP
#define CUDADUMP_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_dump(StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDADUMP_HPP
