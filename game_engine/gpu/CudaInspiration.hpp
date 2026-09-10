#ifndef CUDAINSPIRATION_HPP
#define CUDAINSPIRATION_HPP

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

void run_cuda_inspiration(StateFrame &frame, const InspirationConfig &config);

} // namespace hlt::gpu

#endif // CUDAINSPIRATION_HPP
