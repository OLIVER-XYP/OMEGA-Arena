#ifndef CUDACAPTURE_HPP
#define CUDACAPTURE_HPP

#include <vector>

#include "GameConfig.hpp"
#include "StateFrame.hpp"

namespace hlt::gpu {

struct CaptureFrameDecision {
    Location location;
    Entity::id_type entity;
    Player::id_type old_owner;
    Player::id_type new_owner;
    bool should_capture;
};

std::vector<CaptureFrameDecision> run_cuda_capture_decisions(const StateFrame &state_frame, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDACAPTURE_HPP
