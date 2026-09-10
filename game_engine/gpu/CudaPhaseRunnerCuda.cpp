#include "CudaPhaseRunner.hpp"

#include "CudaDefend.hpp"
#include "CudaDump.hpp"
#include "CudaHaliteRebalance.hpp"
#include "CudaInspiration.hpp"
#include "CudaMining.hpp"
#include "CudaOverShipTax.hpp"
#include "CudaPlunder.hpp"
#include "CudaRegen.hpp"
#include "StateFrame.hpp"
#include "TurnFrame.hpp"

namespace hlt::gpu {

void run_cuda_inspiration_phase(GameState &state, const GameConfig &config) {
    auto frame = capture_state_frame(state);
    run_cuda_inspiration(frame, config.ruleset.inspiration);
    apply_state_frame(state, frame);
}

void run_cuda_defend_phase(GameState &state, const CommandFrame &commands, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_defend(state_frame, commands, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_dump_phase(GameState &state, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_dump(state_frame, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_halite_rebalance_phase(GameState &state, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_halite_rebalance(state_frame, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_mining_phase(GameState &state, const StepResult &result, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    const auto turn_frame = capture_turn_frame(result);
    run_cuda_mining(state_frame, turn_frame, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_plunder_phase(GameState &state, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_plunder(state_frame, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_regen_phase(GameState &state, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_regen(state_frame, config);
    apply_state_frame(state, state_frame);
}

void run_cuda_over_ship_tax_phase(GameState &state, const GameConfig &config) {
    auto state_frame = capture_state_frame(state);
    run_cuda_over_ship_tax(state_frame, config);
    apply_state_frame(state, state_frame);
}

} // namespace hlt::gpu
