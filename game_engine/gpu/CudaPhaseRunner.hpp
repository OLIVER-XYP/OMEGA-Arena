#ifndef CUDAPHASERUNNER_HPP
#define CUDAPHASERUNNER_HPP

#include "GameConfig.hpp"
#include "GameState.hpp"
#include "CommandFrame.hpp"
#include "StepResult.hpp"

namespace hlt::gpu {

void run_cuda_inspiration_phase(GameState &state, const GameConfig &config);
void run_cuda_defend_phase(GameState &state, const CommandFrame &commands, const GameConfig &config);
void run_cuda_dump_phase(GameState &state, const GameConfig &config);
void run_cuda_halite_rebalance_phase(GameState &state, const GameConfig &config);
void run_cuda_mining_phase(GameState &state, const StepResult &result, const GameConfig &config);
void run_cuda_plunder_phase(GameState &state, const GameConfig &config);
void run_cuda_regen_phase(GameState &state, const GameConfig &config);
void run_cuda_over_ship_tax_phase(GameState &state, const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAPHASERUNNER_HPP
