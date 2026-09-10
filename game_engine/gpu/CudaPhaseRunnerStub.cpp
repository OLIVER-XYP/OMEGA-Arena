#include "CudaPhaseRunner.hpp"

#include <stdexcept>

namespace hlt::gpu {

void run_cuda_inspiration_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_defend_phase(GameState &state, const CommandFrame &commands, const GameConfig &config) {
    (void)state;
    (void)commands;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_dump_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_halite_rebalance_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_mining_phase(GameState &state, const StepResult &result, const GameConfig &config) {
    (void)state;
    (void)result;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_plunder_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_regen_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

void run_cuda_over_ship_tax_phase(GameState &state, const GameConfig &config) {
    (void)state;
    (void)config;
    throw std::runtime_error("CUDA phase runner was not built");
}

} // namespace hlt::gpu
