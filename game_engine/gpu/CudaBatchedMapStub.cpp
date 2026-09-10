#include "CudaBatchedMap.hpp"

#include <stdexcept>

namespace hlt::gpu {

BatchedRegenStats run_cuda_batched_regen(std::vector<StateFrame *> frames, const GameConfig &config) {
    (void)frames;
    (void)config;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedRegenStats run_cuda_batched_regen_iterations(std::vector<StateFrame *> frames,
                                                    const GameConfig &config,
                                                    unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedDumpStats run_cuda_batched_dump(std::vector<StateFrame *> frames,
                                       const GameConfig &config,
                                       unsigned long iterations,
                                       energy_type refill_cargo) {
    (void)frames;
    (void)config;
    (void)iterations;
    (void)refill_cargo;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMiningStats run_cuda_batched_mining(std::vector<StateFrame *> frames,
                                           const GameConfig &config,
                                           unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedSpawnStats run_cuda_batched_spawn_apply(std::vector<StateFrame *> frames,
                                               const GameConfig &config,
                                               unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMovementStats run_cuda_batched_movement_decisions(std::vector<StateFrame *> frames,
                                                         const GameConfig &config,
                                                         const std::vector<CommandFrame> &commands,
                                                         std::vector<MovementFrameDecision> &decisions_out,
                                                         unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)commands;
    (void)decisions_out;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedInspirationStats run_cuda_batched_inspiration_resident(std::vector<StateFrame *> frames,
                                                              const GameConfig &config,
                                                              unsigned long turns) {
    (void)frames;
    (void)config;
    (void)turns;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMovementApplyStats run_cuda_batched_movement_apply_no_collision(std::vector<StateFrame *> frames,
                                                                       const GameConfig &config,
                                                                       const std::vector<CommandFrame> &commands,
                                                                       unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)commands;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedDestinationStats run_cuda_batched_destination_counts(std::vector<StateFrame *> frames,
                                                           const GameConfig &config,
                                                           const std::vector<CommandFrame> &commands,
                                                           unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)commands;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMovementCollisionStats run_cuda_batched_movement_collision_apply(std::vector<StateFrame *> frames,
                                                                        const GameConfig &config,
                                                                        const std::vector<CommandFrame> &commands,
                                                                        unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)commands;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedValidationStats run_cuda_batched_validation_decisions(std::vector<StateFrame *> frames,
                                                             const GameConfig &config,
                                                             const std::vector<CommandFrame> &commands,
                                                             std::vector<ValidationFrameDecision> &decisions_out,
                                                             unsigned long iterations) {
    (void)frames;
    (void)config;
    (void)commands;
    (void)decisions_out;
    (void)iterations;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedEndEconomyStats run_cuda_batched_end_economy(std::vector<StateFrame *> frames,
                                                    const GameConfig &config) {
    (void)frames;
    (void)config;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMapPipelineStats run_cuda_batched_dump_regen_pipeline(std::vector<StateFrame *> frames,
                                                            const GameConfig &config,
                                                            unsigned long iterations,
                                                            energy_type dump_refill_cargo) {
    (void)frames;
    (void)config;
    (void)iterations;
    (void)dump_refill_cargo;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline(std::vector<StateFrame *> frames,
                                                          const GameConfig &config,
                                                          unsigned long iterations,
                                                          energy_type dump_refill_cargo,
                                                          energy_type mining_refill_cell_energy) {
    (void)frames;
    (void)config;
    (void)iterations;
    (void)dump_refill_cargo;
    (void)mining_refill_cell_energy;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline_resident(std::vector<StateFrame *> frames,
                                                                   const GameConfig &config,
                                                                   unsigned long iterations,
                                                                   energy_type dump_refill_cargo,
                                                                   energy_type mining_refill_cell_energy) {
    (void)frames;
    (void)config;
    (void)iterations;
    (void)dump_refill_cargo;
    (void)mining_refill_cell_energy;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

BatchedMapPipelineStats run_cuda_batched_economy_pipeline_device_resident(std::vector<StateFrame *> frames,
                                                                          const GameConfig &config,
                                                                          unsigned long turns,
                                                                          unsigned long iterations_per_turn,
                                                                          energy_type dump_refill_cargo,
                                                                          energy_type mining_refill_cell_energy) {
    (void)frames;
    (void)config;
    (void)turns;
    (void)iterations_per_turn;
    (void)dump_refill_cargo;
    (void)mining_refill_cell_energy;
    throw std::runtime_error("CUDA batched map kernels were not built");
}

} // namespace hlt::gpu
