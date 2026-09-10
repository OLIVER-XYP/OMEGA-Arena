#ifndef CUDABATCHEDMAP_HPP
#define CUDABATCHEDMAP_HPP

#include <vector>

#include "GameConfig.hpp"
#include "StateFrame.hpp"
#include "CudaMovement.hpp"
#include "CudaValidation.hpp"

namespace hlt::gpu {

struct BatchedRegenStats {
    std::size_t games{};
    std::size_t cells_per_game{};
    unsigned long long total_delta{};
};

struct BatchedDumpStats {
    std::size_t games{};
    std::size_t entities_per_game{};
    unsigned long long total_deposited{};
};

struct BatchedMiningStats {
    std::size_t games{};
    std::size_t cells_per_game{};
    std::size_t entities_per_game{};
    unsigned long long total_extracted{};
};

struct BatchedSpawnStats {
    std::size_t games{};
    std::size_t players_per_game{};
    std::size_t entity_capacity_per_game{};
    unsigned long long total_spawned{};
};

struct BatchedMovementStats {
    std::size_t games{};
    std::size_t commands_per_game{};
    unsigned long long movable_commands{};
};

struct BatchedInspirationStats {
    std::size_t games{};
    std::size_t entities_per_game{};
    unsigned long long inspired_entities{};
};

struct BatchedMovementApplyStats {
    std::size_t games{};
    std::size_t commands_per_game{};
    unsigned long long applied_commands{};
};

struct BatchedDestinationStats {
    std::size_t games{};
    std::size_t commands_per_game{};
    unsigned long long moved_commands{};
    unsigned long long conflict_cells{};
    unsigned long long max_arrivals{};
    unsigned long long total_arrivals{};
};

struct BatchedMovementCollisionStats {
    std::size_t games{};
    std::size_t commands_per_game{};
    unsigned long long moved_commands{};
    unsigned long long collision_cells{};
    unsigned long long damaged_entities{};
    unsigned long long deaths{};
    unsigned long long dropped_energy{};
};

struct BatchedValidationStats {
    std::size_t games{};
    std::size_t commands_per_game{};
    unsigned long long included_commands{};
    unsigned long long expense_commands{};
    unsigned long long occurrence_commands{};
};

struct BatchedEndEconomyStats {
    std::size_t games{};
    std::size_t players_per_game{};
    std::size_t dropoffs_per_game{};
};

struct BatchedMapPipelineStats {
    std::size_t games{};
    std::size_t cells_per_game{};
    std::size_t entities_per_game{};
    unsigned long long total_regen_delta{};
    unsigned long long total_deposited{};
    unsigned long long total_mining_extracted{};
};

BatchedRegenStats run_cuda_batched_regen(std::vector<StateFrame *> frames, const GameConfig &config);
BatchedRegenStats run_cuda_batched_regen_iterations(std::vector<StateFrame *> frames,
                                                    const GameConfig &config,
                                                    unsigned long iterations);
BatchedDumpStats run_cuda_batched_dump(std::vector<StateFrame *> frames,
                                       const GameConfig &config,
                                       unsigned long iterations = 1,
                                       energy_type refill_cargo = 0);
BatchedMiningStats run_cuda_batched_mining(std::vector<StateFrame *> frames,
                                           const GameConfig &config,
                                           unsigned long iterations = 1);
BatchedSpawnStats run_cuda_batched_spawn_apply(std::vector<StateFrame *> frames,
                                               const GameConfig &config,
                                               unsigned long iterations);
BatchedMovementStats run_cuda_batched_movement_decisions(std::vector<StateFrame *> frames,
                                                         const GameConfig &config,
                                                         const std::vector<CommandFrame> &commands,
                                                         std::vector<MovementFrameDecision> &decisions_out,
                                                         unsigned long iterations = 1);
BatchedInspirationStats run_cuda_batched_inspiration_resident(std::vector<StateFrame *> frames,
                                                              const GameConfig &config,
                                                              unsigned long turns = 1);
BatchedMovementApplyStats run_cuda_batched_movement_apply_no_collision(std::vector<StateFrame *> frames,
                                                                       const GameConfig &config,
                                                                       const std::vector<CommandFrame> &commands,
                                                                       unsigned long iterations = 1);
BatchedDestinationStats run_cuda_batched_destination_counts(std::vector<StateFrame *> frames,
                                                           const GameConfig &config,
                                                           const std::vector<CommandFrame> &commands,
                                                           unsigned long iterations = 1);
BatchedMovementCollisionStats run_cuda_batched_movement_collision_apply(std::vector<StateFrame *> frames,
                                                                        const GameConfig &config,
                                                                        const std::vector<CommandFrame> &commands,
                                                                        unsigned long iterations = 1);
BatchedValidationStats run_cuda_batched_validation_decisions(std::vector<StateFrame *> frames,
                                                             const GameConfig &config,
                                                             const std::vector<CommandFrame> &commands,
                                                             std::vector<ValidationFrameDecision> &decisions_out,
                                                             unsigned long iterations = 1);
BatchedEndEconomyStats run_cuda_batched_end_economy(std::vector<StateFrame *> frames,
                                                    const GameConfig &config);
BatchedMapPipelineStats run_cuda_batched_dump_regen_pipeline(std::vector<StateFrame *> frames,
                                                            const GameConfig &config,
                                                            unsigned long iterations,
                                                            energy_type dump_refill_cargo);
BatchedMapPipelineStats run_cuda_batched_economy_pipeline(std::vector<StateFrame *> frames,
                                                          const GameConfig &config,
                                                          unsigned long iterations,
                                                          energy_type dump_refill_cargo,
                                                          energy_type mining_refill_cell_energy);
BatchedMapPipelineStats run_cuda_batched_economy_pipeline_resident(std::vector<StateFrame *> frames,
                                                                   const GameConfig &config,
                                                                   unsigned long iterations,
                                                                   energy_type dump_refill_cargo,
                                                                   energy_type mining_refill_cell_energy);
BatchedMapPipelineStats run_cuda_batched_economy_pipeline_device_resident(std::vector<StateFrame *> frames,
                                                                          const GameConfig &config,
                                                                          unsigned long turns,
                                                                          unsigned long iterations_per_turn,
                                                                          energy_type dump_refill_cargo,
                                                                          energy_type mining_refill_cell_energy);

} // namespace hlt::gpu

#endif // CUDABATCHEDMAP_HPP
