#ifndef CUDAVALIDATION_HPP
#define CUDAVALIDATION_HPP

#include <vector>

#include "ActionBatch.hpp"
#include "CommandFrame.hpp"
#include "GameConfig.hpp"
#include "StateFrame.hpp"
#include "StepResult.hpp"
#include "Store.hpp"

namespace hlt::gpu {

struct ValidationCommandRef {
    Player::id_type player{Player::None};
    const Command *command{};
    const MoveCommand *move{};
    const ConstructCommand *construct{};
    const SpawnCommand *spawn{};
    const AttackCommand *attack{};
    const DefendCommand *defend{};
    const HealCommand *heal{};
    FlatCommand flat{};
};

struct ValidationFrameDecision {
    Player::id_type player{Player::None};
    Entity::id_type entity{Entity::None};
    FlatCommandType type{FlatCommandType::Move};
    energy_type expense{};
    bool command{};
    bool player_exists{};
    bool ownership_ok{};
    bool combat_enabled{};
    bool occurrence_command{};
    bool expense_command{};
    bool include_in_batch{};
};

std::vector<ValidationCommandRef> flatten_validation_actions(const ActionBatch &actions, const Store &store);
CommandFrame command_frame_from_validation_refs(const std::vector<ValidationCommandRef> &refs);
void apply_validation_decisions(GameState &state,
                                ActionBatch &actions,
                                StepResult &result,
                                const std::vector<ValidationCommandRef> &refs,
                                const std::vector<ValidationFrameDecision> &decisions);
std::vector<ValidationFrameDecision> run_cuda_validation_decisions(const StateFrame &state_frame,
                                                                   const CommandFrame &commands,
                                                                   const GameConfig &config);

} // namespace hlt::gpu

#endif // CUDAVALIDATION_HPP
