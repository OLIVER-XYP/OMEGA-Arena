#include "TurnEngine.hpp"

#include <chrono>
#include <memory>

#include "RuleContext.hpp"
#include "phases/CapturePhase.hpp"
#include "phases/CombatPhase.hpp"
#include "phases/ConstructionPhase.hpp"
#include "phases/DefendPhase.hpp"
#include "phases/DumpPhase.hpp"
#include "phases/HaliteRebalancePhase.hpp"
#include "phases/HealPhase.hpp"
#include "phases/InspirationPhase.hpp"
#include "phases/MiningPhase.hpp"
#include "phases/MovementPhase.hpp"
#include "phases/OverShipTaxPhase.hpp"
#include "phases/PlunderPhase.hpp"
#include "phases/RegenPhase.hpp"
#include "phases/SpawnPhase.hpp"
#include "phases/ValidationPhase.hpp"

namespace hlt {

TurnEngine::TurnEngine(const GameConfig &config) : config(config) {
    ruleset.add_phase(std::make_unique<rules::phases::InspirationPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::ValidationPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::ConstructionPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::DefendPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::MovementPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::CombatPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::HealPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::DumpPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::SpawnPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::MiningPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::PlunderPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::RegenPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::CapturePhase>());
    ruleset.add_phase(std::make_unique<rules::phases::OverShipTaxPhase>());
    ruleset.add_phase(std::make_unique<rules::phases::HaliteRebalancePhase>());
}

StepResult TurnEngine::step(GameState &state,
                            ActionBatch &actions,
                            events::EventSink &event_sink,
                            TaskExecutor &executor,
                            TurnExecutionProfile *profile) const {
    StepResult result{};
    rules::RuleContext context(state, actions, result, config, event_sink, &executor);

    executor.reset_run_stats();
    if (profile != nullptr) {
        profile->phase_timings.clear();
        profile->executor_type = executor.executor_name();
        profile->executor_thread_count = executor.executor_thread_count();
    }

    const auto turn_start = std::chrono::steady_clock::now();
    for (const auto &phase : ruleset.all_phases()) {
        const auto phase_start = std::chrono::steady_clock::now();
        phase->execute(context);
        if (profile != nullptr) {
            const auto phase_end = std::chrono::steady_clock::now();
            profile->phase_timings.push_back(PhaseTimingEntry{phase->name(),
                                                              std::chrono::duration_cast<std::chrono::nanoseconds>(phase_end - phase_start).count()});
        }
    }

    if (profile != nullptr) {
        const auto turn_end = std::chrono::steady_clock::now();
        profile->total_turn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(turn_end - turn_start).count();
        profile->executor_stats = executor.run_stats();
    }

    return result;
}

StepResult TurnEngine::step(GameState &state,
                            ActionBatch &actions,
                            events::EventSink &event_sink,
                            TaskExecutor &executor) const {
    return step(state, actions, event_sink, executor, nullptr);
}

StepResult TurnEngine::step(GameState &state, ActionBatch &actions, events::EventSink &event_sink) const {
    InlineExecutor executor;
    return step(state, actions, event_sink, executor, nullptr);
}

} // namespace hlt
