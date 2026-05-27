#ifndef TURNENGINE_HPP
#define TURNENGINE_HPP

#include <string>
#include <vector>

#include "ActionBatch.hpp"
#include "EventSink.hpp"
#include "GameConfig.hpp"
#include "GameState.hpp"
#include "Ruleset.hpp"
#include "StepResult.hpp"
#include "TaskExecutor.hpp"

namespace hlt {

struct PhaseTimingEntry {
    std::string phase_name;
    long long duration_ns{};
};

struct TurnExecutionProfile {
    long long total_turn_ns{};
    std::string executor_type;
    std::size_t executor_thread_count{};
    ExecutorRunStats executor_stats{};
    std::vector<PhaseTimingEntry> phase_timings;
};

class TurnEngine {
    const GameConfig &config;
    rules::Ruleset ruleset;

public:
    explicit TurnEngine(const GameConfig &config);
    StepResult step(GameState &state,
                    ActionBatch &actions,
                    events::EventSink &event_sink,
                    TaskExecutor &executor,
                    TurnExecutionProfile *profile) const;
    StepResult step(GameState &state, ActionBatch &actions, events::EventSink &event_sink, TaskExecutor &executor) const;
    StepResult step(GameState &state, ActionBatch &actions, events::EventSink &event_sink) const;
};

} // namespace hlt

#endif // TURNENGINE_HPP
