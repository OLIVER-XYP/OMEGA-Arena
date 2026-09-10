#ifndef RULECONTEXT_HPP
#define RULECONTEXT_HPP

#include "GameConfig.hpp"
#include "ActionBatch.hpp"
#include "StepResult.hpp"
#include "GameState.hpp"
#include "EventSink.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules {

class RuleContext {
    hlt::GameState &state_ref;
    hlt::ActionBatch &actions_ref;
    hlt::StepResult &result_ref;
    const hlt::GameConfig &config_ref;
    hlt::events::EventSink &event_sink_ref;
    hlt::TaskExecutor *executor_ref{};

public:
    RuleContext(hlt::GameState &state,
                hlt::ActionBatch &actions,
                hlt::StepResult &result,
                const hlt::GameConfig &config,
                hlt::events::EventSink &event_sink,
                hlt::TaskExecutor *executor = nullptr)
        : state_ref(state), actions_ref(actions), result_ref(result), config_ref(config), event_sink_ref(event_sink), executor_ref(executor) {}

    hlt::GameState &state() { return state_ref; }
    hlt::ActionBatch &actions() { return actions_ref; }
    hlt::StepResult &result() { return result_ref; }
    const hlt::GameConfig &config() const { return config_ref; }
    hlt::events::EventSink &event_sink() { return event_sink_ref; }
    hlt::TaskExecutor *executor() { return executor_ref; }
    const hlt::TaskExecutor *executor() const { return executor_ref; }
};

} // namespace hlt::rules

#endif // RULECONTEXT_HPP
