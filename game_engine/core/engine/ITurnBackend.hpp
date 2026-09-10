#ifndef ITURNBACKEND_HPP
#define ITURNBACKEND_HPP

#include <stdexcept>
#include <string>

#include "ActionBatch.hpp"
#include "EventSink.hpp"
#include "GameState.hpp"
#include "StepResult.hpp"
#include "TaskExecutor.hpp"
#include "TurnEngine.hpp"

namespace hlt {

class ITurnBackend {
public:
    virtual ~ITurnBackend() = default;

    virtual StepResult step(GameState &state,
                            ActionBatch &actions,
                            events::EventSink &event_sink,
                            TaskExecutor &executor,
                            TurnExecutionProfile *profile) = 0;

    virtual bool supports_prevalidated_step() const { return false; }

    virtual StepResult step_from_validated(GameState &state,
                                           ActionBatch &actions,
                                           events::EventSink &event_sink,
                                           TaskExecutor &executor,
                                           TurnExecutionProfile *profile,
                                           StepResult prevalidated_result) {
        (void)state;
        (void)actions;
        (void)event_sink;
        (void)executor;
        (void)profile;
        (void)prevalidated_result;
        throw std::runtime_error("backend does not support prevalidated steps");
    }

    virtual bool supports_split_dump_step() const { return false; }

    virtual StepResult step_until_dump(GameState &state,
                                       ActionBatch &actions,
                                       events::EventSink &event_sink,
                                       TaskExecutor &executor,
                                       TurnExecutionProfile *profile,
                                       StepResult prevalidated_result) {
        (void)state;
        (void)actions;
        (void)event_sink;
        (void)executor;
        (void)profile;
        (void)prevalidated_result;
        throw std::runtime_error("backend does not support split dump steps");
    }

    virtual StepResult step_after_dump(GameState &state,
                                       ActionBatch &actions,
                                       events::EventSink &event_sink,
                                       TaskExecutor &executor,
                                       TurnExecutionProfile *profile,
                                       StepResult partial_result) {
        (void)state;
        (void)actions;
        (void)event_sink;
        (void)executor;
        (void)profile;
        (void)partial_result;
        throw std::runtime_error("backend does not support split dump steps");
    }

    virtual std::string backend_name() const = 0;
};

} // namespace hlt

#endif // ITURNBACKEND_HPP
