#ifndef CPUTURNBACKEND_HPP
#define CPUTURNBACKEND_HPP

#include "ITurnBackend.hpp"
#include "TurnEngine.hpp"

namespace hlt {

class CpuTurnBackend final : public ITurnBackend {
    TurnEngine &turn_engine;

public:
    explicit CpuTurnBackend(TurnEngine &turn_engine) : turn_engine(turn_engine) {}

    StepResult step(GameState &state,
                    ActionBatch &actions,
                    events::EventSink &event_sink,
                    TaskExecutor &executor,
                    TurnExecutionProfile *profile) override {
        return turn_engine.step(state, actions, event_sink, executor, profile);
    }

    std::string backend_name() const override { return "cpu"; }
};

} // namespace hlt

#endif // CPUTURNBACKEND_HPP
