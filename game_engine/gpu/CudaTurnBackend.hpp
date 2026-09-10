#ifndef CUDATURNBACKEND_HPP
#define CUDATURNBACKEND_HPP

#include <memory>

#include "GameConfig.hpp"
#include "ITurnBackend.hpp"
#include "TurnEngine.hpp"

namespace hlt::gpu {

bool cuda_turn_backend_available();
const char *cuda_turn_backend_unavailable_reason();
const char *cuda_turn_backend_capability_summary();
std::unique_ptr<ITurnBackend> make_cuda_turn_backend(TurnEngine &cpu_reference_engine, const GameConfig &config);

class CudaTurnBackend final : public ITurnBackend {
    TurnEngine &cpu_reference_engine;
    const GameConfig &config;

public:
    CudaTurnBackend(TurnEngine &cpu_reference_engine, const GameConfig &config);

    StepResult step(GameState &state,
                    ActionBatch &actions,
                    events::EventSink &event_sink,
                    TaskExecutor &executor,
                    TurnExecutionProfile *profile) override;

    bool supports_prevalidated_step() const override;

    StepResult step_from_validated(GameState &state,
                                   ActionBatch &actions,
                                   events::EventSink &event_sink,
                                   TaskExecutor &executor,
                                   TurnExecutionProfile *profile,
                                   StepResult prevalidated_result) override;

    bool supports_split_dump_step() const override;

    StepResult step_until_dump(GameState &state,
                               ActionBatch &actions,
                               events::EventSink &event_sink,
                               TaskExecutor &executor,
                               TurnExecutionProfile *profile,
                               StepResult prevalidated_result) override;

    StepResult step_after_dump(GameState &state,
                               ActionBatch &actions,
                               events::EventSink &event_sink,
                               TaskExecutor &executor,
                               TurnExecutionProfile *profile,
                               StepResult partial_result) override;

    std::string backend_name() const override;
};

} // namespace hlt::gpu

#endif // CUDATURNBACKEND_HPP
