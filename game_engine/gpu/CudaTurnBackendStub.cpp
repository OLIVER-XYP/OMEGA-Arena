#include "CudaTurnBackend.hpp"

#include <stdexcept>

namespace hlt::gpu {

bool cuda_turn_backend_available() {
    return false;
}

const char *cuda_turn_backend_unavailable_reason() {
    return "CUDA turn backend was not built. Reconfigure with HALITE_ENABLE_CUDA=ON after CUDA kernels are available.";
}

const char *cuda_turn_backend_capability_summary() {
    return "none";
}

std::unique_ptr<ITurnBackend> make_cuda_turn_backend(TurnEngine &cpu_reference_engine, const GameConfig &config) {
    (void)cpu_reference_engine;
    (void)config;
    return nullptr;
}

CudaTurnBackend::CudaTurnBackend(TurnEngine &cpu_reference_engine, const GameConfig &config)
    : cpu_reference_engine(cpu_reference_engine), config(config) {}

StepResult CudaTurnBackend::step(GameState &state,
                                 ActionBatch &actions,
                                 events::EventSink &event_sink,
                                 TaskExecutor &executor,
                                 TurnExecutionProfile *profile) {
    (void)state;
    (void)actions;
    (void)event_sink;
    (void)executor;
    (void)profile;
    throw std::runtime_error(cuda_turn_backend_unavailable_reason());
}

bool CudaTurnBackend::supports_prevalidated_step() const {
    return false;
}

bool CudaTurnBackend::supports_split_dump_step() const {
    return false;
}

StepResult CudaTurnBackend::step_from_validated(GameState &state,
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
    throw std::runtime_error(cuda_turn_backend_unavailable_reason());
}

StepResult CudaTurnBackend::step_until_dump(GameState &state,
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
    throw std::runtime_error(cuda_turn_backend_unavailable_reason());
}

StepResult CudaTurnBackend::step_after_dump(GameState &state,
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
    throw std::runtime_error(cuda_turn_backend_unavailable_reason());
}

std::string CudaTurnBackend::backend_name() const {
    (void)cpu_reference_engine;
    (void)config;
    return "cuda-unavailable";
}

} // namespace hlt::gpu
