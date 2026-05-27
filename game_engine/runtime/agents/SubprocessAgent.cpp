#include "SubprocessAgent.hpp"

namespace hlt::runtime {

std::string SubprocessAgent::initialize(const protocol::InitObservation &observation) {
    (void)observation;
    return {};
}

protocol::RawPlayerCommands SubprocessAgent::next_actions(const protocol::TurnObservation &observation) {
    (void)observation;
    return {};
}

void SubprocessAgent::shutdown() {}

} // namespace hlt::runtime
