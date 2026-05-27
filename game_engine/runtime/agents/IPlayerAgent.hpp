#ifndef IPLAYERAGENT_HPP
#define IPLAYERAGENT_HPP

#include "ActionDTO.hpp"
#include "Observation.hpp"

namespace hlt::runtime {

class IPlayerAgent {
public:
    virtual ~IPlayerAgent() = default;
    virtual std::string initialize(const protocol::InitObservation &observation) = 0;
    virtual protocol::RawPlayerCommands next_actions(const protocol::TurnObservation &observation) = 0;
    virtual void shutdown() = 0;
};

} // namespace hlt::runtime

#endif // IPLAYERAGENT_HPP
