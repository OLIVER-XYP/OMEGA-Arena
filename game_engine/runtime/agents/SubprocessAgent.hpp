#ifndef SUBPROCESSAGENT_HPP
#define SUBPROCESSAGENT_HPP

#include <string>

#include "IPlayerAgent.hpp"

namespace hlt::runtime {

class SubprocessAgent final : public IPlayerAgent {
    std::string command;

public:
    explicit SubprocessAgent(std::string command) : command(std::move(command)) {}
    const std::string &launch_command() const { return command; }
    std::string initialize(const protocol::InitObservation &observation) override;
    protocol::RawPlayerCommands next_actions(const protocol::TurnObservation &observation) override;
    void shutdown() override;
};

} // namespace hlt::runtime

#endif // SUBPROCESSAGENT_HPP
