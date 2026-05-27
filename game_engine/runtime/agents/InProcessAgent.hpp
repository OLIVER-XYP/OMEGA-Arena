#ifndef INPROCESSAGENT_HPP
#define INPROCESSAGENT_HPP

#include <functional>

#include "IPlayerAgent.hpp"

namespace hlt::runtime {

class InProcessAgent final : public IPlayerAgent {
public:
    using InitHandler = std::function<std::string(const protocol::InitObservation &)>;
    using TurnHandler = std::function<protocol::RawPlayerCommands(const protocol::TurnObservation &)>;

private:
    InitHandler init_handler;
    TurnHandler turn_handler;

public:
    InProcessAgent(InitHandler init_handler, TurnHandler turn_handler)
        : init_handler(std::move(init_handler)), turn_handler(std::move(turn_handler)) {}

    std::string initialize(const protocol::InitObservation &observation) override {
        return init_handler ? init_handler(observation) : std::string{};
    }

    protocol::RawPlayerCommands next_actions(const protocol::TurnObservation &observation) override {
        return turn_handler ? turn_handler(observation) : protocol::RawPlayerCommands{};
    }

    void shutdown() override {}
};

} // namespace hlt::runtime

#endif // INPROCESSAGENT_HPP
