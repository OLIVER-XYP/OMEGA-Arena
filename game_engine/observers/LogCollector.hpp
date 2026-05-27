#ifndef LOGCOLLECTOR_HPP
#define LOGCOLLECTOR_HPP

#include <vector>

#include "EventSink.hpp"

namespace hlt::observers {

class LogCollector final : public events::EventSink {
    std::vector<events::DomainEvent> pending_events;

public:
    void emit(const events::DomainEvent &event) override { pending_events.push_back(event); }
    const std::vector<events::DomainEvent> &events() const { return pending_events; }
    void clear() { pending_events.clear(); }
};

} // namespace hlt::observers

#endif // LOGCOLLECTOR_HPP
