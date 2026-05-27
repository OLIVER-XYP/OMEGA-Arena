#ifndef EVENTSINK_HPP
#define EVENTSINK_HPP

#include <vector>

#include "DomainEvent.hpp"

namespace hlt::events {

class EventSink {
public:
    virtual ~EventSink() = default;
    virtual void emit(const DomainEvent &event) = 0;
};

class RecordingEventSink final : public EventSink {
    std::vector<DomainEvent> recorded_events;

public:
    void emit(const DomainEvent &event) override { recorded_events.push_back(event); }
    const std::vector<DomainEvent> &events() const { return recorded_events; }
    void clear() { recorded_events.clear(); }
};

} // namespace hlt::events

#endif // EVENTSINK_HPP
