#ifndef EVENTBUFFER_HPP
#define EVENTBUFFER_HPP

#include <utility>
#include <vector>

#include "DomainEvent.hpp"
#include "EventSink.hpp"

namespace hlt::events {

class EventBuffer {
    std::vector<DomainEvent> buffered_events;

public:
    void emit(const DomainEvent &event) {
        buffered_events.push_back(event);
    }

    void emit(DomainEvent &&event) {
        buffered_events.push_back(std::move(event));
    }

    void flush_into(EventSink &sink) {
        for (const auto &event : buffered_events) {
            sink.emit(event);
        }
        buffered_events.clear();
    }

    const std::vector<DomainEvent> &events() const { return buffered_events; }
    bool empty() const { return buffered_events.empty(); }
    void clear() { buffered_events.clear(); }
};

} // namespace hlt::events

#endif // EVENTBUFFER_HPP
