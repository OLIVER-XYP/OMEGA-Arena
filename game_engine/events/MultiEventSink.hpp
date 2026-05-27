#ifndef MULTIEVENTSINK_HPP
#define MULTIEVENTSINK_HPP

#include <vector>

#include "EventSink.hpp"

namespace hlt::events {

class MultiEventSink final : public EventSink {
    std::vector<EventSink *> sinks;

public:
    void add(EventSink &sink) { sinks.push_back(&sink); }

    void emit(const DomainEvent &event) override {
        for (auto *sink : sinks) {
            sink->emit(event);
        }
    }
};

} // namespace hlt::events

#endif // MULTIEVENTSINK_HPP
