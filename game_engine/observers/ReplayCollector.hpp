#ifndef REPLAYCOLLECTOR_HPP
#define REPLAYCOLLECTOR_HPP

#include "EventSink.hpp"
#include "GameEvent.hpp"
#include "Replay.hpp"

namespace hlt::observers {

class ReplayCollector final : public events::EventSink {
    Replay &replay;

public:
    explicit ReplayCollector(Replay &replay) : replay(replay) {}

    void emit(const events::DomainEvent &event) override {
        std::visit([this](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, events::SpawnedEvent>) {
                replay.full_frames.back().events.push_back(
                    std::make_unique<SpawnEvent>(value.location, value.energy, value.player, value.entity));
            } else if constexpr (std::is_same_v<T, events::CollisionResolvedEvent>) {
                replay.full_frames.back().events.push_back(
                    std::make_unique<CollisionEvent>(value.location, value.ships));
            } else if constexpr (std::is_same_v<T, events::ConstructionResolvedEvent>) {
                replay.full_frames.back().events.push_back(
                    std::make_unique<ConstructionEvent>(value.location, value.player, value.entity));
            } else if constexpr (std::is_same_v<T, events::CombatResolvedEvent>) {
                replay.full_frames.back().events.push_back(
                    std::make_unique<AttackEvent>(value.attacker_location, value.attacker, value.target_location, value.target, value.hit));
            } else if constexpr (std::is_same_v<T, events::CapturedEvent>) {
                replay.full_frames.back().events.push_back(
                    std::make_unique<CaptureEvent>(value.location, value.old_owner, value.old_id, value.new_owner, value.new_id));
            }
        }, event);
    }
};

} // namespace hlt::observers

#endif // REPLAYCOLLECTOR_HPP
