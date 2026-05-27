#ifndef RULESET_HPP
#define RULESET_HPP

#include <memory>
#include <vector>

#include "IPhase.hpp"

namespace hlt::rules {

class Ruleset {
    std::vector<std::unique_ptr<IPhase>> phases;

public:
    Ruleset() = default;
    Ruleset(Ruleset &&) noexcept = default;
    Ruleset &operator=(Ruleset &&) noexcept = default;

    void add_phase(std::unique_ptr<IPhase> phase) { phases.push_back(std::move(phase)); }

    void execute(RuleContext &context) const {
        for (const auto &phase : phases) {
            phase->execute(context);
        }
    }

    const std::vector<std::unique_ptr<IPhase>> &all_phases() const { return phases; }
};

} // namespace hlt::rules

#endif // RULESET_HPP
