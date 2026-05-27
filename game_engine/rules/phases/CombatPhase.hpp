#ifndef COMBATPHASE_HPP
#define COMBATPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class CombatPhase final : public IPhase {
public:
    std::string name() const override { return "combat"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // COMBATPHASE_HPP
