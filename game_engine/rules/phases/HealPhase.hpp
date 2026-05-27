#ifndef HEALPHASE_HPP
#define HEALPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class HealPhase final : public IPhase {
public:
    std::string name() const override { return "heal"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // HEALPHASE_HPP
