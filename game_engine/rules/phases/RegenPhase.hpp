#ifndef REGENPHASE_HPP
#define REGENPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class RegenPhase final : public IPhase {
public:
    std::string name() const override { return "regen"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // REGENPHASE_HPP
