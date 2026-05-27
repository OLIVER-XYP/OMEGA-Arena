#ifndef HALITEREBALANCEPHASE_HPP
#define HALITEREBALANCEPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class HaliteRebalancePhase final : public IPhase {
public:
    std::string name() const override { return "halite_rebalance"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // HALITEREBALANCEPHASE_HPP
