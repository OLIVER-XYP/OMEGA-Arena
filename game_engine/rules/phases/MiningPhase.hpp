#ifndef MININGPHASE_HPP
#define MININGPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class MiningPhase final : public IPhase {
public:
    std::string name() const override { return "mining"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // MININGPHASE_HPP
