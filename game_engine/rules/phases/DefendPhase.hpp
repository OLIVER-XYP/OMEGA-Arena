#ifndef DEFENDPHASE_HPP
#define DEFENDPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class DefendPhase final : public IPhase {
public:
    std::string name() const override { return "defend"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // DEFENDPHASE_HPP
