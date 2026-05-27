#ifndef INSPIRATIONPHASE_HPP
#define INSPIRATIONPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class InspirationPhase final : public IPhase {
public:
    std::string name() const override { return "inspiration"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // INSPIRATIONPHASE_HPP
