#ifndef VALIDATIONPHASE_HPP
#define VALIDATIONPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class ValidationPhase final : public IPhase {
public:
    std::string name() const override { return "validation"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // VALIDATIONPHASE_HPP
