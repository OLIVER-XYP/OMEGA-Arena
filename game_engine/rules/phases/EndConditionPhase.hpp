#ifndef ENDCONDITIONPHASE_HPP
#define ENDCONDITIONPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class EndConditionPhase final : public IPhase {
public:
    std::string name() const override { return "end_condition"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // ENDCONDITIONPHASE_HPP
