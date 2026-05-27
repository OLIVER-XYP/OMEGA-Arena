#ifndef MOVEMENTPHASE_HPP
#define MOVEMENTPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class MovementPhase final : public IPhase {
public:
    std::string name() const override { return "movement"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // MOVEMENTPHASE_HPP
