#ifndef OVERSHIPTAXPHASE_HPP
#define OVERSHIPTAXPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class OverShipTaxPhase final : public IPhase {
public:
    std::string name() const override { return "over_ship_tax"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // OVERSHIPTAXPHASE_HPP
