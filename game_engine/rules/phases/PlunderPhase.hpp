#ifndef PLUNDERPHASE_HPP
#define PLUNDERPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class PlunderPhase final : public IPhase {
public:
    std::string name() const override { return "plunder"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // PLUNDERPHASE_HPP
