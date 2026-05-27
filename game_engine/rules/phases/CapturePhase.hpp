#ifndef CAPTUREPHASE_HPP
#define CAPTUREPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class CapturePhase final : public IPhase {
public:
    std::string name() const override { return "capture"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // CAPTUREPHASE_HPP
