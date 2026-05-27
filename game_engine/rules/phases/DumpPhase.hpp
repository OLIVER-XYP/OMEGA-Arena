#ifndef DUMPPHASE_HPP
#define DUMPPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class DumpPhase final : public IPhase {
public:
    std::string name() const override { return "dump"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // DUMPPHASE_HPP
