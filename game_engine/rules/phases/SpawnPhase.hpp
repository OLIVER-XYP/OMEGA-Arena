#ifndef SPAWNPHASE_HPP
#define SPAWNPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class SpawnPhase final : public IPhase {
public:
    std::string name() const override { return "spawn"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // SPAWNPHASE_HPP
