#ifndef CONSTRUCTIONPHASE_HPP
#define CONSTRUCTIONPHASE_HPP

#include "IPhase.hpp"

namespace hlt::rules::phases {

class ConstructionPhase final : public IPhase {
public:
    std::string name() const override { return "construction"; }
    void execute(RuleContext &context) const override;
};

} // namespace hlt::rules::phases

#endif // CONSTRUCTIONPHASE_HPP
