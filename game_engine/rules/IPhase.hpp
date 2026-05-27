#ifndef IPHASE_HPP
#define IPHASE_HPP

#include <string>

#include "RuleContext.hpp"

namespace hlt::rules {

class IPhase {
public:
    virtual ~IPhase() = default;
    virtual std::string name() const = 0;
    virtual void execute(RuleContext &context) const = 0;
};

} // namespace hlt::rules

#endif // IPHASE_HPP
