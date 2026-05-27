#ifndef ACTIONDTO_HPP
#define ACTIONDTO_HPP

#include <string>
#include <vector>

namespace hlt::protocol {

struct RawPlayerCommands {
    std::string raw_text;
};

struct ActionDTO {
    char type{};
    std::vector<std::string> arguments;
};

} // namespace hlt::protocol

#endif // ACTIONDTO_HPP
