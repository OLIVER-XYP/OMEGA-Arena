#ifndef TEXTPROTOCOLCODEC_HPP
#define TEXTPROTOCOLCODEC_HPP

#include <sstream>
#include <string>

#include "ActionDTO.hpp"
#include "Observation.hpp"

namespace hlt::protocol {

class TextProtocolCodec {
public:
    std::string encode_init(const InitObservation &observation) const;
    std::string encode_turn(const TurnObservation &observation) const;
    RawPlayerCommands decode_commands(std::string raw_text) const;
};

} // namespace hlt::protocol

#endif // TEXTPROTOCOLCODEC_HPP
