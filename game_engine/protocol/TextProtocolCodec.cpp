#include "TextProtocolCodec.hpp"

namespace hlt::protocol {

std::string TextProtocolCodec::encode_init(const InitObservation &observation) const {
    std::ostringstream stream;
    stream << observation.serialized_map;
    return stream.str();
}

std::string TextProtocolCodec::encode_turn(const TurnObservation &observation) const {
    std::ostringstream stream;
    stream << observation.turn_number << '\n';
    stream << observation.serialized_players;
    stream << observation.changed_cells.size() << '\n';
    for (const auto &cell : observation.changed_cells) {
        stream << cell.location << " " << cell.energy << '\n';
    }
    return stream.str();
}

RawPlayerCommands TextProtocolCodec::decode_commands(std::string raw_text) const {
    return RawPlayerCommands{std::move(raw_text)};
}

} // namespace hlt::protocol
