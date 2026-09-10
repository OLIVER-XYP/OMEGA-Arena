// halite_pyenv — pybind11 vectorized step environment for RL self-play.
//
// Exposes a thin, policy-free executor over the engine's CPU turn backend:
//   * HaliteGame       — one game: generate map, inject an ActionBatch, step, score.
//   * HaliteVecEnv     — N parallel games with reset/observe/step/scores.
//
// ALL policy logic (feature encoding, action masking, target selection) stays in
// Python (rl_ai/features.py + rl_ai/actions.py), which is the single source of
// truth shared with inference_bot.py. C++ only executes fully-resolved commands
// (the FlatCommand contract: entity, ctype, dir, target, tx, ty, target_owner).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "ActionBatch.hpp"
#include "Cell.hpp"
#include "Command.hpp"
#include "Constants.hpp"
#include "CpuTurnBackend.hpp"
#include "Dropoff.hpp"
#include "Entity.hpp"
#include "EventSink.hpp"
#include "GameConfig.hpp"
#include "GameState.hpp"
#include "Generator.hpp"
#include "Location.hpp"
#include "Map.hpp"
#include "Player.hpp"
#include "StepResult.hpp"
#include "Statistics.hpp"
#include "Store.hpp"
#include "TaskExecutor.hpp"
#include "TurnEngine.hpp"

namespace py = pybind11;
using namespace hlt;

namespace {

struct NullEventSink final : events::EventSink {
    void emit(const events::DomainEvent &event) override { (void)event; }
};

// Mirrors BatchedRunner's structure_score (factory pool + all dropoff pools).
energy_type score_of(const Player &player) {
    energy_type score = player.factory_halite;
    for (const auto &dropoff : player.dropoffs) {
        score += dropoff.halite_pool;
    }
    return score;
}

Direction dir_from_char(long c) {
    switch (c) {
        case 'n': return Direction::North;
        case 's': return Direction::South;
        case 'e': return Direction::East;
        case 'w': return Direction::West;
        default: return Direction::Still;
    }
}

// FlatCommandType ordering (see CommandFrame.hpp):
// 0 Move, 1 Spawn, 2 Construct, 3 AttackShip, 4 AttackStructure, 5 Defend, 6 Heal
std::unique_ptr<Command> make_command(const std::vector<long> &t) {
    const long entity = t[0];
    const long ctype = t[1];
    const long dch = t[2];
    const long target = t[3];
    const long tx = t[4];
    const long ty = t[5];
    const long towner = t[6];
    switch (ctype) {
        case 1:
            return std::make_unique<SpawnCommand>();
        case 2:
            return std::make_unique<ConstructCommand>(Entity::id_type{entity});
        case 3:
            return std::make_unique<AttackCommand>(Entity::id_type{entity}, Entity::id_type{target});
        case 4:
            return std::make_unique<AttackCommand>(Entity::id_type{entity},
                                                   Player::id_type{towner},
                                                   Location{static_cast<dimension_type>(tx),
                                                            static_cast<dimension_type>(ty)});
        case 5:
            return std::make_unique<DefendCommand>(Entity::id_type{entity});
        case 6:
            return std::make_unique<HealCommand>(Entity::id_type{entity});
        case 0:
        default:
            return std::make_unique<MoveCommand>(Entity::id_type{entity}, dir_from_char(dch));
    }
}

PlayerCommandList build_command_list(const py::handle &cmds) {
    PlayerCommandList out;
    for (const auto &item : cmds) {
        auto t = item.cast<std::vector<long>>();
        if (t.size() != 7) {
            throw std::runtime_error("each command must be a 7-int tuple");
        }
        out.push_back(make_command(t));
    }
    return out;
}

class HaliteGame {
public:
    unsigned int seed;
    unsigned long turn_limit;
    GameConfig config;
    hlt::mapgen::MapParameters map_parameters;
    Map map;
    Store store;
    GameStatistics statistics;
    TurnEngine turn_engine;
    CpuTurnBackend cpu_backend;
    InlineExecutor executor;
    unsigned long turn = 0;
    bool ended = false;

    HaliteGame(unsigned int seed_, unsigned long turn_limit_, GameConfig cfg)
        : seed(seed_),
          turn_limit(turn_limit_),
          config(std::move(cfg)),
          map_parameters{hlt::mapgen::MapType::Fractal, seed_, 32, 32, 2},
          map(32, 32),
          turn_engine(config),
          cpu_backend(turn_engine) {
        initialize();
    }

    void init_player(Player &player) {
        player.energy = Constants::get().INITIAL_ENERGY;
        player.factory_halite = Constants::get().INITIAL_FACTORY_HALITE;
        player.factory_destroyed = false;
        statistics.player_statistics.emplace_back(player.id, seed);
        auto &factory = map.at(player.factory);
        store.map_total_energy -= factory.energy;
        factory.energy = 0;
        factory.initial_energy = 0;
        factory.owner = player.id;
        // Match the external engine protocol: each player begins with an empty
        // shipyard and must issue a SpawnCommand on the first turn. Creating an
        // implicit turn-0 entity here made PPO observe a state that deployment
        // can never produce, while inference_bot correctly force-spawns from
        // the empty-shipyard state.
    }

    void initialize() {
        hlt::mapgen::Generator::generate(map, map_parameters);
        for (dimension_type row = 0; row < map.height; ++row) {
            for (dimension_type col = 0; col < map.width; ++col) {
                auto &cell = map.at(col, row);
                cell.initial_energy = cell.energy;
                store.map_total_energy += cell.energy;
                statistics.map_total_halite += cell.energy;
            }
        }
        if (map.factories.size() < 2) {
            throw std::runtime_error("map generation produced fewer than two factories");
        }
        auto &player0 = store.new_player(map.factories[0], "p0");
        auto &player1 = store.new_player(map.factories[1], "p1");
        init_player(player0);
        init_player(player1);
    }

    bool game_ended() const {
        for (const auto &entry : store.players_ref()) {
            const Player &player = entry.second;
            if (player.terminated) {
                continue;
            }
            if (score_of(player) == 0) {
                return true;
            }
        }
        return false;
    }

    void step(ActionBatch &actions) {
        if (ended || turn >= turn_limit) {
            return;
        }
        const unsigned long next = turn + 1;
        store.current_turn = next;
        statistics.turn_number = next;
        GameState state{store, map, statistics};
        state.turn.number = next;
        NullEventSink sink;
        StepResult result = cpu_backend.step(state, actions, sink, executor, nullptr);
        for (auto player_id : result.eliminated_players) {
            auto it = store.players_ref().find(player_id);
            if (it != store.players_ref().end()) {
                it->second.terminated = true;
                it->second.can_play = false;
            }
        }
        turn = next;
        if (game_ended() || turn >= turn_limit) {
            ended = true;
        }
    }
};

py::dict observe_one(const HaliteGame &game, int me) {
    const Player::id_type my_id{me};
    const Player::id_type opp_id{1 - me};
    const Store &store = game.store;
    const Player &mp = store.players_ref().at(my_id);
    const Player &op = store.players_ref().at(opp_id);

    const int W = static_cast<int>(game.map.width);
    const int H = static_cast<int>(game.map.height);
    py::array_t<float> halite(static_cast<py::ssize_t>(W) * H);
    auto hbuf = halite.mutable_unchecked<1>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            hbuf(y * W + x) = static_cast<float>(game.map.at(x, y).energy);
        }
    }

    auto ship_list = [&store](const Player &p) {
        py::list ships;
        const long owner = p.id.value;
        for (const auto &kv : p.entities) {
            const Entity::id_type eid = kv.first;
            const Location &loc = kv.second;
            const Entity &e = store.get_entity(eid);
            py::dict s;
            s["id"] = eid.value;
            s["owner"] = owner;
            s["x"] = loc.x;
            s["y"] = loc.y;
            s["cargo"] = e.energy;
            s["hp"] = e.hp;
            s["is_inspired"] = e.is_inspired;
            s["is_defending"] = e.is_defending;
            ships.append(std::move(s));
        }
        return ships;
    };

    auto struct_list = [](const Player &p) {
        py::list structs;
        const long owner = p.id.value;
        py::dict factory;
        factory["owner"] = owner;
        factory["x"] = p.factory.x;
        factory["y"] = p.factory.y;
        factory["kind"] = "shipyard";
        structs.append(std::move(factory));
        for (const auto &d : p.dropoffs) {
            py::dict s;
            s["owner"] = owner;
            s["x"] = d.location.x;
            s["y"] = d.location.y;
            s["kind"] = "dropoff";
            structs.append(std::move(s));
        }
        return structs;
    };

    energy_type own_cargo = 0;
    for (const auto &kv : mp.entities) {
        own_cargo += store.get_entity(kv.first).energy;
    }
    energy_type enemy_cargo = 0;
    for (const auto &kv : op.entities) {
        enemy_cargo += store.get_entity(kv.first).energy;
    }

    py::dict rec;
    rec["turn"] = game.turn;
    rec["turns_total"] = game.turn_limit;
    rec["width"] = W;
    rec["height"] = H;
    rec["player_energy"] = mp.energy;
    rec["enemy_energy"] = op.energy;
    rec["own_ship_count"] = mp.entities.size();
    rec["enemy_ship_count"] = op.entities.size();
    rec["own_cargo"] = own_cargo;
    rec["enemy_cargo"] = enemy_cargo;
    rec["own_dropoff_count"] = mp.dropoffs.size();
    rec["own_deposited"] = score_of(mp);
    rec["enemy_deposited"] = score_of(op);
    rec["own_ships"] = ship_list(mp);
    rec["enemy_ships"] = ship_list(op);
    rec["own_structures"] = struct_list(mp);
    rec["enemy_structures"] = struct_list(op);
    rec["halite"] = halite;
    rec["ended"] = game.ended;
    return rec;
}

class HaliteVecEnv {
public:
    GameConfig config;
    unsigned long turn_limit;
    std::vector<std::unique_ptr<HaliteGame>> games;

    HaliteVecEnv(const std::string &config_path, unsigned long turn_limit_)
        : turn_limit(turn_limit_) {
        if (!config_path.empty()) {
            std::ifstream constants_file(config_path);
            if (!constants_file.is_open()) {
                throw std::runtime_error("could not open config file: " + config_path);
            }
            nlohmann::json constants_json;
            constants_file >> constants_json;
            from_json(constants_json, Constants::get_mut());
        }
        config = GameConfig::from_constants();
        config.match.max_turns = turn_limit_;
        config.match.min_turns = turn_limit_;
        config.runtime.engine_backend = EngineBackendMode::Cpu;
        config.runtime.gpu_batch_size = 1;
        config.apply_to_global_constants();
    }

    void reset(const std::vector<unsigned int> &seeds) {
        games.clear();
        games.reserve(seeds.size());
        for (auto seed : seeds) {
            games.push_back(std::make_unique<HaliteGame>(seed, turn_limit, config));
        }
    }

    std::size_t size() const { return games.size(); }

    py::list observe(int me) {
        py::list out;
        for (const auto &game : games) {
            out.append(observe_one(*game, me));
        }
        return out;
    }

    // commands_p0 / commands_p1: list (len n_envs) of list of 7-int command tuples.
    void step(const py::list &commands_p0, const py::list &commands_p1) {
        const std::size_t n = games.size();
        if (py::len(commands_p0) != n || py::len(commands_p1) != n) {
            throw std::runtime_error("command list length must equal number of envs");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (games[i]->ended) {
                continue;
            }
            ActionBatch batch;
            batch[Player::id_type{0}] = build_command_list(commands_p0[i]);
            batch[Player::id_type{1}] = build_command_list(commands_p1[i]);
            games[i]->step(batch);
        }
    }

    std::vector<bool> dones() const {
        std::vector<bool> out;
        out.reserve(games.size());
        for (const auto &game : games) {
            out.push_back(game->ended);
        }
        return out;
    }

    std::vector<long> turns() const {
        std::vector<long> out;
        out.reserve(games.size());
        for (const auto &game : games) {
            out.push_back(static_cast<long>(game->turn));
        }
        return out;
    }

    // Per env: (score_player0, score_player1).
    py::list scores() const {
        py::list out;
        for (const auto &game : games) {
            const Player &p0 = game->store.players_ref().at(Player::id_type{0});
            const Player &p1 = game->store.players_ref().at(Player::id_type{1});
            out.append(py::make_tuple(score_of(p0), score_of(p1)));
        }
        return out;
    }

    bool all_done() const {
        for (const auto &game : games) {
            if (!game->ended) {
                return false;
            }
        }
        return true;
    }
};

} // namespace

PYBIND11_MODULE(halite_pyenv, m) {
    m.doc() = "Halite III vectorized step environment for RL self-play";
    py::class_<HaliteVecEnv>(m, "HaliteVecEnv")
        .def(py::init<const std::string &, unsigned long>(),
             py::arg("config_path"), py::arg("turn_limit") = 300)
        .def("reset", &HaliteVecEnv::reset, py::arg("seeds"))
        .def("observe", &HaliteVecEnv::observe, py::arg("player"))
        .def("step", &HaliteVecEnv::step, py::arg("commands_p0"), py::arg("commands_p1"))
        .def("dones", &HaliteVecEnv::dones)
        .def("turns", &HaliteVecEnv::turns)
        .def("scores", &HaliteVecEnv::scores)
        .def("all_done", &HaliteVecEnv::all_done)
        .def("size", &HaliteVecEnv::size)
        .def_readonly("turn_limit", &HaliteVecEnv::turn_limit);
}
