#include "hlt/game.hpp"
#include "hlt/constants.hpp"
#include "hlt/log.hpp"
#include "bot_params.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>

using namespace hlt;
namespace fs = std::filesystem;

enum class Mode { Collect, Return, Hunt, Camp };

struct State {
    std::unordered_map<EntityId, Mode> mode;
    std::unordered_map<EntityId, Position> target;   // mining target or hunt intercept point
    std::unordered_map<EntityId, EntityId> hunt_id;  // enemy ship ID being hunted
    std::unordered_map<EntityId, Position> camp_pos; // camp position (Camp mode)
    std::unordered_map<EntityId, int> ttl;            // stuck counter
    std::unordered_map<EntityId, Position> prev;      // previous-turn position (oscillation fix)
};

static std::vector<Position> own_drops(const std::shared_ptr<Player>& me){
    std::vector<Position> d{me->shipyard->position};
    for (const auto& kv: me->dropoffs) d.push_back(kv.second->position);
    return d;
}
static std::vector<std::pair<PlayerId,Position>> enemy_drops(const Game& g){
    std::vector<std::pair<PlayerId,Position>> d;
    for (const auto& p: g.players){
        if(p->id==g.my_id) continue;
        // An enemy with no ships and not enough halite to spawn another is
        // eliminated; their dropoffs no longer threaten us so we shouldn't
        // treat their half of the map as off-limits.
        if(p->ships.empty() && p->halite < constants::SHIP_COST) continue;
        d.push_back({p->id,p->shipyard->position});
        for(const auto& kv:p->dropoffs) d.push_back({p->id,kv.second->position});
    }
    return d;
}

// Count enemy ships that still belong to a non-eliminated player.
static int alive_enemy_ship_count(const Game& g){
    int n = 0;
    for (const auto& p: g.players){
        if(p->id==g.my_id) continue;
        if(p->ships.empty() && p->halite < constants::SHIP_COST) continue;
        n += (int)p->ships.size();
    }
    return n;
}
static bool is_our_structure(const std::shared_ptr<Player>& me, const Position& p){
    if(p==me->shipyard->position) return true;
    for(const auto& kv: me->dropoffs) if(kv.second->position==p) return true;
    return false;
}
static Position nearest_drop(const std::unique_ptr<GameMap>& m,const Position& p,const std::vector<Position>& ds){
    Position b=ds.front(); int bd=1e9;
    for(const auto& d:ds){int x=m->calculate_distance(p,d); if(x<bd){bd=x;b=d;}}
    return b;
}
static int nearest_drop_dist(const std::unique_ptr<GameMap>& m,const Position& p,const std::vector<Position>& ds){
    return m->calculate_distance(p, nearest_drop(m,p,ds));
}
static bool can_move(const std::unique_ptr<GameMap>& m,const std::shared_ptr<Ship>& s){
    return s->halite >= m->at(s)->halite / constants::MOVE_COST_RATIO;
}
static int rr(const BotParams& P,int home){
    double t=std::min(1.0, home / (double)std::max(1,P.return_dist));
    double r=P.return_close+(P.return_far-P.return_close)*t;
    return (int)(constants::MAX_HALITE*r);
}
static std::pair<int,int> mine_sim(int h,int stop,int cargo){
    int cap=constants::MAX_HALITE-cargo; if(cap<=0) return {0,0};
    int c=0,t=0;
    while(h>stop&&c<cap){
        int e=h/constants::EXTRACT_RATIO; if(e<=0) break;
        int g=std::min(e,cap-c); if(g<=0) break;
        c+=g; h-=e; t++;
    }
    return {c,t};
}
static int radius_now(const BotParams& P,int turn){
    double ph=turn/(double)std::max(1,constants::MAX_TURNS);
    if(ph<0.33) return P.search_radius_early;
    if(ph<0.75) return P.search_radius_mid;
    return P.search_radius_late;
}
static int local_sum(const std::unique_ptr<GameMap>& m,const Position& p,int r){
    int s=0;
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        Position q=m->normalize(Position{p.x+dx,p.y+dy});
        if(m->calculate_distance(p,q)<=r) s+=m->at(q)->halite;
    }
    return s;
}
static int congestion_around(const std::unique_ptr<GameMap>& m,const Position& p,int r){
    int c=0;
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
        if(dx==0&&dy==0) continue;
        Position q=m->normalize(Position{p.x+dx,p.y+dy});
        if(m->calculate_distance(p,q)<=r && m->at(q)->is_occupied()) c++;
    }
    return c;
}
static double score_cell(const std::unique_ptr<GameMap>& m,const Position& p,const Position& sp,const std::vector<Position>& ds,int stop,int cargo,int left,const BotParams& P){
    int h=m->at(p)->halite;
    if(h<=std::max(stop,P.min_cell_halite_consider)) return 0.0;
    int tr=m->calculate_distance(sp,p);
    auto mm=mine_sim(h,stop,cargo);
    if(mm.first<=0) return 0.0;
    int ret=nearest_drop_dist(m,p,ds);
    int total=tr+mm.second+ret;
    if(total<=0||total>left) return 0.0;

    double sc=mm.first/(double)total;
    if(p==sp) sc*=P.stay_bonus;
    sc-=P.long_distance_penalty*tr;
    int cong=congestion_around(m,p,std::max(1,P.congestion_radius));
    sc-=(P.congestion_penalty*cong)/100.0;
    return sc;
}

// Free a ship's origin cell the instant it commits to a move. The engine
// resolves all moves simultaneously, so once ship s has queued a move OFF its
// cell, that cell is genuinely empty this turn and a trailing ship may step into
// it. Without this, a vacated cell keeps reading is_occupied() for the rest of
// the turn's processing — which deadlocks the end-game dropoff (a loaded ship
// one step away can never enter the shipyard the turn its occupant leaves, so
// cargo never banks). Guard: only clear if the cell still holds THIS ship (i.e.
// nobody else has claimed it as a destination yet — impossible anyway, since a
// ship can't move into an as-yet-unvacated cell, but defensive).
static inline void free_origin(const std::unique_ptr<GameMap>& m, const std::shared_ptr<Ship>& s){
    auto* c = m->at(s->position);
    if(c->ship && c->ship.get() == s.get()) c->ship = nullptr;
}

// nav — moves ship toward dst, or stays still.
// avoid: secondary nav won't go to this cell (= previous cell, prevents 2-step oscillation).
static bool nav(const std::unique_ptr<GameMap>& m, std::shared_ptr<Ship> s, const Position& dst,
                std::vector<Command>& q, const BotParams& P, Position& actual,
                bool secondary_ok=true, const Position* avoid=nullptr,
                const std::vector<double>* threatMap=nullptr, int mapW=0, double threatW=0.0){
    // Threat-aware greedy step: among non-occupied neighbours (and staying put),
    // pick the one minimizing distance-to-dst + threatW*threat(cell). Routes
    // loaded ships around raiders instead of along the shortest (deadly) path.
    if(threatMap && threatW > 0.0){
        Position cur = m->normalize(s->position);
        // Small additive penalty on staying keeps ships making progress when a
        // step is no more dangerous than waiting.
        double bestCost = (double)m->calculate_distance(s->position, dst)
                        + threatW * (*threatMap)[cur.y*mapW + cur.x] + 0.25;
        bool haveMove=false; Direction bestDir=ALL_CARDINALS[0];
        for(auto d : ALL_CARDINALS){
            Position t = s->position.directional_offset(d);
            if(m->at(t)->is_occupied()) continue;
            if(avoid && t == *avoid) continue;
            Position tn = m->normalize(t);
            double cost = (double)m->calculate_distance(t, dst)
                        + threatW * (*threatMap)[tn.y*mapW + tn.x];
            if(cost < bestCost){ bestCost=cost; bestDir=d; haveMove=true; }
        }
        if(haveMove){
            Position t = s->position.directional_offset(bestDir);
            free_origin(m,s); m->at(t)->mark_unsafe(s); q.push_back(s->move(bestDir)); actual=t; return true;
        }
        m->at(s)->mark_unsafe(s); q.push_back(s->stay_still()); actual=s->position; return false;
    }

    auto primary = m->get_unsafe_moves(s->position, dst);
    for(auto d : primary){
        Position t = s->position.directional_offset(d);
        if(!m->at(t)->is_occupied()){
            free_origin(m,s); m->at(t)->mark_unsafe(s); q.push_back(s->move(d)); actual = t; return true;
        }
    }
    if(P.enable_secondary_nav && secondary_ok){
        // Phase 2: sideways only (neither preferred nor backtrack)
        for(auto d : ALL_CARDINALS){
            bool skip = false;
            for(auto pd : primary){ if(d == pd || d == invert_direction(pd)){ skip = true; break; } }
            if(skip) continue;
            Position t = s->position.directional_offset(d);
            if(avoid && t == *avoid) continue;
            if(!m->at(t)->is_occupied()){
                free_origin(m,s); m->at(t)->mark_unsafe(s); q.push_back(s->move(d)); actual = t; return true;
            }
        }
        // Phase 3: last resort — any non-primary, avoid prev cell
        for(auto d : ALL_CARDINALS){
            bool is_prim = false;
            for(auto pd : primary) if(d == pd){ is_prim = true; break; }
            if(is_prim) continue;
            Position t = s->position.directional_offset(d);
            if(avoid && t == *avoid) continue;
            if(!m->at(t)->is_occupied()){
                free_origin(m,s); m->at(t)->mark_unsafe(s); q.push_back(s->move(d)); actual = t; return true;
            }
        }
    }
    m->at(s)->mark_unsafe(s); q.push_back(s->stay_still()); actual = s->position; return false;
}

static bool evacuate_structure_cell(const std::shared_ptr<Player>& me,const std::unique_ptr<GameMap>& m,std::shared_ptr<Ship> s,std::vector<Command>& q, Position& actual){
    if(!is_our_structure(me, s->position)) return false;
    int best=-1; bool hasMove=false; Direction bestD=ALL_CARDINALS[0];
    for(auto d:ALL_CARDINALS){
        Position t=s->position.directional_offset(d);
        auto c=m->at(t);
        if(c->is_occupied()) continue;
        int score = m->at(t)->halite;
        if(is_our_structure(me, t)) score -= 100000;
        if(score>best){ best=score; bestD=d; hasMove=true; }
    }
    if(hasMove){
        Position t=s->position.directional_offset(bestD);
        free_origin(m,s); m->at(t)->mark_unsafe(s); q.push_back(s->move(bestD)); actual=t; return true;
    }
    return false;
}

static int max_ships_now(const BotParams&,const std::unique_ptr<GameMap>&,int){
    return 7;
}

static int spawn_cost_now(int currentShips){
    double factor = 1.0 + std::max(0.0, constants::SPAWN_COST_GROWTH) * (double)currentShips;
    if (constants::SPAWN_QUAD_GROWTH > 0.0 && currentShips + 1 > constants::SPAWN_QUAD_THRESHOLD) {
        int excess = currentShips + 1 - constants::SPAWN_QUAD_THRESHOLD;
        factor += constants::SPAWN_QUAD_GROWTH * (double)excess * (double)excess;
    }
    return (int)(constants::SHIP_COST * factor);
}

// ── Adaptive meta-strategy ───────────────────────────────────────────────
// A scheduler that exploits the iron triangle (aggro>eco>control>aggro): it
// scouts the opponent's archetype for a few dozen turns, then commits to the
// COUNTER profile. Each archetype is just a BotParams set, and the turn loop
// reads P fresh every turn, so "switching strategy" is a single assignment.
// Activated by an adaptive config file (ADAPTIVE=1) that points at the three
// profile files; otherwise the bot loads a single fixed profile as before.
struct AdaptiveCfg {
    bool enabled = false;
    BotParams aggro, eco, control;
    int scout_until = 45;        // hard classification deadline (turn)
    int aggro_commit_turn = 25;  // earliest turn to lock in "opp=aggro"
    // AGGRO is the only archetype distinguishable early: it pushes ships toward
    // us. Two signatures, either one fires: campers parked near our structures,
    // or a large mean enemy distance-from-their-home (raiders crossing the map).
    double camp_thresh = 0.35;       // avg campers within 6 of our structures
    double aggro_davg_thresh = 6.0;  // enemy mean dist from THEIR home
    // ECO and CONTROL are behaviorally identical early (both mine peacefully),
    // so when no aggro signature appears we play ECO: it's the Bayes-optimal
    // hedge over the eco/control ambiguity (beats control ~77%, mirrors eco ~50%;
    // playing aggro here would LOSE catastrophically if the opponent is control).
};

static AdaptiveCfg load_adaptive_cfg(const std::string& path){
    AdaptiveCfg AC;
    std::ifstream in(path);
    if(!in.good()) return AC;
    std::unordered_map<std::string,std::string> kv;
    std::string line;
    while(std::getline(in,line)){
        line=trim_copy(line);
        if(line.empty()||line[0]=='#') continue;
        auto eq=line.find('='); if(eq==std::string::npos) continue;
        kv[trim_copy(line.substr(0,eq))]=trim_copy(line.substr(eq+1));
    }
    auto it=kv.find("ADAPTIVE");
    if(it==kv.end() || !parse_bool(it->second)) return AC;  // plain fixed profile
    AC.enabled=true;
    fs::path base = fs::path(path).parent_path();
    auto resolve=[&](const std::string& p)->std::string{
        fs::path pp(p);
        return pp.is_absolute() ? pp.string() : (base/pp).string();
    };
    if(kv.count("PROFILE_AGGRO"))   AC.aggro   = load_bot_params(resolve(kv["PROFILE_AGGRO"]));
    if(kv.count("PROFILE_ECO"))     AC.eco     = load_bot_params(resolve(kv["PROFILE_ECO"]));
    if(kv.count("PROFILE_CONTROL")) AC.control = load_bot_params(resolve(kv["PROFILE_CONTROL"]));
    auto geti=[&](const char* k,int cur){ return kv.count(k)?std::stoi(kv[k]):cur; };
    auto getd=[&](const char* k,double cur){ return kv.count(k)?std::stod(kv[k]):cur; };
    AC.scout_until        = geti("SCOUT_UNTIL", AC.scout_until);
    AC.aggro_commit_turn  = geti("AGGRO_COMMIT_TURN", AC.aggro_commit_turn);
    AC.camp_thresh        = getd("CAMP_THRESH", AC.camp_thresh);
    AC.aggro_davg_thresh  = getd("AGGRO_DAVG_THRESH", AC.aggro_davg_thresh);
    return AC;
}

int main(int argc,char* argv[]){
    unsigned int seed=(argc>1)?(unsigned int)std::stoul(argv[1]):(unsigned int)time(nullptr);
    std::mt19937 rng(seed); (void)rng;
    std::string cfgPath=(argc>2)?std::string(argv[2]):std::string("bot_params.txt");
    AdaptiveCfg AC=load_adaptive_cfg(cfgPath);
    // Adaptive: open in ECO. Trace data shows aggro's raiders don't reach us
    // until ~turn 40, so an economy opening is safe through the scouting window
    // AND avoids handicapping the eco mirror; we flip to control the instant the
    // aggro signature fires. Fixed mode: load the single profile as before.
    BotParams P = AC.enabled ? AC.eco : load_bot_params(cfgPath);
    bool adapt_committed = false;
    double adapt_inv_sum = 0.0; int adapt_samples = 0;
    State S;

    Game game; game.ready(AC.enabled ? "MyCppBotAdaptive" : "MyCppBotV4Combat");

    for(;;){
        game.update_frame();
        auto me=game.me; auto& m=game.game_map; std::vector<Command> q;

        // ── Adaptive archetype detection ─────────────────────────────────────
        // Runs BEFORE any per-turn value reads P, so a commit takes effect the
        // same turn. Features: (1) invasion = enemy ships in OUR half (aggro
        // signature — it camps/raids our side); (2) ranging = enemy mean
        // distance from THEIR shipyard (eco fans out to mine the whole half;
        // control hugs home to cluster & defend). Counter map: opp aggro->we
        // play control, opp eco->aggro, opp control->eco.
        if(AC.enabled && !adapt_committed){
            // camp = enemy ships within CAMP_RADIUS of any of OUR structures
            // (the aggro signature — it parks raiders on our shipyard/dropoffs).
            // davg = enemy mean distance from THEIR OWN shipyard (eco fans out to
            // mine the whole half -> large; control clusters near home -> small).
            const int CAMP_RADIUS = 6;
            std::vector<Position> my_struct{me->shipyard->position};
            for(const auto& dp : me->dropoffs) my_struct.push_back(dp.second->position);
            int camp_now=0; double home_dist_sum=0.0; int eships=0;
            for(const auto& pl : game.players){
                if(pl->id==game.my_id) continue;
                Position ehome = pl->shipyard->position;
                for(const auto& kv : pl->ships){
                    Position ep = kv.second->position;
                    int dmin=1e9; for(const auto& sp : my_struct) dmin=std::min(dmin, m->calculate_distance(ep,sp));
                    if(dmin <= CAMP_RADIUS) camp_now++;
                    home_dist_sum += m->calculate_distance(ep, ehome); eships++;
                }
            }
            adapt_inv_sum += camp_now; adapt_samples++;
            double camp_avg = adapt_inv_sum / std::max(1, adapt_samples);
            double davg = eships ? home_dist_sum/eships : 0.0;
            bool aggro_sig = (camp_avg >= AC.camp_thresh) || (davg >= AC.aggro_davg_thresh);

            bool decide=false; const char* tag=nullptr;
            if(game.turn_number >= AC.aggro_commit_turn && aggro_sig){
                P = AC.control; tag="control (opp=AGGRO, early)"; decide=true;
            } else if(game.turn_number >= AC.scout_until){
                if(aggro_sig){ P=AC.control; tag="control (opp=AGGRO)"; }
                else         { P=AC.eco;     tag="eco (opp=ECO/CONTROL hedge)"; }
                decide=true;
            }
            if(decide){
                adapt_committed=true;
                hlt::log::log(std::string("ADAPT_COMMIT T=")+std::to_string(game.turn_number)
                    +" camp_avg="+std::to_string(camp_avg)
                    +" davg="+std::to_string(davg)+" choice="+tag);
            }
        }

        int left=constants::MAX_TURNS-game.turn_number;
        int stop=(int)(constants::MAX_HALITE*P.stop_ratio);
        int radius=std::max(1,radius_now(P,game.turn_number));
        auto drops=own_drops(me); auto edrops=enemy_drops(game);

        // Adaptive exploration based on ship-count dominance.
        // When we field substantially more ships than any live enemy, relax (or
        // disable) the territorial filter and grow the search radius so ships
        // venture beyond the midline instead of circling our own half.
        const int my_ship_count = (int)me->ships.size();
        const int enemy_ship_count = alive_enemy_ship_count(game);
        const double dominance = (enemy_ship_count > 0)
            ? (double)my_ship_count / (double)enemy_ship_count
            : 1e9; // no live enemy = total dominance
        int margin_eff = P.safe_margin;
        bool filter_off = false;
        if (dominance >= P.dominance_disable_ratio) {
            filter_off = true;
        } else if (dominance >= P.dominance_relax_ratio) {
            margin_eff = 0;
        }
        const int radius_eff = std::max(1, radius
            + ((dominance >= P.dominance_relax_ratio) ? P.expand_radius_bonus : 0));

        // ── Clean up dead ships ──────────────────────────────────────────────
        std::set<EntityId> alive;
        for(const auto& kv:me->ships) alive.insert(kv.first);
        for(auto it=S.mode.begin();it!=S.mode.end();){
            if(!alive.count(it->first)){
                S.target.erase(it->first); S.ttl.erase(it->first);
                S.hunt_id.erase(it->first); S.camp_pos.erase(it->first);
                it=S.mode.erase(it);
            } else ++it;
        }
        for(const auto& kv:me->ships){
            if(!S.mode.count(kv.first)){ S.mode[kv.first]=Mode::Collect; S.ttl[kv.first]=0; }
        }

        // ── Build enemy ship list (sorted richest first) ─────────────────────
        struct EnemyInfo { EntityId id; PlayerId owner; Position pos; int halite; int hp; };
        std::vector<EnemyInfo> enemy_ships;
        for(const auto& player : game.players){
            if(player->id==game.my_id) continue;
            for(const auto& kv : player->ships)
                enemy_ships.push_back({kv.first, player->id, kv.second->position,
                                       kv.second->halite, kv.second->hp});
        }
        std::sort(enemy_ships.begin(),enemy_ships.end(),[](const auto& a,const auto& b){
            return a.halite > b.halite;
        });

        // Helper: find an enemy Ship object by ID
        auto find_enemy = [&](EntityId eid) -> std::shared_ptr<Ship> {
            for(const auto& player : game.players){
                if(player->id==game.my_id) continue;
                auto it=player->ships.find(eid);
                if(it!=player->ships.end()) return it->second;
            }
            return nullptr;
        };

        // ── Sort our ships ───────────────────────────────────────────────────
        std::vector<std::shared_ptr<Ship>> ships;
        ships.reserve(me->ships.size());
        for(const auto& kv:me->ships) ships.push_back(kv.second);
        std::sort(ships.begin(),ships.end(),[&](const auto&a,const auto&b){
            bool aReturn=(S.mode[a->id]==Mode::Return), bReturn=(S.mode[b->id]==Mode::Return);
            int aHome=nearest_drop_dist(m,a->position,drops), bHome=nearest_drop_dist(m,b->position,drops);
            bool aOnStruct=is_our_structure(me,a->position), bOnStruct=is_our_structure(me,b->position);
            if(aOnStruct!=bOnStruct) return aOnStruct;
            if(aReturn!=bReturn) return aReturn;
            if(aReturn && bReturn && aHome!=bHome) return aHome<bHome;
            return a->halite>b->halite;
        });

        // ── Spawn / economy ──────────────────────────────────────────────────
        int avgLocal=local_sum(m,me->shipyard->position,3)/49;
        int maxShips=max_ships_now(P,m,(int)me->dropoffs.size());
        int dynamicMinTurnsLeft = std::max(20,(int)std::round(constants::MAX_TURNS*P.spawn_cutoff_turn_ratio));
        int minTurnsLeftForSpawn = std::min(P.spawn_min_turns_left, dynamicMinTurnsLeft);
        const int effectiveSpawnCost = spawn_cost_now((int)me->ships.size());
        bool spawnIntent=(game.turn_number<=(int)(constants::MAX_TURNS*P.spawn_end_ratio))
                       &&(left>=minTurnsLeftForSpawn)
                       &&((int)me->ships.size()<maxShips)
                       &&(me->halite>=effectiveSpawnCost);
        if(spawnIntent && P.enable_spawn_roi){
            double minedPerTurn = avgLocal / (double)std::max(1, constants::EXTRACT_RATIO);
            double paybackTurns = effectiveSpawnCost / std::max(1.0, minedPerTurn);
            if(left < P.spawn_payback_turns) paybackTurns *= 1.25;
            double roiScore = P.spawn_payback_turns / std::max(1.0, paybackTurns);
            spawnIntent = roiScore >= P.spawn_roi_min;
        }
        // Anti death-spiral: if the enemy outnumbers us by the margin, stop
        // spawning — replacement ships would die before banking, draining the
        // treasury to ~0 (the diagnosed score=1 collapse). Preserve it instead.
        if(spawnIntent && (enemy_ship_count - my_ship_count) >= P.spawn_stop_if_behind_by){
            spawnIntent = false;
        }

        // ── Hunt management ──────────────────────────────────────────────────
        // Validate existing hunt assignments; count hunters committed per enemy
        // (a pack can put several ships on one target).
        std::unordered_map<EntityId,int> hunted_count;
        int hunterCount = 0;
        for(const auto& kv : me->ships){
            if(!S.mode.count(kv.first) || S.mode[kv.first]!=Mode::Hunt) continue;
            bool hunt_valid = false;
            if(S.hunt_id.count(kv.first)){
                EntityId eid = S.hunt_id[kv.first];
                for(const auto& e : enemy_ships){
                    if(e.id==eid){ hunt_valid=true; break; }
                }
            }
            if(!hunt_valid){
                S.mode[kv.first]=Mode::Collect;
                S.hunt_id.erase(kv.first);
            } else {
                hunted_count[S.hunt_id[kv.first]]++;
                hunterCount++;
            }
        }

        // Assign new hunters from idle collect ships. Concentrate up to
        // P.hunters_per_target on the SAME enemy (richest first) so the pack can
        // land a 2-hit KILL (2x70 > 100 HP), not a lone bruise the target just
        // walks away from. A claimed ship flips to Hunt and is excluded from the
        // idle pool on the next pick, so no ship is double-assigned.
        if(P.enable_attack && constants::ENABLE_COMBAT_COMMANDS && game.turn_number >= P.hunt_min_turn){
            for(const auto& enemy : enemy_ships){
                if(hunterCount >= P.hunt_max_hunters) break;
                if(enemy.halite < P.hunt_min_enemy_halite) break; // sorted descending
                while(hunted_count[enemy.id] < P.hunters_per_target
                      && hunterCount < P.hunt_max_hunters){
                    std::shared_ptr<Ship> best_hunter = nullptr;
                    int best_dist = P.hunt_max_range + 1;
                    for(const auto& ship : ships){
                        if(S.mode[ship->id]!=Mode::Collect) continue;
                        if(ship->halite > P.attack_max_self_halite_for_risk) continue;
                        int d=m->calculate_distance(ship->position, enemy.pos);
                        if(d <= P.hunt_max_range && d < best_dist){ best_dist=d; best_hunter=ship; }
                    }
                    if(!best_hunter) break;  // no more reachable idle ships for this target
                    S.mode[best_hunter->id]=Mode::Hunt;
                    S.hunt_id[best_hunter->id]=enemy.id;
                    if(S.target.count(best_hunter->id)){ S.target.erase(best_hunter->id); }
                    S.ttl[best_hunter->id]=0;
                    hunted_count[enemy.id]++;
                    hunterCount++;
                    hlt::log::log("HUNT_ASSIGN T="+std::to_string(game.turn_number)
                        +" hunter="+std::to_string(best_hunter->id)
                        +" enemy="+std::to_string(enemy.id)
                        +" enemy_halite="+std::to_string(enemy.halite)
                        +" pack="+std::to_string(hunted_count[enemy.id])
                        +" dist="+std::to_string(best_dist));
                }
            }
        }

        // ── Camp assignment ─────────────────────────────────────────────────
        // When camp is enabled, idle ships are posted near enemy structures
        // (shipyard + dropoffs) instead of chasing individual ships.  This
        // eliminates the travel-time bottleneck: targets come to the campers.
        int camperCount = 0;
        if(P.camp_enabled && game.turn_number >= P.camp_assign_turn){
            // Validate existing campers; release those whose structure is gone
            for(const auto& kv : me->ships){
                if(!S.mode.count(kv.first) || S.mode[kv.first]!=Mode::Camp) continue;
                camperCount++;
            }
            // Find enemy structures
            std::vector<Position> enemy_structures;
            for(const auto& player : game.players){
                if(player->id==game.my_id) continue;
                // Shipyard is always known (initial state has it)
                enemy_structures.push_back(player->shipyard->position);
                for(const auto& dp : player->dropoffs)
                    enemy_structures.push_back(dp.second->position);
            }
            // Assign new campers from idle Collect ships
            int maxCampers = (int)enemy_structures.size() * P.campers_per_structure;
            for(const auto& struct_pos : enemy_structures){
                int at_this = 0;
                for(const auto& kv : me->ships){
                    if(S.mode.count(kv.first) && S.mode[kv.first]==Mode::Camp
                       && S.camp_pos.count(kv.first)
                       && S.camp_pos[kv.first] == struct_pos) at_this++;
                }
                for(const auto& ship : ships){
                    if(camperCount >= maxCampers) break;
                    if(at_this >= P.campers_per_structure) break;
                    if(S.mode[ship->id]!=Mode::Collect) continue;
                    // Pick a camp cell: any passable cell within attack_range of the structure
                    Position best_camp{-1,-1};
                    int best_d = P.attack_range + 2;
                    for(int dy=-P.attack_range; dy<=P.attack_range; dy++){
                        for(int dx=-P.attack_range; dx<=P.attack_range; dx++){
                            int md = abs(dx)+abs(dy);
                            if(md < 1 || md > P.attack_range) continue;
                            Position p{struct_pos.x+dx, struct_pos.y+dy};
                            if(p.x<0||p.x>=m->width||p.y<0||p.y>=m->height) continue;
                            if(!m->at(p)->is_empty() && !(p==ship->position)) continue;
                            int sd = m->calculate_distance(ship->position, p);
                            if(sd < best_d){ best_d=sd; best_camp=p; }
                        }
                    }
                    if(best_camp.x < 0) continue;
                    S.mode[ship->id]=Mode::Camp;
                    S.camp_pos[ship->id]=struct_pos;  // remember which structure
                    S.target[ship->id]=best_camp;       // navigate to camp cell
                    if(S.hunt_id.count(ship->id)) S.hunt_id.erase(ship->id);
                    camperCount++; at_this++;
                    hlt::log::log("CAMP_ASSIGN T="+std::to_string(game.turn_number)
                        +" ship="+std::to_string(ship->id)
                        +" structure=("+std::to_string(struct_pos.x)+","+std::to_string(struct_pos.y)+")"
                        +" camp=("+std::to_string(best_camp.x)+","+std::to_string(best_camp.y)+")");
                }
            }
        }

        // ── Claimed / targeted sets ──────────────────────────────────────────
        std::unordered_set<Position> claimed;
        std::unordered_set<Position> targeted;
        for(const auto& kv:me->ships){
            if(S.target.count(kv.first)) targeted.insert(S.target[kv.first]);
        }

        auto clearTarget = [&](EntityId id){
            if(S.target.count(id)){ targeted.erase(S.target[id]); S.target.erase(id); }
            S.ttl[id]=0;
        };

        int turnShips=(int)ships.size(), turnStay=0, turnNavFail=0, turnStructOccupyStart=0;
        int turnAttacks=0;

        // ── Threat map ────────────────────────────────────────────────────────
        // threat[y*W+x] = danger of occupying (x,y): enemy ships that can attack
        // it this turn (adjacent, weight 1.0) or reach-and-attack next (dist 2,
        // weight 0.4). Built only when threat-aware navigation is enabled.
        std::vector<double> threatMap;
        if(P.threat_avoid_weight > 0.0){
            threatMap.assign((size_t)m->width * m->height, 0.0);
            for(const auto& player : game.players){
                if(player->id==game.my_id) continue;
                for(const auto& kv : player->ships){
                    Position ep = kv.second->position;
                    for(int dy=-2; dy<=2; ++dy) for(int dx=-2; dx<=2; ++dx){
                        Position c = m->normalize(Position{ep.x+dx, ep.y+dy});
                        int d = m->calculate_distance(ep, c);
                        if(d==0 || d>2) continue;
                        threatMap[(size_t)c.y*m->width + c.x] += (d==1) ? 1.0 : 0.4;
                    }
                }
            }
        }

        // ── Focus-fire coordination ──────────────────────────────────────────
        // If >= focus_fire_min_ships of our ships are adjacent to the same enemy
        // ship, commit all of them to attack it this turn. Concentrated fire
        // (N x ATTACK_HP_DAMAGE) kills the raider before it can trade, and our
        // ships survive to heal at home. The coordinated counter to aggression.
        std::unordered_map<EntityId, std::shared_ptr<Ship>> focusTarget; // our ship -> enemy
        if(P.focus_fire_min_ships > 0 && P.enable_attack && constants::ENABLE_COMBAT_COMMANDS){
            for(const auto& player : game.players){
                if(player->id==game.my_id) continue;
                for(const auto& ek : player->ships){
                    auto enemy=ek.second;
                    // Ships that can strike this enemy now (adjacent, healthy enough).
                    std::vector<std::shared_ptr<Ship>> adj;
                    int ourNear=0;
                    for(const auto& mk : me->ships){
                        auto myship=mk.second;
                        int d=m->calculate_distance(myship->position, enemy->position);
                        if(d<=2) ourNear++;
                        if(d <= P.attack_range && myship->hp >= P.attack_min_self_hp) adj.push_back(myship);
                    }
                    // Enemy force in the same area (incl. target) — its potential support.
                    int enemyNear=0;
                    for(const auto& p2 : game.players){
                        if(p2->id==game.my_id) continue;
                        for(const auto& e2 : p2->ships)
                            if(m->calculate_distance(e2.second->position, enemy->position)<=2) enemyNear++;
                    }
                    // Engage ONLY with clear local superiority: enough strikers AND
                    // we outnumber the enemy's local cluster. Never feed a losing fight.
                    if((int)adj.size() >= P.focus_fire_min_ships && ourNear > enemyNear){
                        for(const auto& s : adj){
                            if(!focusTarget.count(s->id)) focusTarget[s->id]=enemy;
                        }
                    }
                }
            }
        }

        // ── Main ship loop ───────────────────────────────────────────────────
        for(auto ship:ships){
            if(is_our_structure(me, ship->position)) turnStructOccupyStart++;
            Position home=nearest_drop(m,ship->position,drops);
            int hd=m->calculate_distance(ship->position,home);

            // HP-aware return: damaged ship retreats to base to heal
            if(ship->hp < P.low_hp_return && S.mode[ship->id]!=Mode::Return){
                S.mode[ship->id]=Mode::Return;
                S.hunt_id.erase(ship->id);
                S.camp_pos.erase(ship->id);
                clearTarget(ship->id);
            }

            // End-game and return triggers
            if(left<=P.end_rush_turns && ship->halite>0){
                S.mode[ship->id]=Mode::Return;
                S.hunt_id.erase(ship->id); clearTarget(ship->id);
            }
            if(left<=hd+P.return_buffer){ S.mode[ship->id]=Mode::Return; S.hunt_id.erase(ship->id); clearTarget(ship->id); }
            if(ship->halite>=rr(P,hd)){ S.mode[ship->id]=Mode::Return; S.hunt_id.erase(ship->id); clearTarget(ship->id); }
            if(S.mode[ship->id]==Mode::Return && ship->position==home){
                S.mode[ship->id]=Mode::Collect; clearTarget(ship->id);
            }

            Position actual=ship->position;

            // Hard rule: evacuate own structure cell
            if(evacuate_structure_cell(me,m,ship,q,actual)){
                claimed.insert(actual); continue;
            }

            // ── Focus-fire: commit to the coordinated attack assigned above ───
            if(focusTarget.count(ship->id)){
                auto enemy=focusTarget[ship->id];
                m->at(ship)->mark_unsafe(ship);
                q.push_back(ship->attack(enemy));
                actual=ship->position;
                turnAttacks++; turnStay++;
                claimed.insert(actual);
                continue;
            }

            // ── Defensive immunity ────────────────────────────────────────────
            // A loaded ship with a raider adjacent issues 'defend' (full attack
            // immunity this turn) instead of mining/moving, protecting its cargo
            // and itself. Hard-counters raiders that can't kill what they can't
            // damage. Off by default (defend_min_cargo huge); the CONTROL profile
            // enables it via DEFEND_MIN_CARGO.
            // Defend only while staying is already the plan: productively mining a
            // rich cell, or (optionally) sitting on high cargo. NEVER defend in
            // Return mode — a loaded ship must keep moving home to BANK its cargo,
            // because unbanked cargo scores nothing (defending-with-cargo = freeze
            // = economic collapse, the diagnosed failure vs raiders).
            bool defendWorthwhile = (ship->halite >= P.defend_min_cargo)
                || (!ship->is_full() && m->at(ship)->halite >= P.defend_min_cell_halite);
            if(constants::ENABLE_COMBAT_COMMANDS
               && defendWorthwhile
               && S.mode[ship->id]!=Mode::Return
               && !is_our_structure(me, ship->position)){
                bool enemy_adjacent=false;
                for(const auto& player : game.players){
                    if(player->id==game.my_id) continue;
                    for(const auto& kv : player->ships){
                        if(m->calculate_distance(ship->position, kv.second->position) <= P.defend_trigger_range){
                            enemy_adjacent=true; break;
                        }
                    }
                    if(enemy_adjacent) break;
                }
                if(enemy_adjacent){
                    m->at(ship)->mark_unsafe(ship);
                    q.push_back(ship->defend());
                    turnStay++;
                    claimed.insert(ship->position);
                    continue;
                }
            }

            // ── Hunt mode ───────────────────────────────────────────────────
            if(S.mode[ship->id]==Mode::Hunt){
                std::shared_ptr<Ship> enemy_ship=nullptr;
                if(S.hunt_id.count(ship->id))
                    enemy_ship=find_enemy(S.hunt_id[ship->id]);

                if(!enemy_ship){
                    // Target gone — revert to collect
                    S.mode[ship->id]=Mode::Collect;
                    S.hunt_id.erase(ship->id);
                    clearTarget(ship->id);
                    // fall through to collect logic below
                } else {
                    int dist=m->calculate_distance(ship->position, enemy_ship->position);
                    if(dist <= P.attack_range && ship->hp >= P.attack_min_self_hp){
                        // In range — attack!
                        m->at(ship)->mark_unsafe(ship);
                        q.push_back(ship->attack(enemy_ship));
                        actual=ship->position;
                        turnAttacks++;
                        hlt::log::log("ATTACK T="+std::to_string(game.turn_number)
                            +" hunter="+std::to_string(ship->id)
                            +" enemy="+std::to_string(enemy_ship->id)
                            +" enemy_halite="+std::to_string(enemy_ship->halite)
                            +" our_hp="+std::to_string(ship->hp));
                    } else if(dist==0){
                        // Same cell (collision) — stay
                        m->at(ship)->mark_unsafe(ship);
                        q.push_back(ship->stay_still());
                        actual=ship->position;
                    } else {
                        // Navigate toward enemy's current position
                        S.target[ship->id]=enemy_ship->position;
                        const Position* avP=S.prev.count(ship->id)?&S.prev[ship->id]:nullptr;
                        if(can_move(m,ship)){ if(!nav(m,ship,enemy_ship->position,q,P,actual,true,avP)) turnNavFail++; }
                        else{ m->at(ship)->mark_unsafe(ship); q.push_back(ship->stay_still()); actual=ship->position; }
                    }
                    if(actual==ship->position) turnStay++;
                    claimed.insert(actual);
                    continue;
                }
            }

            // ── Camp mode ───────────────────────────────────────────────────
            if(S.mode[ship->id]==Mode::Camp){
                // Attack an enemy ship within range, respecting wolfpack limits.
                // Count how many campers are already committed to each enemy
                // this turn so we don't waste 7 attacks on the same ship.
                std::unordered_map<EntityId,int> camper_on_target;
                for(const auto& mk : me->ships){
                    if(S.mode.count(mk.first) && S.mode[mk.first]==Mode::Camp
                       && S.hunt_id.count(mk.first))
                        camper_on_target[S.hunt_id[mk.first]]++;
                }
                std::shared_ptr<Ship> best_target=nullptr;
                int best_halite = P.attack_min_target_halite - 1;
                for(const auto& player : game.players){
                    if(player->id==game.my_id) continue;
                    for(const auto& kv : player->ships){
                        int d = m->calculate_distance(ship->position, kv.second->position);
                        if(d <= P.attack_range && kv.second->halite > best_halite
                           && ship->hp >= P.attack_min_self_hp
                           && camper_on_target[kv.first] < P.hunters_per_target){
                            best_halite = kv.second->halite;
                            best_target = kv.second;
                        }
                    }
                }
                if(best_target){
                    S.hunt_id[ship->id] = best_target->id; // track which target
                    m->at(ship)->mark_unsafe(ship);
                    q.push_back(ship->attack(best_target));
                    actual = ship->position;
                    turnAttacks++;
                    hlt::log::log("CAMP_ATK T="+std::to_string(game.turn_number)
                        +" ship="+std::to_string(ship->id)
                        +" enemy="+std::to_string(best_target->id)
                        +" enemy_halite="+std::to_string(best_target->halite)
                        +" pack="+std::to_string(camper_on_target[best_target->id]+1));
                } else if(S.target.count(ship->id)){
                    // Navigate to camp cell
                    Position camp = S.target[ship->id];
                    int cd = m->calculate_distance(ship->position, camp);
                    if(cd <= 1){
                        m->at(ship)->mark_unsafe(ship);
                        q.push_back(ship->stay_still());
                        actual = ship->position;
                    } else {
                        const Position* avP = S.prev.count(ship->id)?&S.prev[ship->id]:nullptr;
                        if(can_move(m,ship)){ if(!nav(m,ship,camp,q,P,actual,true,avP)) turnNavFail++; }
                        else{ m->at(ship)->mark_unsafe(ship); q.push_back(ship->stay_still()); actual=ship->position; }
                    }
                } else {
                    m->at(ship)->mark_unsafe(ship);
                    q.push_back(ship->stay_still());
                    actual = ship->position;
                }
                if(actual==ship->position) turnStay++;
                claimed.insert(actual);
                continue;
            }

            // ── Return mode ─────────────────────────────────────────────────
            if(S.mode[ship->id]==Mode::Return){
                const Position* avP=S.prev.count(ship->id)?&S.prev[ship->id]:nullptr;
                const std::vector<double>* tm=(P.threat_avoid_weight>0.0)?&threatMap:nullptr;
                if(can_move(m,ship)){ if(!nav(m,ship,home,q,P,actual,true,avP,tm,m->width,P.threat_avoid_weight)) turnNavFail++; }
                else{ m->at(ship)->mark_unsafe(ship); q.push_back(ship->stay_still()); actual=ship->position; }
                if(actual==ship->position) turnStay++;
                claimed.insert(actual);
                continue;
            }

            // ── Collect mode ─────────────────────────────────────────────────

            // Stay to mine if cell is rich enough
            if(m->at(ship)->halite>stop && !ship->is_full()){
                m->at(ship)->mark_unsafe(ship);
                q.push_back(ship->stay_still());
                turnStay++; claimed.insert(ship->position);
                continue;
            }

            // Opportunistic attack: adjacent rich enemy
            if(P.enable_attack && constants::ENABLE_COMBAT_COMMANDS
               && ship->hp >= P.attack_min_self_hp
               && ship->halite <= P.attack_max_self_halite_for_risk){
                std::shared_ptr<Ship> best_target=nullptr;
                int best_halite=P.attack_min_target_halite - 1;
                for(const auto& player : game.players){
                    if(player->id==game.my_id) continue;
                    for(const auto& kv : player->ships){
                        if(m->calculate_distance(ship->position, kv.second->position) <= P.attack_range
                           && kv.second->halite > best_halite){
                            best_halite=kv.second->halite;
                            best_target=kv.second;
                        }
                    }
                }
                if(best_target){
                    m->at(ship)->mark_unsafe(ship);
                    q.push_back(ship->attack(best_target));
                    actual=ship->position;
                    turnAttacks++; turnStay++;
                    claimed.insert(actual);
                    hlt::log::log("ATTACK_OPP T="+std::to_string(game.turn_number)
                        +" ship="+std::to_string(ship->id)
                        +" enemy="+std::to_string(best_target->id)
                        +" enemy_halite="+std::to_string(best_halite)
                        +" our_hp="+std::to_string(ship->hp));
                    continue;
                }
            }

            // Find best mining target
            Position best=ship->position; double bestSc=-1e18; bool needScan=true;

            if(S.target.count(ship->id)){
                Position cur=S.target[ship->id];
                int curH=m->at(cur)->halite;
                if(curH>stop && !claimed.count(cur)){
                    int tr=m->calculate_distance(ship->position,cur);
                    int ret=nearest_drop_dist(m,cur,drops);
                    if(tr+2+ret<=left){ best=cur; needScan=false; }
                }
                if(needScan){ clearTarget(ship->id); }
            }

            if(needScan){
                for(int dy=-radius_eff;dy<=radius_eff;++dy) for(int dx=-radius_eff;dx<=radius_eff;++dx){
                    Position p=m->normalize(Position{ship->position.x+dx,ship->position.y+dy});
                    if(claimed.count(p)||targeted.count(p)) continue;
                    double sc=score_cell(m,p,ship->position,drops,stop,ship->halite,left,P);
                    // Soft territorial penalty: cells closer to the enemy than to
                    // us cost score, but a rich-enough cell can still win.  When
                    // we dominate the enemy, the filter is relaxed/disabled.
                    if(!filter_off && !edrops.empty()){
                        int ourd=nearest_drop_dist(m,p,drops);
                        int ed=1e9; for(const auto& e:edrops) ed=std::min(ed,m->calculate_distance(p,e.second));
                        int gap=ed-ourd;
                        if(gap<margin_eff){
                            sc -= P.invasion_penalty * (double)(margin_eff - gap);
                        }
                    }
                    if(spawnIntent && m->calculate_distance(p,me->shipyard->position)<=1) sc -= P.shipyard_target_penalty;
                    if(sc>bestSc){ bestSc=sc; best=p; }
                }
                if(best!=ship->position){ S.target[ship->id]=best; targeted.insert(best); }
            }

            if(best==ship->position){
                m->at(ship)->mark_unsafe(ship);
                q.push_back(ship->stay_still());
                turnStay++; claimed.insert(ship->position);
                continue;
            }

            // NOTE: threat-aware nav is applied to Return (protect loaded cargo),
            // NOT to outbound mining — avoiding threat while seeking ore makes
            // ships too timid to mine contested cells and stalls the economy.
            const Position* avP=S.prev.count(ship->id)?&S.prev[ship->id]:nullptr;
            if(can_move(m,ship)){ if(!nav(m,ship,best,q,P,actual,true,avP)) turnNavFail++; }
            else{ m->at(ship)->mark_unsafe(ship); q.push_back(ship->stay_still()); actual=ship->position; }
            if(actual==ship->position){
                turnStay++;
                S.ttl[ship->id]++;
                if(S.ttl[ship->id]>=P.stuck_reset_turns) clearTarget(ship->id);
            } else {
                S.ttl[ship->id]=0;
            }
            claimed.insert(actual);
        }

        if(spawnIntent && m->at(me->shipyard)->is_occupied()) spawnIntent=false;
        if(spawnIntent) q.push_back(me->shipyard->spawn());

        // ── Periodic diagnostics ─────────────────────────────────────────────
        if(P.enable_periodic_log && P.log_period>0 && game.turn_number%P.log_period==0){
            int cargo=0; for(const auto& kv:me->ships) cargo+=kv.second->halite;
            hlt::log::log("T="+std::to_string(game.turn_number)
                +" ships="+std::to_string(me->ships.size())
                +" drops="+std::to_string(me->dropoffs.size())
                +" bank="+std::to_string(me->halite)
                +" cargo="+std::to_string(cargo)
                +" hunters="+std::to_string(hunterCount));
        }
        int turnStuck2=0;
        for(const auto& kv : me->ships){
            int ttl=S.ttl.count(kv.first)?S.ttl[kv.first]:0;
            if(ttl>=2){
                turnStuck2++;
                auto& pos=kv.second->position;
                hlt::log::log("STUCK2 T="+std::to_string(game.turn_number)
                    +" ship="+std::to_string(kv.first)
                    +" pos=("+std::to_string(pos.x)+","+std::to_string(pos.y)+")"
                    +" mode="+(S.mode.count(kv.first)&&S.mode[kv.first]==Mode::Return?"R":"C"));
            }
            S.prev[kv.first]=kv.second->position;
        }
        if(turnShips>0){
            hlt::log::log("METRIC T="+std::to_string(game.turn_number)
                +" stay="+std::to_string(turnStay)
                +" ships="+std::to_string(turnShips)
                +" nav_fail="+std::to_string(turnNavFail)
                +" struct_occ="+std::to_string(turnStructOccupyStart)
                +" stuck2="+std::to_string(turnStuck2)
                +" attacks="+std::to_string(turnAttacks));
        }
        if(!game.end_turn(q)) break;
    }
    return 0;
}
