"""
measure_combat.py
Parse .hlt replay files to measure combat halite economics:
  - Halite stolen by attacker (goes directly to attacker ship)
  - Halite dropped on map when ship killed (contested, wasted)
  - Attack events: hits, misses, kills
Run from E:\Halite-III\starter_kits\C++
"""
import json, os, struct, sys, glob, gzip

def load_replay(path):
    """Load an .hlt replay. May be gzipped JSON or plain JSON."""
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        pass
    with open(path, 'rb') as f:
        raw = f.read()
    # Some versions store length-prefixed JSON
    try:
        return json.loads(raw)
    except Exception:
        # Try skipping a 4-byte length prefix
        return json.loads(raw[4:])

def analyze_replay(path):
    try:
        data = load_replay(path)
    except Exception as e:
        print(f"  [ERR] {path}: {e}")
        return None

    n_players = len(data.get('players', []))
    turns = data.get('full_frames', data.get('frames', []))

    stats = {
        'attacks_hit': 0,
        'attacks_miss': 0,
        'kills': 0,
        'halite_stolen': 0,       # goes directly to attacker
        'halite_dropped_kill': 0, # drops on map when ship dies from attack
        'halite_dropped_coll': 0, # drops on map from collision
    }

    # Track entity halite over time to compute what was dropped on kill
    entity_halite = {}  # entity_id -> halite at last frame

    for frame in turns:
        events = frame.get('events', [])

        # Update entity halite snapshot from frame entities
        entities = frame.get('entities', {})
        for pid_str, ships in entities.items():
            for eid_str, ship_info in ships.items():
                entity_halite[eid_str] = ship_info.get('energy', 0)

        for ev in events:
            t = ev.get('type', '')
            if t == 'attack':
                if ev.get('hit', False):
                    stats['attacks_hit'] += 1
                    # Halite stolen = ATTACK_HALITE_STEAL_RATIO * target halite at time of hit
                    # We can't compute exactly from replay (no per-attack steal logged),
                    # but we CAN count hit vs miss and kill events
                else:
                    stats['attacks_miss'] += 1
            elif t == 'death' or t == 'collision':
                ships_dead = ev.get('ships', ev.get('ship_ids', []))
                for eid in ships_dead:
                    eid_s = str(eid)
                    dropped = entity_halite.get(eid_s, 0)
                    if t == 'death':
                        stats['kills'] += 1
                        stats['halite_dropped_kill'] += dropped
                    else:
                        stats['halite_dropped_coll'] += dropped

    return stats

def main():
    replay_dir = os.path.join(os.path.dirname(__file__), 'eval_runs')
    replays = glob.glob(os.path.join(replay_dir, '*.hlt'))
    if not replays:
        print("No replays found in eval_runs/. Run games without --no-replay first.")
        return

    totals = {k: 0 for k in ['attacks_hit','attacks_miss','kills',
                               'halite_stolen','halite_dropped_kill','halite_dropped_coll']}
    n = 0
    for rp in replays:
        s = analyze_replay(rp)
        if s:
            n += 1
            for k in totals:
                totals[k] += s[k]
            print(f"  {os.path.basename(rp)}: hits={s['attacks_hit']} miss={s['attacks_miss']} "
                  f"kills={s['kills']} dropped_kill={s['halite_dropped_kill']} dropped_coll={s['halite_dropped_coll']}")

    if n > 0:
        print(f"\n=== Totals across {n} replays ===")
        print(f"  Attack hits  : {totals['attacks_hit']}")
        print(f"  Attack misses: {totals['attacks_miss']}")
        print(f"  Kills        : {totals['kills']}")
        print(f"  Halite dropped on map (kills) : {totals['halite_dropped_kill']}")
        print(f"  Halite dropped on map (collisions): {totals['halite_dropped_coll']}")
        if totals['kills'] > 0:
            print(f"  Avg halite wasted per kill: {totals['halite_dropped_kill']//totals['kills']}")

if __name__ == '__main__':
    main()
