import zstandard as zstd, json, sys

fname = sys.argv[1]
with open(fname, "rb") as f:
    dctx = zstd.ZstdDecompressor()
    data = dctx.decompress(f.read(), max_output_size=50*1024*1024)
replay = json.loads(data)

players = replay["players"]
if isinstance(players, list):
    players = {str(p["player_id"]): p for p in players}

for pid, player in players.items():
    sy = player.get("factory_location") or player.get("shipyard", {})
    print("P" + pid + " shipyard: (" + str(sy["x"]) + ", " + str(sy["y"]) + ")")

stats = replay.get("game_statistics", {}).get("player_statistics", [])
for s in stats:
    print("\nP" + str(s["player_id"]) + " stats:")
    print("  final_production:", s["final_production"])
    print("  total_production:", s["total_production"])
    print("  total_mined:", s["total_mined"])
    print("  halite_per_dropoff:", s["halite_per_dropoff"])
    print("  ships_spawned:", s["ships_spawned"])
    print("  last_turn_ship_spawn:", s["last_turn_ship_spawn"])

turns = replay["full_frames"]
print("\nTurn | P0 ships | P1 ships | P0 deposited | P1 deposited")
for i in [2, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400]:
    if i >= len(turns): break
    frame = turns[i]
    ents = frame.get("entities", {})
    dep = frame.get("deposited", {})
    p0_ships = len(ents.get("0", {}))
    p1_ships = len(ents.get("1", {}))
    print(f"  {i:4d} | {p0_ships:8d} | {p1_ships:8d} | {dep.get('0',0):12d} | {dep.get('1',0):12d}")

# Check P0's dropoff ship at early turns
print("\nP0 ships at turn 10:")
for sid, ship in turns[9].get("entities", {}).get("0", {}).items():
    print("  ship", sid, "at", (ship["x"], ship["y"]), "cargo:", ship["energy"])

print("\nP0 ships at turn 60:")
for sid, ship in turns[59].get("entities", {}).get("0", {}).items():
    print("  ship", sid, "at", (ship["x"], ship["y"]), "cargo:", ship["energy"])
