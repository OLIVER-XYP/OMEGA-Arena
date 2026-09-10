import json
for name in ("eco_vs_control_bc_eco", "eco_vs_control_bc_control"):
    try:
        d = json.load(open(f"/home/hp/halite-rl/rl_ai/data/{name}/meta.json"))
        print(name, "games", d["n_games"], "frames", d["n_frames"],
              "ships", d["n_ship_decisions"], "schema", d["feature_schema"])
    except Exception as exc:
        print(name, "MISSING", exc)
