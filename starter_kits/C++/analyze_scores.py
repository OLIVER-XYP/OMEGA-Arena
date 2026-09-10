import csv
from collections import defaultdict

# Look at raw score distributions for base vs i20t100, opponent_N=20
buckets = defaultdict(list)  # key: (tag, N) -> list of (s12, sN)

with open("h2h_econ100_partial.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        buckets[(row['tag'], int(row['opponent_N']))].append(
            (int(row['s12']), int(row['sN']))
        )

print(f"{'tag':<10} {'N':>3} {'avg_s12':>9} {'avg_sN':>9} {'diff':>8} {'n':>4}")
for (tag, N), rows in sorted(buckets.items()):
    if not rows: continue
    avg12 = sum(r[0] for r in rows) / len(rows)
    avgN  = sum(r[1] for r in rows) / len(rows)
    print(f"{tag:<10} {N:>3} {avg12:>9.0f} {avgN:>9.0f} {(avg12-avgN):>+8.0f} {len(rows):>4}")
