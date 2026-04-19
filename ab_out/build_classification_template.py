"""
Scan ab_out/ and ab_out_k27/ for all resumed videos and emit a CSV template
you can fill with F (fully survived) / P (partial) / D (dead) per row.

Re-running is safe: existing classifications are preserved by row id.
"""
import csv, glob, json, os

ROOTS  = ["/home/soxehli/repos/cse585/ab_out",
          "/home/soxehli/repos/cse585/ab_out_k27"]
BASES  = ["dog", "car", "dancer"]
AXES   = ["subject", "setting", "color", "motion", "texture", "style", "time_of_day"]
KS     = [2, 5, 7, 10]
WK_TPL = "/home/soxehli/repos/cse585/eval/teacache/vbench/workload_ab_axes_{b}.json"
OUT    = "/home/soxehli/repos/cse585/ab_out/classification_template.csv"
FIELDS = ["id", "classification", "base", "axis", "k", "pair"]

# prompt → axis lookup from workload files
prompt_to_axis = {}
for b in BASES:
    p = WK_TPL.format(b=b)
    if os.path.exists(p):
        for e in json.load(open(p)):
            prompt_to_axis[e["prompt_en"]] = e["axis"]

# preserve prior classifications if the CSV already exists
prior = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        if r.get("classification", "").strip():
            prior[r["id"]] = r["classification"].strip()

rows = []
for root in ROOTS:
    for b in BASES:
        for pair_dir in sorted(glob.glob(os.path.join(root, b, "pair_*"))):
            mpath = os.path.join(pair_dir, "metrics.csv")
            if not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                reader = list(csv.DictReader(f))
            if not reader:
                continue
            prompt_a = reader[0]["prompt_a"]
            prompt_b = reader[0]["prompt_b"]
            axis = prompt_to_axis.get(prompt_b, "?")
            pair_name = os.path.basename(pair_dir)
            pair_num = pair_name.split("_", 2)[1] if "_" in pair_name else pair_name
            for r in reader:
                k = int(r["variant"].replace("B_from_step", ""))
                rid = f"{b}_{axis}_k{k}"
                rows.append({
                    "id": rid,
                    "classification": prior.get(rid, ""),
                    "base": b,
                    "axis": axis,
                    "k": k,
                    "pair": pair_num,
                })

# Fill in placeholder rows for any (base, axis, k) combos that haven't
# been generated yet, so the CSV shows the full target matrix.
seen_ids = {r["id"] for r in rows}
for b in BASES:
    for axis in AXES:
        for k in KS:
            rid = f"{b}_{axis}_k{k}"
            if rid in seen_ids:
                continue
            rows.append({
                "id": rid,
                "classification": prior.get(rid, ""),
                "base": b,
                "axis": axis,
                "k": k,
                "pair": f"{AXES.index(axis):02d}",
            })

rows.sort(key=lambda r: (r["base"], r["pair"], r["k"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

filled = sum(1 for r in rows if r["classification"])
print(f"{OUT}")
print(f"  {len(rows)} rows total ({filled} already classified)")
print(f"  k values present: {sorted(set(r['k'] for r in rows))}")
