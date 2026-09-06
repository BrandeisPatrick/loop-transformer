#!/usr/bin/env python
"""Print a table of all results/*.summary.json"""
import glob, json, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "*.summary.json"))):
    s = json.load(open(p))
    rows.append((s["benchmark"], s["model"], json.dumps(s.get("extra", {})), s["n"], s["accuracy"],
                 s.get("truncated"), s.get("mean_completion_tokens"), s.get("mean_latency_s")))
print(f"{'bench':8} {'model':28} {'extra':22} {'n':>5} {'acc':>6} {'trunc':>5} {'tok':>7} {'s/it':>6}")
for r in rows:
    print(f"{r[0]:8} {r[1][:28]:28} {r[2][:22]:22} {r[3]:5d} {r[4]:6.3f} {str(r[5]):>5} {str(round(r[6])) if r[6] else '-':>7} {r[7]:6.1f}")
