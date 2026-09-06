#!/usr/bin/env python
"""Build results/REPORT.md: local GSM8K results vs published numbers, plus an Ouro loop-count table."""
import glob, json, os, math
ROOT = os.path.join(os.path.dirname(__file__), "..")
PAPER = {  # published reference numbers (full test sets), see NOTES.md §4
    "ouro-1.4b|T4": ("Ouro-1.4B, T=4 (paper Tab. 16, 3-shot CoT strict)", 78.92),
    "hf.co/mradermacher/Qwen3-1.7B-Base-GGUF:Q8_0": ("Qwen3-1.7B-Base (Ouro paper baseline, 3-shot CoT strict)", 70.28),
    "hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M": ("Nanbeige4.2-3B-Base card: 92.7 (setting unpublished; instruct/thinking model here)", 92.7),
}
def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.jsonl"))):
        if "smoke" in p: continue
        rs = [json.loads(l) for l in open(p) if l.strip()]
        if not rs: continue
        n = len(rs); ok = sum(1 for r in rs if r.get("correct")); acc = ok / n
        se = math.sqrt(acc * (1 - acc) / n) if n else 0
        loops = (rs[0].get("extra") or {}).get("num_loops")
        toks = [r.get("usage", {}).get("completion_tokens") for r in rs if r.get("usage")]
        toks = [t for t in toks if isinstance(t, int)]
        rows.append(dict(file=os.path.basename(p), model=rs[0].get("model"), loops=loops, n=n, ok=ok, acc=acc, se=se,
                         trunc=sum(1 for r in rs if r.get("finish_reason") == "length"),
                         err=sum(1 for r in rs if r.get("error")),
                         lat=sum(r.get("latency_s", 0) for r in rs) / n,
                         tok=(sum(toks) / len(toks)) if toks else None,
                         done=any(os.path.exists(p.replace(".jsonl", ".summary.json")) for _ in [0])))
    return rows
def main():
    rows = load()
    out = ["# looplm results (GSM8K)\n", "Local numbers are on a fixed 200-problem subset (seed 0) unless n says otherwise; ± is one binomial standard error. Paper numbers are on the full 1319-problem test set.\n"]
    out.append("| run | model | loops | n | acc | paper | trunc | err | s/item | tok/item | status |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        key = f"{r['model']}|T{r['loops']}" if r['loops'] else r['model']
        ref = PAPER.get(key) or PAPER.get(r['model'])
        paper = f"{ref[1]:.1f}" if ref and (r['loops'] in (None, 4) or 'ouro' not in r['model']) else "—"
        out.append(f"| {r['file'].replace('.jsonl','')} | {r['model'][:40]} | {r['loops'] or ''} | {r['n']} | **{100*r['acc']:.1f}** ± {100*r['se']:.1f} | {paper} | {r['trunc']} | {r['err']} | {r['lat']:.1f} | {round(r['tok']) if r['tok'] else '—'} | {'done' if r['done'] else 'running'} |")
    ouro = sorted([r for r in rows if 'ouro' in (r['model'] or '') and r['loops']], key=lambda r: r['loops'])
    if ouro:
        out.append("\n## Ouro-1.4B: accuracy vs recurrent steps (GSM8K 3-shot strict)\n")
        out.append("| loops (T) | n | acc | s/item |\n|---|---|---|---|")
        for r in ouro: out.append(f"| {r['loops']} | {r['n']} | {100*r['acc']:.1f} ± {100*r['se']:.1f} | {r['lat']:.1f} |")
        out.append("\nThe paper only reports GSM8K at T=4 (78.92); its per-step ablation is on MMLU (41.21 / 60.43 / 66.71 / 67.45 at T=1..4).")
    open(os.path.join(ROOT, "results", "REPORT.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
if __name__ == "__main__":
    main()
