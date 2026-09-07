#!/usr/bin/env python
"""Build results/REPORT.md: local GSM8K results vs published numbers, plus an Ouro loop-count table."""
import glob, json, os, math, re
ROOT = os.path.join(os.path.dirname(__file__), "..")
PAPER = {  # published reference numbers (full test sets), see NOTES.md §4
    "ouro-1.4b|T4": ("Ouro-1.4B, T=4 (paper Tab. 16, 3-shot CoT strict)", 78.92),
    "hf.co/mradermacher/Qwen3-1.7B-Base-GGUF:Q8_0": ("Qwen3-1.7B-Base (Ouro paper baseline, 3-shot CoT strict)", 70.28),
    "hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M": ("Nanbeige4.2-3B-Base card: 92.7 (setting unpublished; instruct/thinking model here)", 92.7),
    "hf.co/mradermacher/Qwen3-4B-Base-GGUF:Q4_K_M": ("Qwen3-4B-Base (Ouro paper Table 7, 3-shot CoT strict)", 72.86),
    "nanbeige4.2:loops1": ("no published number; forced below trained depth", None),
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
        nt = [r for r in rs if r.get("finish_reason") != "length"]
        acc_ct = (sum(1 for r in nt if r.get("correct")) / len(nt)) if nt else 0.0
        rows.append(dict(file=os.path.basename(p), model=rs[0].get("model"), loops=loops, n=n, ok=ok, acc=acc, se=se,
                         n_ct=len(nt), acc_ct=acc_ct,
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
        paper = f"{ref[1]:.1f}" if ref and ref[1] is not None and (r['loops'] in (None, 4) or 'ouro' not in r['model']) else "—"
        out.append(f"| {r['file'].replace('.jsonl','')} | {r['model'][:40]} | {r['loops'] or ''} | {r['n']} | **{100*r['acc']:.1f}** ± {100*r['se']:.1f} | {paper} | {r['trunc']} | {r['err']} | {r['lat']:.1f} | {round(r['tok']) if r['tok'] else '—'} | {'done' if r['done'] else 'running'} |")
    resc = os.path.join(ROOT, "results", "rescored_lmeval.json")
    if os.path.exists(resc):
        rs = {r["file"]: r for r in json.load(open(resc))}
        out.append("\n## Same generations, four extraction rules\n")
        out.append("See NOTES.md \u00a78: lm-eval's verbatim strict-match cannot capture a `$`-prefixed answer, "
                   "which penalises the two models unequally.\n")
        out.append("| run | as-run | lm-eval strict | strict with `$` | lm-eval flexible | `$`-formatted |\n|---|---|---|---|---|---|")
        for f, r in rs.items():
            out.append(f"| {f.replace('.jsonl','')} | {100*r['as_run']:.1f} | {100*r['strict']:.1f} | "
                       f"{100*r['strict_fixed']:.1f} | {100*r['flexible']:.1f} | {100*r['dollar_rate']:.1f}% |")
    ouro = sorted([r for r in rows if 'ouro' in (r['model'] or '') and r['loops']], key=lambda r: r['loops'])
    if ouro:
        out.append("\n## Ouro-1.4B: accuracy vs recurrent steps (GSM8K 3-shot strict)\n")
        out.append("| loops (T) | n | acc | truncated | acc on non-truncated | s/item |\n|---|---|---|---|---|---|")
        for r in ouro: out.append(f"| {r['loops']} | {r['n']} | {100*r['acc']:.1f} ± {100*r['se']:.1f} | {r['trunc']} | {100*r['acc_ct']:.1f} (n={r['n_ct']}) | {r['lat']:.1f} |")
        out.append("\nTruncation matters at low depth: a run that hits the 256-token cap never emits "
                   "\"The answer is N\" and is scored wrong. The last column removes those, separating "
                   "\"reasoned badly\" from \"never finished\".")
        out.append("\nThe paper only reports GSM8K at T=4 (78.92); its per-step ablation is on MMLU (41.21 / 60.43 / 66.71 / 67.45 at T=1..4).")
    # MMLU depth sweep from lm-eval outputs (results/mmlu/ouro-1.4b_T*/.../results_*.json)
    mm = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "mmlu", "ouro-1.4b_T*", "**", "results_*.json"), recursive=True)):
        j = json.load(open(f)); T = int(re.search(r"_T(\d)", f).group(1)); r = j["results"]["mmlu"]
        groups = {g.split("_", 1)[1]: 100 * j["results"][g]["acc,none"] for g in
                  ("mmlu_humanities", "mmlu_other", "mmlu_social_sciences", "mmlu_stem") if g in j["results"]}
        mm.append((T, 100 * r["acc,none"], 100 * r["acc_stderr,none"], groups))
    if mm:
        paper = {1: 41.21, 2: 60.43, 3: 66.71, 4: 67.45, 5: 66.64, 6: 65.77, 7: 65.28, 8: 64.49}
        out.append("\n## Ouro-1.4B: MMLU 5-shot vs recurrent steps (lm-eval, the paper's published ablation)\n")
        out.append("~3% of each subject (all 57 subjects, ~420 questions), log-likelihood scoring, no chat template. "
                   "Paper numbers are Table 10 on the full set; ± is lm-eval's reported standard error.\n")
        out.append("| loops (T) | MMLU (ours) | paper | humanities | other | social sci | STEM |\n|---|---|---|---|---|---|---|")
        for T, a, se, g in sorted(mm):
            out.append(f"| {T} | **{a:.1f}** ± {se:.1f} | {paper.get(T, '—')} | {g.get('humanities', 0):.1f} | {g.get('other', 0):.1f} | {g.get('social_sciences', 0):.1f} | {g.get('stem', 0):.1f} |")
    open(os.path.join(ROOT, "results", "REPORT.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
if __name__ == "__main__":
    main()
