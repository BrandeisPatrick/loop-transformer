#!/usr/bin/env python
r"""Re-score saved generations with lm-evaluation-harness's EXACT gsm8k_cot filters.

Nothing is regenerated — this reads results/*.jsonl and re-applies the upstream rules, so the cost is
zero and the comparison to published numbers becomes methodologically exact.

Upstream (lm_eval/tasks/gsm8k/gsm8k-cot.yaml, version 3.0):
  strict-match     regex  r"The answer is (\-?[0-9\.\,]+)."   then take_first
  flexible-extract regex  r"(-?[$0-9.,]{2,})|(-?[0-9]+)"      group_select=-1, then take_first
  exact_match      ignore_case: true, ignore_punctuation: false,
                   regexes_to_ignore: [",", r"\$", r"(?s).*#### ", r"\.$"]

Two deliberate differences existed in eval/run_eval.py and both are corrected here:
  1. it took the LAST "The answer is N" rather than the first (matters when a stop sequence leaks and the
     model starts answering a fresh Q:);
  2. its regex did not require the trailing character that upstream's unescaped "." demands, so an answer
     running off the end of the token budget counted as extracted.
"""
import argparse, glob, json, os, re

STRICT = re.compile(r"The answer is (\-?[0-9\.\,]+).")
# Same rule, but allowing the currency prefix the model actually emits. lm-eval's own metric config
# lists "\$" in regexes_to_ignore, so the intent is clearly that a dollar sign should not affect
# correctness -- but that list is applied to the string the regex already captured, and the strict
# pattern cannot capture "$250" in the first place. So the ignore rule never gets to run and the
# answer is scored wrong. This variant restores the declared intent.
STRICT_FIXED = re.compile(r"The answer is \$?(\-?[0-9\.\,]+)")
FLEXIBLE = re.compile(r"(-?[\$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]

def normalize(s):
    """lm-eval's exact_match with ignore_case=True and the gsm8k regexes_to_ignore."""
    if s is None:
        return None
    for pat in IGNORE:
        s = re.sub(pat, "", s)
    return s.strip().lower()

def strict_match(text):
    m = STRICT.findall(text or "")
    return m[0] if m else None            # take_first

def strict_fixed(text):
    m = STRICT_FIXED.findall(text or "")
    return m[0] if m else None

def flexible_extract(text):
    out = []
    for m in FLEXIBLE.finditer(text or ""):
        out.append(m.group(0))
    return out[-1] if out else None       # group_select=-1 over the matches, then take_first

def score(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    n = len(rows)
    strict = flex = fixed = dollar = 0
    for r in rows:
        gold = normalize(r["answer"])
        s, f, sf = strict_match(r.get("text")), flexible_extract(r.get("text")), strict_fixed(r.get("text"))
        if s is not None and normalize(s) == gold:
            strict += 1
        if f is not None and normalize(f) == gold:
            flex += 1
        if sf is not None and normalize(sf) == gold:
            fixed += 1
        if re.search(r"The answer is \$", r.get("text") or ""):
            dollar += 1
    return dict(file=os.path.basename(path), model=rows[0].get("model"),
                loops=(rows[0].get("extra") or {}).get("num_loops"), n=n,
                strict=strict / n, strict_fixed=fixed / n, flexible=flex / n,
                dollar_rate=dollar / n,
                as_run=sum(1 for r in rows if r.get("correct")) / n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(os.path.dirname(__file__), "..", "results", "gsm8k_*.jsonl"))
    a = ap.parse_args()
    rows = [r for r in (score(p) for p in sorted(glob.glob(a.glob)) if "smoke" not in p) if r]
    w = max(len(r["file"]) for r in rows) if rows else 10
    print(f"{'run':{w}} {'n':>4} {'as-run':>8} {'strict':>8} {'strict+$':>9} {'flexible':>9} {'$-fmt':>7}")
    for r in rows:
        print(f"{r['file']:{w}} {r['n']:4d} {100*r['as_run']:7.1f}% {100*r['strict']:7.1f}% "
              f"{100*r['strict_fixed']:8.1f}% {100*r['flexible']:8.1f}% {100*r['dollar_rate']:6.1f}%")
    print("\nstrict   = lm-eval gsm8k_cot strict-match, verbatim\n"
          "strict+$ = same, with the currency prefix allowed into the capture (what lm-eval's own\n"
          "           regexes_to_ignore shows it meant to do)\nflexible = lm-eval flexible-extract")
    out = os.path.join(os.path.dirname(a.glob), "rescored_lmeval.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
