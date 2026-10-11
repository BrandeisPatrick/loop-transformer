#!/usr/bin/env python
"""Logit parity: the Ouro GGUF (llama.cpp) against the transformers reference, at every position of
20 prompts, at several loop counts.

Both sides see the SAME token ids (tokenized once with the HF tokenizer), so the comparison is of the
network alone. Reported per loop count: top-1 agreement over all positions, max and mean absolute
logit difference, and the number of positions where the two top-1 margins are within 0.1 (near-ties,
where bf16-vs-F16 rounding legitimately flips the argmax).

Usage (in .venv-tf4):
  python scripts/logit_parity.py --gguf models/Ouro-1.4B-F16.gguf --loops 1 2 4 --out results/logit_parity.json
Reference weights come from the HF cache (ByteDance/Ouro-1.4B). Only one model is resident at a time.
"""
import argparse, json, os, subprocess, sys, time, gc
import numpy as np

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. It was a bright cold day in April, and the clocks were striking thirteen.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))",
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: Let's think step by step.",
    "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 10; i++) printf(\"%d\\n\", i * i);\n    return 0;\n}",
    "{\"name\": \"Ouro\", \"layers\": 24, \"loops\": 4, \"params\": \"1.4B\", \"license\": \"apache-2.0\"}",
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
    "La inteligencia artificial es la simulación de procesos de inteligencia humana por parte de máquinas, especialmente sistemas informáticos.",
    "Die Hauptstadt von Deutschland ist Berlin. Die Stadt liegt an der Spree und hat etwa 3,7 Millionen Einwohner.",
    "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。",
    "Le Petit Prince est une œuvre de langue française, la plus connue d'Antoine de Saint-Exupéry.",
    "Theorem. For every integer n >= 1, the sum 1 + 2 + ... + n equals n(n+1)/2.\nProof. We proceed by induction on n.",
    "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2024-01-01' GROUP BY customer_id ORDER BY total DESC LIMIT 10;",
    "User: What is the capital of Australia?\nAssistant: The capital of Australia is Canberra, not Sydney as many people assume.",
    "Ingredients: 2 cups flour, 1 tsp baking soda, 1/2 tsp salt, 1 cup butter, 3/4 cup sugar, 2 eggs, 2 cups chocolate chips.",
    "In 1905, Albert Einstein published four papers that changed physics: the photoelectric effect, Brownian motion, special relativity, and mass-energy equivalence.",
    "<html><head><title>Test</title></head><body><h1>Hello</h1><p>This is a paragraph.</p></body></html>",
    "The mitochondrion is the powerhouse of the cell. It produces ATP through oxidative phosphorylation.",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987",
    "Once upon a time, in a kingdom far away, there lived a princess who could speak to birds.",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,",
]


def reference_logits(model_id, loops_list, token_lists, device, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                 torch_dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    out = {}
    for L in loops_list:
        model.config.total_ut_steps = L
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "total_ut_steps"):
            inner.total_ut_steps = L
        res = []
        t0 = time.time()
        with torch.no_grad():
            for ids in token_lists:
                x = torch.tensor([ids], device=device)
                logits = model(input_ids=x).logits[0].float().cpu().numpy()
                res.append(logits.astype(np.float16 if dtype == "bfloat16" else np.float32))
        out[L] = res
        print(f"  reference loops={L}: {len(res)} prompts in {time.time()-t0:.0f}s", flush=True)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return out


def llamacpp_logits(dump_bin, gguf, loops, token_lists, work):
    os.makedirs(work, exist_ok=True)
    paths = []
    for i, ids in enumerate(token_lists):
        p = os.path.join(work, f"p{i:02d}.txt")
        with open(p, "w") as f:
            f.write("\n".join(str(t) for t in ids) + "\n")
        paths.append(p)
    t0 = time.time()
    subprocess.run([dump_bin, gguf, work, str(loops)] + paths, check=True, stderr=subprocess.DEVNULL)
    res = []
    for p in paths:
        with open(p + ".bin", "rb") as f:
            n_pos, n_vocab = np.frombuffer(f.read(8), dtype=np.int32)
            res.append(np.frombuffer(f.read(), dtype=np.float32).reshape(n_pos, n_vocab))
    print(f"  llama.cpp loops={loops}: {len(res)} prompts in {time.time()-t0:.0f}s", flush=True)
    return res


def compare(ref, cpp):
    total = agree = near_tie = 0
    max_diff = 0.0
    sum_diff = 0.0
    n_vals = 0
    per_prompt = []
    for r, c in zip(ref, cpp):
        v = min(r.shape[1], c.shape[1])        # vocab may be padded differently
        r = r[:, :v].astype(np.float32)
        c = c[:, :v]
        ra, ca = r.argmax(1), c.argmax(1)
        match = (ra == ca)
        d = np.abs(r - c)
        # near-tie: the reference's top-1 margin over its runner-up is < 0.1
        top2 = np.partition(r, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        nt = int(((~match) & (margin < 0.1)).sum())
        total += len(ra); agree += int(match.sum()); near_tie += nt
        max_diff = max(max_diff, float(d.max())); sum_diff += float(d.sum()); n_vals += d.size
        per_prompt.append({"positions": int(len(ra)), "agree": int(match.sum()),
                           "max_abs_diff": round(float(d.max()), 4)})
    return {"positions": total, "top1_agree": agree, "near_tie_mismatches": near_tie,
            "max_abs_diff": round(max_diff, 4), "mean_abs_diff": round(sum_diff / n_vals, 5),
            "per_prompt": per_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump-bin", default="third_party/llama.cpp/build-clt/bin/logit_dump")
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ref-dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="float32 on cpu removes bf16 rounding from the reference (slower, 5.6 GB)")
    ap.add_argument("--work", default="results/parity_work")
    ap.add_argument("--out", default="results/logit_parity.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    token_lists = [tok(p).input_ids for p in PROMPTS]
    n_pos = sum(len(t) for t in token_lists)
    print(f"{len(PROMPTS)} prompts, {n_pos} positions, bos={tok.bos_token_id}", flush=True)

    print(f"reference (transformers, {args.ref_dtype} on {args.device}):", flush=True)
    ref = reference_logits(args.model_id, args.loops, token_lists, args.device, args.ref_dtype)

    print("llama.cpp (GGUF):", flush=True)
    report = {"model_id": args.model_id, "gguf": os.path.basename(args.gguf), "prompts": len(PROMPTS),
              "positions": n_pos, "reference_dtype": args.ref_dtype, "device": args.device, "loops": {}}
    for L in args.loops:
        cpp = llamacpp_logits(args.dump_bin, args.gguf, L, token_lists, args.work)
        report["loops"][str(L)] = compare(ref[L], cpp)
        r = report["loops"][str(L)]
        print(f"loops={L}: top-1 agree {r['top1_agree']}/{r['positions']}  "
              f"({r['near_tie_mismatches']} of the mismatches are near-ties)  "
              f"max|diff| {r['max_abs_diff']}  mean|diff| {r['mean_abs_diff']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
