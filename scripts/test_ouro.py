#!/usr/bin/env python
"""Load Ouro on MPS (transformers 4.x venv) and generate at several recurrent-step counts.
Usage: .venv-tf4/bin/python scripts/test_ouro.py [ByteDance/Ouro-1.4B] [steps e.g. 1,2,3,4]"""
import sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
repo = sys.argv[1] if len(sys.argv) > 1 else "ByteDance/Ouro-1.4B"
steps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4").split(",")]
attn = sys.argv[3] if len(sys.argv) > 3 else None
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
print("config:", {k: getattr(cfg, k, None) for k in ["num_hidden_layers", "total_ut_steps", "early_exit_threshold", "early_exit_step", "torch_dtype"]})
kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
if attn: kw["attn_implementation"] = attn
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(repo, **kw).to(dev).eval()
print(f"loaded on {dev} in {time.time()-t0:.1f}s; attn={getattr(model.config, '_attn_implementation', None)}; params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
prompts = ["Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's think step by step.",
           "The capital of France is"]
for T in steps:
    model.config.total_ut_steps = T
    if hasattr(model, "model") and hasattr(model.model, "total_ut_steps"): model.model.total_ut_steps = T
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(dev)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=60, do_sample=False)
        n = out.shape[1] - ids["input_ids"].shape[1]; dt = time.time() - t0
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n[T={T}] {n} tok in {dt:.1f}s ({n/dt:.1f} tok/s) | {p[:40]!r}\n  -> {text!r}")
