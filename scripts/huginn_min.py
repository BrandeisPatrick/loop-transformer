"""Minimal Huginn-0125 (recurrent-depth) inference on Apple Silicon.

Verified 2026-09-06 on M4/16GB: Python 3.12.13, torch 2.14.0 (MPS), transformers 4.57.6.
MUST pin transformers<5: v5 crashes in tie_weights with
"AttributeError: 'list' object has no attribute 'keys'".
"""
import os, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

REPO = "tomg-group-umd/huginn-0125"          # or a local bf16 copy (see convert_to_bf16 below)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# NOTE: transformers 4.57 renamed torch_dtype -> dtype (torch_dtype still works, warns).
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, trust_remote_code=True)
model = model.to(device).eval()
tok = AutoTokenizer.from_pretrained(REPO)     # PreTrainedTokenizerFast, tokenizer.json only

# num_steps is NOT allowed inside GenerationConfig -- it must be a call kwarg.
gen_cfg = GenerationConfig(
    max_new_tokens=256,
    stop_strings=["<|end_text|>", "<|end_turn|>"],
    use_cache=True,
    do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None,
    return_dict_in_generate=True,
    bos_token_id=65504, eos_token_id=65505, pad_token_id=65509,
)

# ---- chat template (the model understands it despite having no post-training) ----
messages = [
    {"role": "system", "content": "You are a helpful assistant that can assist users with mathematical reasoning."},
    {"role": "user",   "content": "Natalia sold clips to 48 friends in April, and half as many in May. How many total?"},
]
chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# -> '<|begin_text|><|begin_header|>system<|end_header|>\n\n...<|end_turn|><|begin_header|>Huginn<|end_header|>\n\n'
ids = tok.encode(chat, return_tensors="pt", add_special_tokens=False).to(device)

# ---------------- 1. fixed recurrence: THE loop knob ----------------
# r = num_steps. 4 = minimum coherent, 32 = paper default (config.mean_recurrence),
# 64 = saturation.  Materialized params ~ num_steps * 1.5B + 2B.
out = model.generate(ids, gen_cfg, tokenizer=tok, num_steps=32,
                     cache_lookup_strategy="latest-m4-compress-s16")  # cuts KV RAM ~6.6x
print(tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True))

# a single forward pass at a chosen depth (no generation):
with torch.no_grad():
    logits = model(ids, num_steps=32).logits

# ---------------- 2. per-token adaptive depth (zero-shot) ----------------
# criterion in {"kl","entropy-diff","latent-diff","argmax-stability","cosine","minp-kl"}
# exit_threshold defaults to "auto"; auto for kl == 1e-3, but the PAPER uses 5e-4 -> pass it.
out2 = model.generate_with_adaptive_compute(
    ids, gen_cfg, tokenizer=tok,
    num_steps=64,                    # upper bound on r
    criterion="kl", exit_threshold=5e-4,
    continuous_compute=False,        # True = warm-start latent state = "continuous CoT"
    cache_lookup_strategy="latest-m4-compress-s16",   # README's cache_kwargs={} is STALE
)
print(tok.decode(out2.sequences[0, ids.shape[1]:], skip_special_tokens=True))
