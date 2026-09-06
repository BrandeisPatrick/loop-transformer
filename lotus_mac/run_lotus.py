"""Minimal LOTUS (yingfanbot/gsm-lotus-llama3b) inference on Apple Silicon.

Mirrors scripts/eval.py exactly: raw "question\n" prompt (NO chat template),
latent block appended to the prompt, greedy decode, answer after "###".

  python run_lotus.py --loops 6 --question "Natalia sold clips to 48 friends..."
"""
import os, sys, argparse, re

# MPS: let any op without a Metal kernel fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mps_shim import _pick_device, patch_cuda_calls

# scripts/lotus.py lives in the cloned LOTUS repo; put it on the path.
LOTUS_REPO = os.environ.get("LOTUS_REPO", os.path.expanduser("~/Documents/vibe/looplm/lotus"))
sys.path.insert(0, os.path.join(LOTUS_REPO, "scripts"))

MODEL_ID = "yingfanbot/gsm-lotus-llama3b"


def extract_answer(text):
    """Verbatim from scripts/eval.py."""
    parts = text.split("###")
    if len(parts) > 1:
        ans = parts[-1].replace(",", "").strip()
        if ans:
            return ans
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return text.split("#")[-1].replace(",", "").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", default="Natalia sold clips to 48 of her friends in April, "
                                         "and then she sold half as many clips in May. "
                                         "How many clips did Natalia sell altogether in April and May?")
    p.add_argument("--loops", type=int, default=6, help="R = n_looped_iters. Trained at 6; paper Table 6 sweeps 1-7.")
    p.add_argument("--c_thought", type=int, default=25, help="latent tokens per block (25 for the Llama checkpoints)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = p.parse_args()

    device = _pick_device(args.device)
    patch_cuda_calls(device)          # must happen before Lotus.generate runs
    from lotus import Lotus           # noqa: E402  (after sys.path + shim)

    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}")

    # --- tokenizer -------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token     # eval.py does exactly this
    # The 3 latent tokens are ALREADY in this repo's tokenizer_config.json at
    # 128256/128257/128258. add_tokens() is a no-op here but keeps parity with eval.py.
    tok.add_tokens("<|start-latent|>")
    tok.add_tokens("<|end-latent|>")
    tok.add_tokens("<|latent|>")
    start_id = tok.convert_tokens_to_ids("<|start-latent|>")
    end_id = tok.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    assert (start_id, end_id, latent_id) == (128256, 128257, 128258), (start_id, end_id, latent_id)

    # --- weights ---------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if base.get_input_embeddings().weight.shape[0] < len(tok):
        base.resize_token_embeddings(len(tok))

    model = Lotus(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tok.eos_token_id,   # 128009 <|eot_id|>
        pad_token_id=tok.pad_token_id,
        c_thought=args.c_thought,
    ).to(device).to(dtype)
    model.eval()

    # --- prompt: raw text, NOT the chat template -------------------------
    q_tokens = tok.encode(args.question + "\n", add_special_tokens=True)
    n_latent = args.loops * args.c_thought          # 6 * 25 = 150
    latent_block = [start_id] + [latent_id] * n_latent + [end_id]
    input_ids = torch.tensor([q_tokens + latent_block], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(                       # Lotus.generate, not HF generate
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            n_looped_iters=args.loops,              # <-- the recurrence count, per call
        )

    gen = out[0][input_ids.shape[1]:]               # returns CPU tensor of full sequence
    text = tok.decode(gen, skip_special_tokens=True)
    print(f"\nraw: {text!r}\nanswer: {extract_answer(text)}")
    print("timing (s):", model._last_timing)        # query_prefill / thought / decode


if __name__ == "__main__":
    main()
