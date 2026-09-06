# RECIPES — how to actually run each looped model locally

Per-model inference recipes produced by the research workflow and cross-checked against this Mac. Each was written after fetching the model card, the custom `modeling_*.py`, and the upstream issue tracker. Sources are in the workflow transcripts; the hands-on confirmations are in NOTES.md §5.

## Verified availability (all ungated)

| repo | params | weights | loop control | notes |
|---|---|---|---|---|
| harims95/LoopLM-135M-naive | 0.134B | 0.742 GB float32 in model.safetensors (config torch_dtype float32; trained bf16). The -sft sibling is bf16 (536 MB). | Via the HF AutoModel wrapper: FIXED at config field mu_rec (default 6); LoopLMForCausalLM.forward drops **kwar | apache-2.0 |
| ml-ryanlee/looped-16x2-32L-d640-1e18-a100 | 0.1436B | 0.583 GB float32 (safetensors F32; trained with bf16 AMP per train_config.json) | Fixed via config field `num_stacks` (=2 here; `num_layers_in_stack`=16). LoopedTransformer.forward does `for i | apache-2.0 |
| Girinath11/recursive-language-model-198m | 0.198B | 0.794 GB float32 (config dtype: float32; HF tensor types I64, F32; safetensors 794 MB) | Not configurable via kwarg. forward(input_ids, labels=None, attention_mask=None, **kwargs) has no step argumen | MIT (HF API license: |
| armenjeddi/LoopFormer-3block-8iterations-FineWeb300K | 0.28B | 0.556 GB bfloat16 (config 'dtype': 'bfloat16'; model.safetensors 556 MB) | Set via the `steps` kwarg to forward()/generate(): a list of step sizes summing to 1, default [1/8]*8 = 8 loop | Not specified on the |
| mshapiro123/recurrent-qwen2.5-0.5b-full-block | 0.5B | 0.364 GB delta tensors: 144 bfloat16 + 6 float32 (conversion_receipt.json); runtime dtype follows whatever the Qwen backbone is loaded in | Set per call, not in config: forward(max_loops: int = 1, loop_selection: int|LongTensor|None = None, return_lo | apache-2.0 |
| summerMC/Trm-text-1B | 1.032B | 4.13 GB float32 (config dtype float32; safetensors F32 1,032,390,656 params) | Set via config field only: model.config.recurrence_steps (read each forward pass, `for _ in range(self.config. | MIT (cardData licens |
| sapientinc/HRM-Text-1B | 1.18B | 2.37 GB bfloat16 (single model.safetensors, BF16 1,182,795,264 params) | Config fields only: HrmTextConfig.H_cycles (=2) and L_cycles (=3) in config.json; there is no generate() kwarg | apache-2.0 |
| SandyResearch/parcae-1.3b | 1.3B | 5.36 GB Not declared in config; single pytorch_model.bin of 5.36GB, consistent with fp32 (~1.34B params). from_pretrained accepts dtype= to cast. | Yes. In eval mode with no override the model runs exactly config.mean_recurrence steps (config.json: 8). Overr | HF repo: no license  |
| smcleish/Recurrent-Llama-3.2-train-recurrence-16 | 1.385B | 5.54 GB float32 (config torch_dtype; safetensors F32, 5.54 GB) | Yes: pass num_steps=N to model(input_ids, num_steps=...) or model.generate(input_ids, gen_config, num_steps=N) | Apache-2.0 (repo tag |
| irafm-llm/Recurrent-Llama-3.2-1B | 1.385B | 2.77 GB bfloat16 (config torch_dtype; safetensors metadata BF16 1,385,228,288 params) | generate kwarg `num_steps` (int) -- README: model.generate(ids, max_new_tokens=40, num_steps=32, tokenizer=tok | llama3.2 (Llama 3.2  |
| smcleish/Recurrent-Llama-3.2-train-recurrence-32 | 1.385B | 5.54 GB float32 (safetensors F32; README loads torch.float32; load as bfloat16 to halve RAM; HuginnStaticCache hardcodes bf16) | Yes. Forward kwarg num_steps (int, or a (no_grad_steps, grad_steps) pair): model(input_ids, num_steps=32) per  | apache-2.0 (README + |
| ByteDance/Ouro-1.4B-Thinking | 1.4347B | 2.87 GB bfloat16 | Config field only: config.total_ut_steps (default 4); modeling_ouro.py reads getattr(config,'total_ut_steps',4 | apache-2.0 |
| ByteDance/Ouro-1.4B | 1.435B | 2.87 GB bfloat16 (config torch_dtype; safetensors all BF16) | Loop count = config field `total_ut_steps` (default/max 4 as trained; read via getattr(self.config,'total_ut_s | Apache-2.0 (HF API l |
| Thrillcrazyer/Qwen3_1.7B_LoopUS | 2.032B | 4.07 GB bf16 (safetensors BF16 for 2,031,739,904 params; F32 for 534,529 gate/confidence-head params); repo code runs fp32 on non-CUDA by default | Set via config/instance attribute, not a generate() kwarg: LDSConfig.N (this checkpoint N=20; class default 6) | Apache-2.0 (HF card  |
| irafm-llm/Recurrent-Gemma-2-2b | 2.28B | 4.56 GB bfloat16 (single model.safetensors, config torch_dtype bfloat16) | Yes: `num_steps` kwarg passed to model.generate(...) / model.forward(...) (README example: model.generate(ids, | gemma (Gemma Terms o |
| ByteDance/Ouro-2.6B | 2.668B | 5.34 GB bfloat16 (safetensors BF16; mlx-community 4-bit variant ~1.5 GB) | Config field only: `total_ut_steps` (default 4) and `early_exit_threshold` (default 1.0 = always run all steps | Apache-2.0 |
| ByteDance/Ouro-2.6B-Thinking | 2.67B | 5.34 GB bfloat16 (config torch_dtype=bfloat16; single 5.34 GB safetensors) | Config field `total_ut_steps` (default 4; set via AutoConfig then pass config= to from_pretrained, per model c | Apache-2.0 |
| yingfanbot/gsm-lotus-llama3b | 3.21B | 6.43 GB bfloat16 | Per-call argument, not a config field: Lotus.generate(..., n_looped_iters=N) / Lotus.forward(n_looped_iters);  | MIT |
| tomg-group-umd/huginn-0125 | 3.565B | 15.65 GB Stored float32 (config torch_dtype=float32, 15.6 GB, duplicated tied embedding); bf16 recommended and used at load (~7.1 GB); HuginnStaticCache hardcodes bfloat16. | Recurrence is a per-call kwarg: model(input_ids, num_steps=N) and model.generate(input_ids, generation_config, | apache-2.0 |
| Nanbeige/Nanbeige4.2-3B-Base | 4.17B | 8.34 GB bfloat16 (single model.safetensors, BF16 4,169,800,704 params) | Config only, not a generate kwarg: config.json num_loops=2 (with loop_loss_weights=[] and skip_loop_final_norm | Apache-2.0 |
| Nanbeige/Nanbeige4.2-3B | 4.17B | 8.34 GB bfloat16 (config torch_dtype; HF safetensors metadata BF16 4,169,800,704 params) | Fixed by config, not a generate kwarg. modeling_nanbeige.py: num_loops = self._get_num_loops(), which returns  | Apache-2.0 (README f |

## Recipes

### Ouro-1.4B (ByteDance Seed, looped/universal-transformer LM: 24 shared layers applied total_ut_steps=4 times, learned early-exit gate; 1.43 B params, bf16, Apache-2.0)

**Repo:** ByteDance/Ouro-1.4B (reasoning-SFT sibling: ByteDance/Ouro-1.4B-Thinking; transformers-5.x community fork: KristianS7/Ouro-1.4B)  
**Download:** 3 GB · **RAM:** ~5 GB

**Loop control:** Config field `total_ut_steps` (config.json: 4; configuration_ouro.py default 4). At init OuroModel does `self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)` and forward runs `for current_ut in range(self.total_ut_steps): for decoder_layer in self.layers[:num_hidden_layers]: ...`, appending a normed hidden state and `early_exit_gate(hidden)` logit per loop. So: set `cfg.total_ut_steps` before from_pretrained (also sizes the cache: max_cache_size = num_hidden_layers * total_ut_steps = 96 layer slots), or per call `model.model.total_ut_steps = n` (1..4; 5-8 is untrained extrapolation, paper Table 10 shows degradation). It is NOT a generate() kwarg. Post-hoc readout controls on OuroForCausalLM.forward: `exit_at_step` (0-based int, picks hidden_states_list[k]), `exit_threshold` (float; per token, exits at first loop where cumulative sigmoid-gate prob >= threshold, else last), `use_weighted_exit=True` (gate-weighted expected logits); defaults come from config `early_exit_step` (None) / `early_exit_threshold` (1.0). In the HF code these do not skip compute - all loops run, then the readout is gathered (KristianS7 fork card: "Proper early exit implementation is pending; currently only post-hoc gating is supported"). vLLM always runs all 4 steps.

**Thinking mode:** Ouro-1.4B (base): no thinking mode. Its tokenizer_config.json ships a plain ChatML template (system/user/assistant with <|im_start|>/<|im_end|>) but the model is a base LM; discussion #8 shows applying the chat template during evals drops GSM8K from ~79 to ~61, so evaluate with raw text. Ouro-1.4B-Thinking (same 2.87 GB size, separate repo): SFT reasoning model; its tokenizer chat template ends with `{%- if enable_thinking is defined and enable_thinking is true -%}<think>\n{%- endif -%}` after the assistant header, so pass `enable_thinking=True` to apply_chat_template to force a <think> block; <think>/</think> are special tokens in both tokenizers. Its README recommends temperature=1.0, top_p=0.7 (paper Table 17 uses the same). Thinking repo tokenizer has bos=<|im_start|>, eos=pad=<|im_end|>; the base repo still has bos=eos=<|endoftext|> (id 0) with PR #11 (bos->1, eos->2) unmerged, hence the explicit eos_token_id=[0, 2] in the snippet.

**Dependencies:**

- torch==2.14.0  (PyPI wheel torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl, 127 MB, requires_python>=3.10, needs macOS>=14; torch 2.9.1 cp312 macosx_11 arm64 74 MB also fine)
- transformers==4.57.6  (last 4.x release, 2026-01-16; MUST be >=4.57.1 and <5 for the official repo: HF discussion #14 shows 4.54.1/4.55.0 crash with "property 'key_cache' has no setter" and <=4.52.4 fails on missing layer_type_validation; 4.57.6 is what KristianS7 used to reproduce the paper in discussion #8)
- accelerate==1.14.0  (optional; only if you want device_map=; the snippet uses .to(device) and does not need it)
- safetensors, tokenizers, huggingface_hub  (pulled in by transformers; no sentencepiece needed - tokenizer is GPT2Tokenizer with vocab.json/merges.txt/tokenizer.json)
- lm-eval==0.4.13  (for replicating Table 7; authors used lm-eval-harness, KristianS7 reproduced with lm-eval 0.4.11)
- evalplus==0.3.1  (for HumanEval/HumanEval+/MBPP/MBPP+ pass@1, as in the paper)
- fastapi + uvicorn  (optional, only for an OpenAI-compatible shim next to Ollama)
- MLX alternative instead of torch: mlx-lm from git+https://github.com/kernelpool/mlx-lm@feature/ouro  (unmerged PR #599; stock mlx-lm 0.31.3 has no ouro model type)

**Quantized / alternative runtimes:** GGUF: none found (no Ouro architecture in llama.cpp; ollama/ollama#14252 open, unanswered). MLX: mlx-community/Ouro-1.4B-4bit exists (807 MB model.safetensors, converted with mlx-lm 0.28.4 on 2025-11-09, 4-bit, config keeps total_ut_steps=4; also mlx-community/Ouro-1.4B-Thinking-4bit, Ouro-2.6B-4bit, Ouro-2.6B-Thinking-4bit). Stock mlx-lm (PyPI 0.31.3) has no mlx_lm/models/ouro.py - support lives in the still-open mlx-lm PR #599 (opened 2025-11-09, branch refreshed 2026-08-21 and tested on all four Ouro repos; maintainer awni is lukewarm because data-dependent early exit stalls the GPU). Run: `uv venv -p 3.12 ~/ouro-mlx && source ~/ouro-mlx/bin/activate && uv pip install "git+https://github.com/kernelpool/mlx-lm@feature/ouro"` then `mlx_lm.generate --model mlx-community/Ouro-1.4B-4bit -p "who is albert einstein?" -m 4096` (command from the PR). Python: `from mlx_lm import load, generate; model, tok = load("mlx-community/Ouro-1.4B-4bit")`; the PR's Model.__call__ accepts exit_threshold=0.7 / exit_at_step=2 / use_weighted_exit=True, but total_ut_steps is fixed at init (ModelArgs, default 4 - edit config.json to change loops). `mlx_lm.server --model mlx-community/Ouro-1.4B-4bit` should give an OpenAI-compatible endpoint next to Ollama (not verified on this branch). Caveat: 4-bit quantization (early_exit_gate kept at 8-bit) means MLX numbers will not exactly replicate the bf16 paper results; use it for the Ollama pairing / speed, and the transformers bf16 path for replication. PR: https://github.com/ml-explore/mlx-lm/pull/599

**Minimal load + generate:**

```python
# ouro_mps.py -- ByteDance/Ouro-1.4B on Apple Silicon (MPS, CPU fallback)
# Setup (once):
#   uv venv -p 3.12 ~/ouro-venv && source ~/ouro-venv/bin/activate
#   uv pip install "torch==2.14.0" "transformers==4.57.6"
# Run:  python ouro_mps.py
import os, sys, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any op MPS lacks (not confirmed to be needed)
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

REPO = "ByteDance/Ouro-1.4B"            # base LM.  Reasoning model: "ByteDance/Ouro-1.4B-Thinking"
# REVISION = "<full sha of 'Fix EOS/BOS problem.' (Jan 18 2026) from /commits/main>"  # pin trust_remote_code for reproducibility

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32   # bf16 on MPS needs macOS>=14; fp32 on CPU (~5.7 GB weights)

cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
cfg.total_ut_steps = 4          # recurrence count the model was trained with (max). Sizes the KV cache = 24 layers * this.
cfg.early_exit_threshold = 1.0  # 1.0 = always take the last loop's hidden state; <1.0 = learned-gate per-token selection
cfg.pad_token_id = 0            # repo config ships with no pad_token_id (HF discussion #12)

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token                     # <|endoftext|>, id 0
IM_END = tok.convert_tokens_to_ids("<|im_end|>")      # id 2 -- ChatML stop token; config eos_token_id is 0 (PR #11 unmerged)

model = AutoModelForCausalLM.from_pretrained(
    REPO, config=cfg, trust_remote_code=True,
    torch_dtype=dtype,                # (`dtype=` on newer transformers; `torch_dtype` still accepted on 4.57)
    attn_implementation="sdpa",       # "sdpa" or "eager". flash_attention_2 is CUDA-only.
).to(device).eval()

# --- Batched-generation fix (HF PR #10, unmerged on Ouro-1.4B): without this, batch_size>1 gives garbage (eager)
# --- or a RuntimeError (sdpa). Harmless for batch_size 1.
_mod = sys.modules[type(model).__module__]
def _get_mask_sizes(self, cache_position, layer_idx=0):
    return self.get_seq_length(layer_idx) + cache_position.shape[0], 0
_mod.UniversalTransformerCache.get_mask_sizes = _get_mask_sizes

def set_loops(n: int):
    """Change the number of recurrent passes executed per forward, per call, without reloading (1..cfg.total_ut_steps).
    OuroModel.forward loops `for current_ut in range(self.total_ut_steps)`; the attribute is read at call time.
    Never reuse a KV cache across different n (each generate() call builds a fresh UniversalTransformerCache)."""
    assert 1 <= n <= cfg.total_ut_steps
    model.model.total_ut_steps = n

def complete(prompt: str, max_new_tokens=128, loops=4, **gen):
    """Raw LM completion -- this (no chat template) is how the base model was evaluated in the paper."""
    set_loops(loops)
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, **gen)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

def chat(messages, max_new_tokens=512, loops=4, enable_thinking=False, **gen):
    """ChatML chat. `enable_thinking` is only honored by Ouro-*-Thinking's template (adds '<think>\\n' after
    '<|im_start|>assistant\\n'); the base template silently ignores it."""
    set_loops(loops)
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=enable_thinking).to(device)
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             eos_token_id=[tok.eos_token_id, IM_END], pad_token_id=tok.pad_token_id, **gen)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    print("device:", device, "| dtype:", dtype)
    print(complete("Question: What is 17 * 23?\\nAnswer:", loops=4))
    print(complete("Question: What is 17 * 23?\\nAnswer:", loops=1))     # 1 pass = ~1/4 compute, degraded quality
    print(chat([{"role": "user", "content": "Solve 17 * 23 step by step."}], do_sample=False))
    # Thinking model (REPO = "ByteDance/Ouro-1.4B-Thinking"): authors' sampling is temperature=1.0, top_p=0.7
    # print(chat([...], enable_thinking=True, do_sample=True, temperature=1.0, top_p=0.7, max_new_tokens=4096))
    #
    # Post-hoc exit controls (all loops are STILL computed; only the readout changes) -- forward kwargs, so they
    # should pass through generate(): model.generate(ids, max_new_tokens=64, exit_at_step=1)   # 0-based step
    #   or exit_threshold=0.7 (cumulative learned-gate prob), or use_weighted_exit=True (gate-weighted logits).
    # To actually SAVE compute use set_loops(n).
```

**Gotchas:**

- DISK: model.safetensors is 2.87 GB (2,869,336,434 bytes) + ~6 MB tokenizer/code, plus torch wheel 127 MB (2.14.0; installs to roughly 0.4-0.5 GB) + transformers/deps ~0.2 GB. Total ~3.6 GB vs 3.9 GB free measured today (df: 3.9Gi avail) - free at least 1-2 GB more first, or use the 807 MB MLX 4-bit port. HF downloads go to ~/.cache/huggingface/hub on the same volume.
- RAM: bf16 weights 2.87 GB in unified memory. The UniversalTransformerCache keeps a separate K/V slot per (layer, loop) = 24*4 = 96 slots, 16 heads*128 dim, i.e. ~768 KB per token in bf16 - 4x a normal 24-layer model: ~1.5 GB at 2k context, ~3 GB at 4k. Expect ~4 GB for short chats, 6-7 GB at 4k context; fine on 16 GB. CPU fp32 fallback doubles weight memory (~5.7 GB).
- TRANSFORMERS VERSION: README says transformers<4.56 / recommends 4.54.1 - that is BACKWARDS after the Nov-16-2025 'Update Ouro Cache' commit (HF discussion #14, empirically tested): 4.54.1/4.55.0 crash with "AttributeError: property 'key_cache' ... has no setter", <=4.52.4 fails importing layer_type_validation, 4.57.1+ works. Pin transformers==4.57.6 (<5). The official 1.4B repo does not load on transformers 5.x (draft PR #13 is empty, +0 -0); for 5.x use KristianS7/Ouro-1.4B (tested 5.9.0, adds rope_parameters handling, get_mask_sizes fix, bos=1/eos=2/pad=0) - PyPI latest is 5.16.1, untested with that fork.
- APPLE SILICON IS UNVERIFIED: no report anywhere (HF discussions, GitHub, web search) of Ouro running on MPS or CPU via transformers. Evidence it should: modeling_ouro.py has no .cuda() / CUDA-only paths, no hard flash-attn import (_supports_sdpa=True, eager fallback), and its rotary code explicitly maps device_type 'mps' -> 'cpu' for the disabled-autocast block, so someone anticipated MPS. PYTORCH_ENABLE_MPS_FALLBACK=1 is a precaution, not a confirmed requirement. Do the first run with `--limit`/short prompts and compare logits to CPU fp32 if outputs look off.
- DTYPE: use bfloat16 on MPS (torch 2.12+ wheels require macOS 14 anyway; this machine is Darwin 25). Avoid fp16 (overflow risk in the 4x looped residual stream). Eager attention upcasts softmax to fp32; sdpa is fine on MPS for batch 1.
- BATCHED GENERATION BUG (open, discussions #9/#10): batch_size>1 corrupts all but the longest sequence with eager (whitespace garbage) and raises with sdpa, because UniversalTransformerCache.get_mask_sizes returned query_length instead of cached+query. The fix (PR #10, 'ready to merge' but still unmerged on 1.4B; merged on 2.6B-Thinking Feb 26 2026) is the 3-line monkeypatch in the snippet. Without it, run lm-eval with --batch_size 1.
- TOKEN IDS: config.json has bos_token_id=eos_token_id=0 (<|endoftext|>) and NO pad_token_id (-> AttributeError 'OuroConfig' object has no attribute 'pad_token_id' in discussion #12, unresolved) and no generation_config.json. Set tok.pad_token, cfg.pad_token_id=0, and stop on both 0 and <|im_end|> (id 2) for chat. PR #11 proposing bos=1/eos=2 is unmerged on 1.4B.
- CHAT TEMPLATE vs EVAL: the base model ships a ChatML template but was evaluated WITHOUT it; using apply_chat_template in lm-eval dropped GSM8K 78.9->60.8 and BBH 71.0->60.8 (discussion #8). Use raw few-shot text for base evals.
- LOOP COUNT SEMANTICS: exit_at_step / exit_threshold / use_weighted_exit change only the readout; all 4 loops still execute. To reduce compute set model.model.total_ut_steps (per call). Passing exit_at_step etc. through generate() relies on transformers forwarding unknown kwargs whose names appear in forward's signature - expected to work, not verified here.
- SPEED: every token costs 4 full passes over 24 layers, so expect roughly 1/4 the tok/s of a comparable 1.4B model. No MPS numbers exist; the only Apple Silicon datapoint is the MLX PR: 71 tok/s generation for Ouro-1.4B-4bit on an M3 Ultra (80 GPU cores) - an M4 MacBook will be far lower. Use max_new_tokens explicitly; MMLU 5-shot logprob (14k questions) will take hours on MPS.
- trust_remote_code caches modeling_ouro.py under ~/.cache/huggingface/modules; pin `revision=` to the current main sha for reproducibility (last 1.4B commit: 'Fix EOS/BOS problem.', Jan 18 2026; the Feb-26-2026 rope/pad/batched-cache fixes exist only on Ouro-2.6B-Thinking).
- OLLAMA PAIRING: no GGUF and no llama.cpp/Ollama architecture for Ouro (ollama/ollama#14252 open since 2026-02-14, no maintainer reply). Serve Ouro from this venv behind a tiny FastAPI /v1/chat/completions shim (or `mlx_lm.server` on the MLX path, which is OpenAI-compatible) and run Qwen3-1.7B/4B baselines in Ollama; point the eval client at both endpoints.
- COULD NOT CONFIRM: (a) any MPS/CPU run of Ouro; (b) torch cp313/cp314 macOS wheels; (c) the exact lm-eval task YAMLs/versions the authors used (Table 16 gives shots/metrics only; MATH500 was an in-house harness); (d) that stock mlx-lm 0.31.3 still rejects model_type ouro (inferred from PR #599 still open on 2026-08-24; last direct user report is on mlx-community/Ouro-2.6B-Thinking-4bit discussion #1); (e) which transformers version produced discussion #12's pad_token_id error; (f) PyTorch docs page on MPS bf16 (page fetched but had no dtype statement).

**Official eval recipe:** Paper (arXiv 2510.25741, Sec. 5 + Appendix C.1 Table 16, quoted verbatim from the HTML): "All benchmarks are evaluated using lm-eval-harness and evalplus frameworks". Base models, all at T=4 recurrent steps (R4): MMLU - logprobs, 5-shot, lm-eval-harness; MMLU-Pro - strict match, 5-shot CoT, lm-eval-harness; BBH - strict match, 3-shot CoT, lm-eval-harness; ARC-C - logprobs, 25-shot; HellaSwag - logprobs, 10-shot; Winogrande - logprobs, 5-shot (all lm-eval-harness); GSM8k - strict match, 3-shot CoT, lm-eval-harness; MATH500 - strict match, 5-shot CoT, IN-HOUSE harness (not public); HumanEval/HumanEval+/MBPP/MBPP+ - pass@1, evalplus. Generative tasks are greedy (lm-eval default) with NO chat template - HF discussion #8: ByteDance's Ridger confirms "the paper reports log-prob results while we used a standard 5-shot setting in lm-eval", and KristianS7 reproduced MMLU 67.46 / BBH 71.06 / GSM8K 79.38 (paper 67.35 / 71.02 / 78.92) with lm-eval 0.4.11 + transformers 4.57.6 using "5-shot MMLU, 3-shot CoT for BBH and GSM8K, 25-shot ARC-C" and no apply_chat_template; applying the chat template gave 60.8/60.8. Reasoning models (Table 17): AIME24/25 (pass@1 and pass@10), OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE - "In-house harness; LLM-as-judge", "temp=1.0, top_p=0.7" - not exactly replicable. Targets (Table 7, Ouro-1.4B R4 vs Qwen3-4B base): MMLU 67.35 (73.19), MMLU-Pro 48.62 (51.40), BBH 71.02 (70.95), ARC-C 60.92 (63.65), HellaSwag 74.29 (75.66), Winogrande 72.30 (71.19), GSM8K 78.92 (72.86), MATH500 82.40 (59.60), HumanEval 74.40 (77.40), HumanEval+ 67.40 (70.70), MBPP 73.00 (78.80), MBPP+ 62.70 (65.90); Qwen3-1.7B: MMLU 62.46, BBH 53.51, GSM8K 70.28, MATH500 25.80. Local commands (task names are lm-eval's closest public equivalents; the authors' exact YAMLs were not published): `lm_eval --model hf --model_args pretrained=ByteDance/Ouro-1.4B,trust_remote_code=True,dtype=bfloat16 --device mps --batch_size 1 --tasks gsm8k_cot --num_fewshot 3 --output_path out/ --limit 200` (drop --limit for full; strict-match column); `--tasks bbh_cot_fewshot` (3-shot built in); `--tasks mmlu --num_fewshot 5`; `--tasks mmlu_pro` (5-shot CoT default); `--tasks arc_challenge --num_fewshot 25`, `hellaswag --num_fewshot 10`, `winogrande --num_fewshot 5`. Do NOT pass --apply_chat_template. Code: `evalplus.evaluate --model ByteDance/Ouro-1.4B --dataset humaneval --backend hf --greedy --trust_remote_code` (and --dataset mbpp); evalplus reports base and plus pass@1. For a paired baseline run the same commands against Qwen/Qwen3-1.7B-Base / Qwen3-4B-Base in the same harness (Ollama-served Qwen can only be used for generative tasks via lm-eval's local-completions model, not for logprob tasks). Loop ablation: repeat with model.model.total_ut_steps in {1,2,3,4} (paper Table 10 reports per-step curves; T=4 peaks). Links: https://arxiv.org/html/2510.25741v2 (Appendix C.1, Tables 7/10/16/17), https://huggingface.co/ByteDance/Ouro-1.4B/discussions/8, https://github.com/EleutherAI/lm-evaluation-harness, https://github.com/evalplus/evalplus.

**Python:** Do NOT use the system /usr/bin/python3 (3.9.6): torch>=2.9 requires Python>=3.10 (PyPI requires_python). `python3` on this Mac's PATH is actually Homebrew 3.14.7, but torch 2.14.0 macOS-arm64 wheels were confirmed only for cp310/cp311/cp312 (cp313/cp314 not checked). uv already has cpython-3.12.13 installed at ~/.local/share/uv/python/cpython-3.12-macos-aarch64-none, so: `uv venv -p 3.12 ~/ouro-venv && source ~/ouro-venv/bin/activate && uv pip install torch==2.14.0 transformers==4.57.6 accelerate==1.14.0`. transformers 4.57.6 itself supports >=3.9 but is irrelevant given torch.


---

### Nanbeige4.2-3B (Nanbeige LLM Lab / BOSS Zhipin) — 22-layer looped transformer, num_loops=2 (44 effective layers), 4.17B total params (3B non-embedding), Apache-2.0, thinking + non-thinking modes

**Repo:** Nanbeige/Nanbeige4.2-3B  
**Download:** 8.36 GB · **RAM:** ~10 GB

**Loop control:** config key `num_loops` (=2 in the shipped config.json; `loop_loss_weights=[]`, `skip_loop_final_norm=false`, `enable_double_loop_split` absent). HF transformers: NO generate()/forward() kwarg exists; modeling_nanbeige.py computes `NanbeigeModel._get_num_loops()` = 1 if enable_double_loop_split, else len(loop_loss_weights)+1 if non-empty, else config.num_loops — and it is called inside every forward() (lines ~2074/2111/2217), while NanbeigeForCausalLM.generate() creates a fresh DynamicCache whenever config.num_loops>1 and StaticCache is rejected for looped models. So the per-call knob is `model.config.num_loops = N` before each generate() (shown as set_num_loops() in the snippet). KV cache is NOT shared across loops: cache slot = layer_idx + loop_idx*num_hidden_layers (`_get_loop_cache_layer_idx`), i.e. 44 KV slots. mlx-lm: `ModelArgs.effective_num_loops` -> `self.num_loops` on NanbeigeModel (set at init) and `make_cache()` allocates num_loops*len(layers) KVCaches, so `model.model.num_loops = N` before generate() changes it per call (unverified in practice). llama.cpp: read at load from GGUF metadata `nanbeige.num_loops` / `nanbeige.skip_loop_final_norm` (PR #25994 expands logical layers with separate KV slots per loop); a `--override-kv nanbeige.num_loops=int:N` load-time override is plausible but UNVERIFIED. Trained at exactly 2 passes; the report says >2 passes gave 'only marginal' gains and destabilised training, so anything other than 2 is an ablation.

**Thinking mode:** Yes — hybrid Think / Non-Think model. Controlled purely through the chat template (tokenizer_config.json): `enable_thinking` (default True when undefined; `enable_thinking=False` makes add_generation_prompt emit `<|im_start|>assistant\n<think>\n\n</think>\n\n`, otherwise `<|im_start|>assistant\n<think>\n` and the model writes its reasoning then `</think>`), and `preserve_thinking` (whether earlier assistant turns keep their <think> blocks; README: False for chat/QA, True for multi-turn tool use / office / code-agent; ALL official benchmark numbers use thinking mode with preserve_thinking=true). Default system prompt (Chinese, BOSS Zhipin identity) is injected if none given. Official decoding: temperature 0.6 / top_p 0.95 / top_k 20, max_new_tokens 131072 for reasoning+chat, temperature 1.0 / 65536 for agentic. llama-server: `--jinja --reasoning-format deepseek` (thoughts -> message.reasoning_content) and `--chat-template-kwargs '{"enable_thinking":false}'` or `--reasoning off`; `--reasoning-budget N` caps thinking tokens. mlx-lm: pass enable_thinking=... to tokenizer.apply_chat_template exactly as with HF.

**Dependencies:**

- torch==2.8.0  (macOS arm64 wheels exist for cp39-cp313; 2.8.0 is the version verified with this model's code in HF discussion #15; MPS bf16 OK)
- transformers==4.45.1  (README pin; all cache APIs the custom code calls — DynamicCache.from_legacy_cache, get_max_length, get_seq_length, DynamicCache.__len__ — verified present in v4.45.1 cache_utils.py; do NOT use transformers 5.x with the ORIGINAL repo)
- sentencepiece>=0.2.0  (tokenizer_class=LlamaTokenizer, README loads with use_fast=False -> needs sentencepiece)
- protobuf  (slow LlamaTokenizer conversion path)
- numpy<2  (1.26.4 was the version in the verified stack; keeps 4.45.1 happy)
- safetensors>=0.4.3
- accelerate>=0.34  (OPTIONAL: only if you insist on device_map='auto'; the snippet avoids it)
- huggingface_hub[cli]>=0.24  (for hf download / huggingface-cli)
- --- alternative MLX path: mlx-lm from git main (pip install git+https://github.com/ml-explore/mlx-lm.git ; main is 0.32.0 and contains mlx_lm/models/nanbeige.py; PyPI latest 0.31.3 (2026-04-22) predates the 2026-08-29 merge and does NOT have it)
- --- alternative OptiQ path: mlx-optiq>=0.4.6 (PyPI latest 0.5.6, 2026-09-05, requires Python>=3.11)
- --- alternative GGUF path: brew install llama.cpp (Homebrew stable = 0.4.0, the release whose notes say 'Added support for nanbeige4.2-3B')

**Quantized / alternative runtimes:** GGUF (recommended for a 16 GB M4 and fits the 3.5 GB free): bartowski/Nanbeige_Nanbeige4.2-3B-GGUF, file Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf (2.68 GB; also IQ4_XS 2.44, Q4_K_S 2.55, Q5_K_M 3.08, Q6_K 3.60, Q8_0 4.43, BF16 8.34), built with llama.cpp b10159. Run:  `brew install llama.cpp`  (stable 0.4.0, whose notes list 'Added support for nanbeige4.2-3B')  then  `llama-server -hf bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M --port 8081 -c 16384 -ngl 99 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20`  (non-thinking: add `--chat-template-kwargs '{"enable_thinking":false}'` or `--reasoning off`; cap thinking with `--reasoning-budget N`); query http://localhost:8081/v1/chat/completions with model name from /v1/models. CLI: `llama-cli -hf bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M -ngl 99 --jinja`. Alternatives: owao/Nanbeige4.2-3B-GGUF, Andgihat/Nanbeige4.2-3B-GGUF (UD-Q4_K_XL 2.62 GB). MLX: (a) MercuriusDream/Nanbeige4.2-3B-mlx-4bit (2.35 GB, 4-bit gs64, logits match reference to 9.5e-05) — `uv pip install git+https://github.com/ml-explore/mlx-lm.git` (main = 0.32.0 has mlx_lm/models/nanbeige.py via PR #1597 merged 2026-08-29; PyPI 0.31.3 does not) then `mlx_lm.generate --model MercuriusDream/Nanbeige4.2-3B-mlx-4bit --prompt "hello"` or `mlx_lm.server --model MercuriusDream/Nanbeige4.2-3B-mlx-4bit --port 8082` (OpenAI-compatible); other widths MercuriusDream/Nanbeige4.2-3B-mlx-{2,3,5,6,8}bit and bf16. (b) mlx-community/Nanbeige4.2-3B-OptiQ-4bit (3.15 GB, mixed 4/8-bit, 5.5 bpw): `pip install "mlx-optiq>=0.4.6"` then `import optiq; from mlx_lm import load, generate; model, tok = load("mlx-community/Nanbeige4.2-3B-OptiQ-4bit")` or `optiq serve --model mlx-community/Nanbeige4.2-3B-OptiQ-4bit`. Ollama: ndavat/Nanbeige4.2-3B on ollama.com (arch nanbeige, Q4_K_M, 2.6 GB) exists but mainline Ollama has no nanbeige arch (see gotchas) — unverified/likely fails; official fork `git clone -b nanbeige42 https://github.com/Nanbeige/ollama.git` (README: `./ollama run nanbeige/nanbeige4.2:3b-Q4_K_M`) — unverified.

**Minimal load + generate:**

```python
# ---------------------------------------------------------------------------
# Nanbeige4.2-3B on Apple Silicon (M4, 16 GB) — HF transformers path (bf16, MPS)
# ---------------------------------------------------------------------------
# One-time setup (needs ~9 GB free disk for the bf16 checkpoint):
#   uv python install 3.11
#   uv venv --python 3.11 .venv && source .venv/bin/activate
#   uv pip install "torch==2.8.0" "transformers==4.45.1" "numpy<2" sentencepiece protobuf safetensors "huggingface_hub[cli]"
#   hf download Nanbeige/Nanbeige4.2-3B --local-dir ./Nanbeige4.2-3B      # 8.34 GB weights
#   export PYTORCH_ENABLE_MPS_FALLBACK=1   # CPU fallback for any op MPS lacks
#
# Sources: README pins transformers==4.45.1 and uses use_fast=False;
#          modeling_nanbeige.py reads the loop count from config at EVERY forward
#          (NanbeigeModel._get_num_loops -> config.num_loops when loop_loss_weights=[]).
import os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("NANBEIGE_MODEL", "Nanbeige/Nanbeige4.2-3B")  # or a local dir
EOS_ID   = 166101   # <|im_end|>, from config.json / generation_config.json

# ---- device / dtype ------------------------------------------------------
if torch.backends.mps.is_available():
    device, dtype = torch.device("mps"), torch.bfloat16   # bf16 on MPS works on torch>=2.3 for M-series
else:
    device, dtype = torch.device("cpu"), torch.bfloat16   # keep bf16 for RAM; use torch.float32 if you see NaNs on CPU

# ---- tokenizer + model ---------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True,
    attn_implementation="sdpa",      # flash_attn is not installable on macOS; "eager" is the fallback if sdpa misbehaves
    low_cpu_mem_usage=False,         # deliberately NOT device_map="auto": meta-device init is what zeroed the RoPE
                                     # inv_freq buffer in arXiv 2608.13987 (non-persistent buffer never re-materialised)
).to(device).eval()

# ---- RoPE sanity check (bug #4 in arXiv 2608.13987) ----------------------
def fix_rope_if_zeroed(m):
    for name, mod in m.named_modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "dim") and hasattr(mod, "base"):
            if not torch.any(mod.inv_freq != 0):
                inv = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, dtype=torch.int64).float() / mod.dim))
                mod.inv_freq = inv.to(mod.inv_freq.device)
                print(f"[fix] recomputed zeroed inv_freq in {name}", file=sys.stderr)
fix_rope_if_zeroed(model)

# ---- loop / recurrence count, settable PER CALL ---------------------------
# Shipped config: num_loops=2, loop_loss_weights=[] (no intermediate-loop heads).
# _get_num_loops() is evaluated inside every forward() and generate() builds a fresh
# DynamicCache per call, so mutating config.num_loops between calls is safe.
# The model was TRAINED at 2 loops; 1 or 3 are for ablation only (expect degraded output).
def set_num_loops(m, n: int):
    assert n >= 1
    m.config.num_loops = n
    m.config.loop_loss_weights = []          # must stay empty or it overrides num_loops
    m.config.enable_double_loop_split = False

# ---- chat template + thinking-mode toggles --------------------------------
def build_prompt(messages, thinking=True, preserve_thinking=False):
    # enable_thinking  : default True; False makes the template emit '<think>\n\n</think>\n\n' (non-thinking mode)
    # preserve_thinking: keep earlier assistant <think> blocks in multi-turn; README: True for agents/tools, False for chat
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking, preserve_thinking=preserve_thinking,
    )

@torch.inference_mode()
def chat(messages, thinking=True, preserve_thinking=False, num_loops=2,
         max_new_tokens=4096, temperature=0.6, top_p=0.95, top_k=20):
    set_num_loops(model, num_loops)
    prompt = build_prompt(messages, thinking, preserve_thinking)
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)  # template already contains <|im_start|>
    t0 = time.time()
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,      # authors use 131072 (reasoning) / 65536 (agentic); local runs need a budget
        do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,   # official settings (README + report App. B.1)
        eos_token_id=EOS_ID, pad_token_id=tok.pad_token_id or 0,
        use_cache=True,
    )
    new = out[0, enc.input_ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    dt = time.time() - t0
    reasoning, _, answer = text.partition("</think>")
    if not _:                      # no closing tag (non-thinking mode or budget exhausted)
        reasoning, answer = "", text
    return {"reasoning": reasoning.strip(), "answer": answer.strip(),
            "new_tokens": int(new.numel()), "tok_per_s": new.numel() / dt}

if __name__ == "__main__":
    msgs = [{"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}]
    print(chat(msgs, thinking=False, num_loops=2))                 # fast non-thinking answer
    print(chat(msgs, thinking=True,  num_loops=2, max_new_tokens=2048))  # thinking mode (official eval mode)
    # ablation: run the shared stack once / three times (NOT what it was trained for)
    print(chat(msgs, thinking=False, num_loops=1)["answer"])
    print(chat(msgs, thinking=False, num_loops=3)["answer"])

# ---------------------------------------------------------------------------
# MPS-verified fallback if the original repo misbehaves on your transformers/torch combo:
#   uv pip install -U "transformers>=5.8" torch     # johnhalloran's config.json was saved by transformers 5.8.1
#   NANBEIGE_MODEL=johnhalloran/Nanbeige4.2-3B-mps-fix python this_script.py
# (same code; that checkpoint bakes in inv_freq as a persistent buffer + 4 other fixes; its
#  harness `python harness/nanbeige_harness_server.py --port 8100` gives an OpenAI-compatible server)
# ---------------------------------------------------------------------------
```

**Gotchas:**

- DISK: the bf16 checkpoint is 8.34 GB (4,973,547,960 + 3,366,076,760 bytes) + 18.5 MB tokenizer.json — it does NOT fit in the ~3.5 GB currently free; free >=10 GB first or start with bartowski Q4_K_M GGUF (2.68 GB) / MercuriusDream MLX 4-bit (2.35 GB).
- RAM (16 GB M4): bf16 weights 8.34 GB + KV cache. Because KV is NOT shared across the 2 loops there are 44 KV slots: 2 loops x 22 layers x 8 KV heads x 128 dim x (K+V) x 2 bytes = ~176 KB/token -> ~0.7 GB at 4k ctx, ~5.6 GB at 32k. Halloran (arXiv 2608.13987) measured on 32 GB and reports layer reuse 'doubles peak attention memory'; naive prefill fails around 8k tokens and needs chunked prefill. Expect ~10 GB peak for short prompts on the HF path; keep prompts <4k on 16 GB. Q4_K_M GGUF needs only ~4-5 GB total.
- SPEED: HF/MPS path is slow — Halloran's patched harness: 15.5 tok/s decode at trivial context falling to 2.1 tok/s at 5,120-token context; HF discussion #31 users see ~half the speed of Gemma-4-E4B and the team confirms 'the Looped Transformer and disabled KV sharing ... result in slower decoding'. mlx-lm PR #1597 test: 18.6 tok/s for 1,024 tokens with 8.6 GB peak (bf16, Mac unspecified). No llama.cpp Mac tok/s numbers were found anywhere. With official max_new_tokens of 131072 in thinking mode, a single HMMT problem could take hours locally — budget max_new_tokens.
- TRANSFORMERS VERSION TRAP: README says transformers==4.45.1, config.json was written by 4.42.4, generation_config.json by 4.51.0. The custom code uses DynamicCache.from_legacy_cache, Cache.get_max_length (returns None), DynamicCache.__len__, `if past_key_values:` truthiness — all present in 4.45.1 (verified in v4.45.1 cache_utils.py) but changed/removed in transformers 5.x, which is exactly what produced Halloran's 'removed cache-API call', 'get_max_cache_shape returns -1', 'TypeError tuple + int' and MPS position-ids matmul crash. Either pin 4.45.1 with the ORIGINAL repo, or use transformers>=5.8 with johnhalloran/Nanbeige4.2-3B-mps-fix (its config was saved by 5.8.1). Do not mix.
- RoPE ZEROED (silent gibberish, looping </think>): inv_freq is a non-persistent buffer; when the model is materialised on the meta device (device_map='auto' / low_cpu_mem_usage in newer transformers) it is never repopulated -> 'character-level word salad'. The snippet avoids device_map and re-checks inv_freq after load. If you still get salad, switch to the mps-fix checkpoint (persistent inv_freq).
- rope_scaling KeyError on newer transformers: code does `config.rope_scaling['type']`; recent transformers may populate rope_scaling with a dict lacking 'type' (Halloran bug #1). Not an issue on 4.45.1 with rope_scaling=null.
- ATTENTION IMPL: flash_attn import is guarded by is_flash_attn_2_available() (False on macOS) so pass attn_implementation='sdpa' (or 'eager'); never 'flash_attention_2'. The RoPE forward already special-cases MPS (disables autocast on cpu for the freq matmul). StaticCache / cache_implementation='static' raises ValueError for looped models; only DynamicCache works.
- bf16 on MPS: fine on M-series with torch>=2.3 (torch 2.8.0 verified with this model on CUDA; MPS bf16 untested by anyone I could cite). If you hit an unsupported MPS op, set PYTORCH_ENABLE_MPS_FALLBACK=1; fp16 on MPS risks overflow in the second loop pass — prefer bf16, or float32 on CPU.
- TOKENIZER: tokenizer_class is LlamaTokenizer (tokenizer.model 2.8 MB + tokenizer.json 18.5 MB present; no vocab.json/merges.txt/chat_template.jinja — template lives in tokenizer_config.json). README loads with use_fast=False -> requires sentencepiece + protobuf. add_bos_token=true but the README tokenizes the rendered template with add_special_tokens=False (the template already emits <|im_start|>) — do the same or you double-BOS. bos=<|im_start|> (166100), eos=<|im_end|> (166101), pad=<unk> (0).
- save_pretrained() on newer transformers fails (AttributeError 'list' has no attribute 'keys' from _get_tied_weight_keys) — Halloran bug #5; irrelevant for inference on 4.45.1.
- OLLAMA: mainline Ollama does NOT support the 'nanbeige' architecture — GitHub code search for 'nanbeige' in ollama/ollama returns 0 hits, latest release v0.33.3 (2026-09-02, GitHub API) has no mention, ollama/ollama#14266 closed without a PR, and Andgihat's GGUF card warns stock Ollama/LM Studio fail with unknown architecture. The ollama.com upload ndavat/Nanbeige4.2-3B (arch 'nanbeige', Q4_K_M, 4.17B, ctx 4096, 1,559 pulls, no readme) therefore almost certainly fails to load on mainline — unverified. The Nanbeige/ollama fork branch nanbeige42 is 1 commit ahead ('support nanbeige4.2 model', 2026-07-20) and 185 behind upstream; its diff adds a nanbeige renderer/parser, a KV-size estimator that multiplies layers by num_loops, and an MLX-runner model (x/models/nanbeige/nanbeige.go) — it does NOT touch the vendored llama.cpp C++, so its GGUF path is unclear. Practical pairing: run `llama-server` (OpenAI-compatible /v1/chat/completions on :8081) next to `ollama serve` (:11434) serving baselines such as qwen3.5:4b (3.4 GB) / qwen3.5:9b (6.6 GB), and point your eval client at both. I could NOT confirm any Ollama-compatible /api/chat shim in llama-server (0 code-search hits for 'api/tags', nothing in the server README) — treat it as OpenAI-compatible only.
- llama.cpp: need build >= b10199 / release 0.4.0 (brew's stable is 0.4.0). bartowski's GGUFs were made with b10159 (before the DSpark PR #27730); they load per discussion #23 but the PR author noted quantized-KV paths and perplexity vs reference were not extensively validated — avoid -ctk/-ctv q8_0. HF discussion #17 reports llama.cpp silently dropping some tool calls (parser bug). Default Ollama-style ctx of 4096 in ndavat's upload is far below the 256k the model supports.
- mlx-lm: PyPI 0.31.3 (2026-04-22) does NOT contain nanbeige.py; install from git main (0.32.0). loop_share_kv / enable_double_loop_split / enable_depth_attention raise NotImplementedError there (none are enabled in the 4.2 checkpoint). mlx-community OptiQ-4bit needs `import optiq` (mlx-optiq>=0.4.6, Python>=3.11) before mlx_lm.load.
- Quantization vs replication: all published numbers are bf16 with 256k context and up to 131k output tokens; Q4 GGUF/MLX numbers will differ. For local replication use the bf16 HF/MLX-bf16 path on short benchmarks (GSM8K/MATH-500/AIME-style) with a capped thinking budget and report the cap.
- Baseline: the model is NOT Qwen-derived (pretrained from scratch on 28T tokens); the published 'comparable' baselines are Qwen3.5-4B / Qwen3.5-9B / Gemma4-E4B / Gemma4-12B (instruct) and Qwen3.5-4B-Base / Gemma4-E4B-Base / Nanbeige4-3B-Base for the base model. Nanbeige/Nanbeige4.2-3B-Base exists on HF (public, lastModified 2026-08-02) if you want a same-architecture non-looped comparison — note there is no released 1-loop variant; the non-looped predecessor Nanbeige4.1-3B is a plain 32-layer Llama.

**Official eval recipe:** Source: tech report arXiv 2607.22083 (Nanbeige42_report.pdf in the HF repo), Appendix B 'Evaluation settings', plus README table footnotes. B.1 General inference settings (unless otherwise specified): temperature 0.6, top-p 0.95, top-k 20, context window 256k tokens; README adds max_new_tokens 131,072 for reasoning/chat and temperature 1.0 / 65,536 for agentic tasks; generation_config.json ships do_sample=true, temperature 0.6, top_k 20, top_p 0.95, eos 166101. Mode: 'All evaluations are conducted in thinking mode with preserve_thinking=true in the chat template' (README footnote 1); for judged agent tasks the text before the closing </think> is stripped before scoring. Prompting is zero-shot chat via the chat template (no few-shot is described anywhere for the instruct model). Reasoning benchmarks reported (thinking mode): GPQA-Diamond 87.4, HMMT-Feb-2026 82.8, IMO-Answer-Bench 67.3, LiveCodeBench-V6 72.5, HLE w/o search 17.8, SciCode 35.6, AA-LCR 58.7, IF-Bench 54.6 vs Qwen3.5-9B / Qwen3.5-4B / Gemma4-12B / Gemma4-E4B — the harness and avg@k for these reasoning benchmarks are NOT stated (not confirmed). Code agent (B.2): SWE-bench Verified = OpenHands scaffold, 256k ctx, 32k max output, temperature 1.0, 4 h timeout, 250 turns, averaged over 8 runs; SWE-bench Pro = SWE-agent, 256k/32k, T=1.0, 10 h, 250 turns, 8 runs; Terminal-Bench 2.0 = Harbor/Terminus-2 with JSON parser, 256k/32k, T=1.0, 8 CPU / 24 GB, 4 h, 250 turns, 8 runs. General agent (B.3): GDPval-rubrics / AgentIF-Oneday / OfficeQA-Pro via in-house harness (web search + sandbox, agent-as-judge, 0-100); Claw-Eval 157 general tasks, full reasoning kept in context (sglang reasoning parser off), DeepSeek-V4-Pro judge, 3,600 s; MCP-Atlas with system prompt, GLM-5.1 judge, 1,200/3,600 s timeouts, <=20 tool rounds. OpenClaw (B.4): PinchBench-V2/Claw-Gym with DuckDuckGo search, 10,800 s, Qwen3.7-Plus judge; DeepResearch Bench II / ResearchRubrics with Brave Search, 3,600 s, 200 turns, DeepSeek-V4-Flash judge at T=0. Base model (Table 1, report sec. 2.3): Nanbeige4.2-3B-Base GSM8K 92.7, BBH 81.6, MBPP 67.6, MMLU-Pro 63.8, SuperGPQA 35.2, GPQA 53.3 vs Qwen3.5-4B-Base (84.4/79.1/57.1/51.8/32.1/43.1) and Gemma4-E4B-Base — harness/few-shot for these base evals NOT stated. Architecture facts for replication: 2-pass loop chosen because it 'retains approximately 75% of the token efficiency' of a standard transformer with significant capacity gain, more passes gave marginal gains and unstable training; KV-sharing across loops was tried and rejected for quality (report sec. 2.1). Serving stack used by the authors: their sglang fork (branch nbg42, --reasoning-parser nanbeige --tool-call-parser nanbeige) / vllm fork (branch nanbeige42). Links: https://arxiv.org/abs/2607.22083 , https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/README.md , https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/generation_config.json

**Python:** Use Python 3.11 via uv (`uv python install 3.11 && uv venv --python 3.11 .venv`). transformers 4.45.1 declares requires_python>=3.8 and torch 2.8.0 ships a cp39 macOS-arm64 wheel, so the system 3.9 could technically run the HF path, but (a) 3.9 is EOL and untested with this stack, (b) johnhalloran's MPS harness requires >=3.10, (c) mlx-optiq requires >=3.11. The only reported working transformers stack for this model used Python 3.11.15 (HF discussion #15). Do not use 3.13 with transformers 4.45.1 (never tested by HF for that version).


---

### Ouro-1.4B-Thinking (ByteDance Seed looped / recurrent-depth LM, 24 shared layers x 4 UT steps, reasoning-SFT variant; trained from scratch, NOT Qwen-derived)

**Repo:** ByteDance/Ouro-1.4B-Thinking (main = commit 3aaa2224, last modified 2026-06-03; Apache-2.0; not gated)  
**Download:** 2.88 GB · **RAM:** ~6 GB

**Loop control:** config.total_ut_steps (int, default 4). It is NOT a generate()/forward() kwarg. modeling_ouro.py: `self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)` in OuroModel.__init__ and `for current_ut in range(self.total_ut_steps)` in forward; the KV cache index is `current_ut * num_hidden_layers + layer_idx`, and `max_cache_size = num_hidden_layers * config.total_ut_steps` is recomputed from config on every forward. So: (a) before load: `config = AutoConfig.from_pretrained(..); config.total_ut_steps = N` (README method); (b) after load / per call: set BOTH `model.config.total_ut_steps = N` AND `model.model.total_ut_steps = N` (verified N=1, 4, 6 on MPS). Setting only model.model.total_ut_steps > config value reproduces `IndexError: Cache index 8 exceeds configured max_cache_size=8` (same bug class as open HF PR #7). Per-call knobs that ARE forward kwargs (verified they pass through generate()): `exit_at_step=k` (use step-k hidden state for logits; all steps still computed), `exit_threshold=p` (adaptive exit on the learned early_exit_gate), `use_weighted_exit=True`; config.early_exit_threshold=1.0 = never exit early (paper numbers are at fixed R4). vLLM ignores adaptive exit.

**Thinking mode:** Yes. tokenizer_config.json chat_template (added by community PR #4 merged 2026-02-26, commit b80c8178) supports `enable_thinking`: with `enable_thinking=True` the generation prompt becomes `<|im_start|>assistant\n<think>\n`; with False/omitted it is just `<|im_start|>assistant\n` (verified by rendering the template). Special tokens: <think>=id 3, </think>=id 4, <|im_end|>=id 2 (eos), <|im_start|>=id 1 (bos). Default system prompt 'You are a helpful assistant.' is injected if none given. The model is SFT'd on OpenThoughts3/AceReason-style traces, so it may emit <think>...</think> on its own even without the prefix; the paper/README never mention enable_thinking, and I could not verify (no weights downloaded) whether omitting the prefix actually suppresses thinking. Paper results are for full reasoning traces; the README quick-start uses no prefix at all.

**Dependencies:**

- python>=3.10 (used 3.12.13 via uv)
- torch==2.14.0 (macOS arm64 wheel; MPS available; verified)
- transformers==4.57.6 (verified; do NOT use 4.54.1/4.55.0 despite README, and NOT 5.x)
- accelerate (latest; only needed for device_map)
- safetensors
- huggingface_hub
- lm_eval==0.4.13 (only for evaluation)
- OPTIONAL: mlx-lm==0.29.1 was installed only to prove 'Model type ouro not supported' - do not rely on it
- NOT needed: flash-attn, vllm, bitsandbytes

**Quantized / alternative runtimes:** MLX: mlx-community/Ouro-1.4B-Thinking-4bit exists (807 MB, converted with mlx-lm 0.28.4 on 2025-11-28, config num_hidden_layers=24, total_ut_steps=4, 4-bit group 64) and its card says `pip install mlx-lm; mlx_lm.chat --model mlx-community/Ouro-1.4B-Thinking-4bit`, but it does NOT run: mlx-lm (0.29.1 installed, 0.31.3 latest) has no mlx_lm/models/ouro.py and `_get_classes({'model_type':'ouro'})` raises 'ValueError: Model type ouro not supported' (verified locally); the repo's modeling_ouro.py is the PyTorch file. Also its chat_template.jinja lacks enable_thinking. GGUF: none for the 1.4B; only AXIOM-TECH/Ouro-2.6B-Thinking-IBNN-GGUF (2.6B) with no llama.cpp architecture support found. Net: no usable MLX/GGUF/Ollama path - use PyTorch MPS via transformers.

**Minimal load + generate:**

```python
# ouro_mps.py  -- verified structure on Apple Silicon (M-series, macOS 26.6.2, torch 2.14.0, transformers 4.57.6, Python 3.12)
# Run:  uv venv --python 3.12 ~/ouro-venv && VIRTUAL_ENV=~/ouro-venv uv pip install torch transformers==4.57.6 accelerate safetensors huggingface_hub
#       ~/ouro-venv/bin/python ouro_mps.py
import os, re, time, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = "ByteDance/Ouro-1.4B-Thinking"
LOOPS = int(os.environ.get("OURO_LOOPS", 4))   # recurrence count R (trained at 4; paper shows R5-R8 extrapolate with degradation)

# ---- device / dtype -------------------------------------------------------
if torch.backends.mps.is_available():
    device, dtype = "mps", torch.bfloat16      # bf16 on MPS worked in our smoke test (torch 2.14, macOS 26); use float16 if you get "BFloat16 is not supported on MPS"
elif torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
else:
    device, dtype = "cpu", torch.float32       # CPU bf16 also worked but is slow; fp32 is safest on CPU

# ---- loop count is a CONFIG field, not a generate() kwarg -----------------
config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
config.total_ut_steps = LOOPS          # modeling_ouro.py: OuroModel.__init__ reads getattr(config,"total_ut_steps",4)
config.early_exit_threshold = 1.0      # 1.0 = always run all steps (adaptive early exit off); lower => earlier exit

tok = AutoTokenizer.from_pretrained(REPO)      # GPT2TokenizerFast; pad_token=<|im_end|> comes from special_tokens_map.json
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    config=config,
    trust_remote_code=True,            # REQUIRED: architecture lives in modeling_ouro.py on the Hub (no pip package)
    dtype=dtype,                       # `torch_dtype=` still works but is deprecated in 4.57
    attn_implementation="sdpa",        # "sdpa" (default) and "eager" both verified on MPS; there is no flash-attn on Mac
    low_cpu_mem_usage=True,
).to(device).eval()

def set_loops(n: int):
    """Change recurrence count AFTER loading. You MUST set BOTH attributes, otherwise the
    UniversalTransformerCache is sized num_layers*config.total_ut_steps and raises
    IndexError 'Cache index N exceeds configured max_cache_size' (this is the bug behind HF PR #7)."""
    model.config.total_ut_steps = n
    model.model.total_ut_steps = n

def chat(user, system=None, thinking=True, max_new_tokens=2048, temperature=1.0, top_p=0.7, loops=None, **fwd_kwargs):
    if loops is not None:
        set_loops(loops)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    # enable_thinking=True prepends "<think>\n" to the assistant turn (chat_template in tokenizer_config.json, added Feb-2026 PR #4).
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=thinking,
                                  return_tensors="pt", return_dict=True).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature, top_p=top_p,   # paper/README decoding for Thinking models
            eos_token_id=tok.eos_token_id,           # <|im_end|> (id 2)
            pad_token_id=tok.pad_token_id,
            **fwd_kwargs,   # optional per-call exit control forwarded to OuroForCausalLM.forward:
                            #   exit_at_step=k   -> read logits from recurrence step k (0-based); all steps still computed
                            #   exit_threshold=p -> adaptive exit when cumulative gate prob >= p
        )
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
    n = out.shape[1] - enc["input_ids"].shape[1]
    m = re.search(r"(.*?)</think>(.*)", text, re.S)
    think, answer = (m.group(1), m.group(2)) if m else ("", text)
    answer = answer.replace("<|im_end|>", "").strip()
    return {"think": think.strip(), "answer": answer, "tokens": n, "tok_per_s": n / (time.time() - t0), "loops": model.model.total_ut_steps}

if __name__ == "__main__":
    print(f"device={device} dtype={dtype} attn={model.config._attn_implementation} loops={model.model.total_ut_steps}")
    r = chat("Solve: If 2x + 3 = 11, what is x? Put the final answer in \\boxed{}.", thinking=True, max_new_tokens=1024)
    print(r["answer"], f"\n[{r['tokens']} tok, {r['tok_per_s']:.1f} tok/s, R={r['loops']}]")
    # same prompt with 2 loops (cheaper, worse) and 6 loops (extrapolation), greedy for comparability
    for L in (2, 6):
        r = chat("What is 17*23?", thinking=False, loops=L, temperature=0.0, max_new_tokens=64)
        print(L, r["answer"][:120])
    # read the answer from recurrence step 1 without changing the loop count (Table 12-style analysis)
    r = chat("What is 17*23?", thinking=False, loops=4, temperature=0.0, max_new_tokens=64, exit_at_step=1)
    print("exit_at_step=1:", r["answer"][:120])

# ---- OpenAI-compatible server next to Ollama (no native Ollama/GGUF path exists) ----------------
# ~/ouro-venv/bin/transformers serve --host 127.0.0.1 --port 8001 --trust_remote_code --dtype bfloat16 --device mps
# then: curl http://127.0.0.1:8001/v1/chat/completions -d '{"model":"ByteDance/Ouro-1.4B-Thinking","messages":[{"role":"user","content":"hi"}]}'
# (flags verified in `transformers serve --help` for 4.57.6; end-to-end serving of Ouro NOT verified). Ollama stays on :11434 for baselines.
```

**Gotchas:**

- DISK: weights are a single model.safetensors of 2,869,336,434 bytes (2.87 GB) + ~5 MB tokenizer/code; the Python venv with torch 2.14 is another ~0.8-0.95 GB. The machine showed 3.9 GiB free at start (6.5 GiB after purgeable space was reclaimed during install) - free space first; HF cache lives in ~/.cache/huggingface unless HF_HOME is set.
- RAM: bf16 weights ~2.9 GB, but the UniversalTransformerCache stores K/V for EVERY recurrence step: 24 layers x 4 steps x 2 x 16 heads x 128 dim x 2 B = 0.75 MiB per token (4x a normal 24-layer model). 4k context ~3 GB KV, 8k ~6 GB, 16k ~12 GB -> on 16 GB unified memory keep prompt+generation <= ~8k tokens (paper eval used max_new_tokens=8192); R=6/8 scales the cache 1.5x/2x. The cache also grows by torch.cat every step (O(n^2) copies), so long generations get slow. Compute is 4x a 1.4B dense model per token (~5.7B-param-equivalent FLOPs): expect low-single-digit to ~15 tok/s on an M4 in bf16 - budget for hours for AIME-style evals.
- TRANSFORMERS VERSION: README (and config.json transformers_version 4.55.0) say transformers==4.54.1 / <4.56 - this is STALE and wrong. Base-repo PR #14 tests: <=4.52.4 ImportError 'layer_type_validation'; 4.54.1/4.55.0 AttributeError "property 'key_cache' of 'UniversalTransformerCache' object has no setter"; 4.57.1 works. I verified 4.57.6 (latest 4.x) loads the custom code and generates on MPS/CPU. transformers 5.x: unsupported (PR #13 'Fix for transformers 5.x.x' is an empty +0/-0 draft); do not install 5.16.x.
- OPEN BUG PR #7 (AXIOM-TECH, 2026-06-01, unmerged): IndexError 'Cache index 24 exceeds configured max_cache_size=24'. Root cause from code: max_cache_size = num_hidden_layers * config.total_ut_steps while the loop count comes from model.model.total_ut_steps; they diverge if config lacks total_ut_steps or you change only one. Fix: always set both attributes (see set_loops) - then the shipped code works; PR #7's branch is also stale (it reverts the June-3 RoPE fix), so do not apply it wholesale.
- MPS: no reports existed before; my smoke test (tiny random model built from the real modeling_ouro.py/configuration_ouro.py + real tokenizer, transformers 4.57.6, torch 2.14.0, macOS 26.6.2) passed on device=mps for bf16/fp16/fp32 with attn_implementation sdpa and eager, loops 1/4/6, batched left-padded generation, exit_threshold and exit_at_step kwargs, and CPU bf16/fp32. Real 1.4B weights were NOT run (disk budget), so numerical quality on MPS bf16 is unverified. If you see 'BFloat16 is not supported on MPS' (older macOS/torch), fall back to float16 (or float32, ~5.7 GB weights). RoPE is computed in fp32 with autocast disabled and the code special-cases device 'mps'.
- ATTENTION: config.json ships no _attn_implementation, so transformers picks sdpa (verified). flash_attention_2 is unavailable on Mac. If MPS sdpa misbehaves with masks, pass attn_implementation='eager' (verified works).
- TOKENIZER/PAD: config.pad_token_id ends up None (configuration_ouro.py sets it before super().__init__ overwrites it) - this is the base-repo #12 'no attribute pad_token_id' symptom on old transformers. The tokenizer DOES have pad_token=<|im_end|> via special_tokens_map.json (PR #2 that adds it to tokenizer_config.json is redundant/open). For batched generation use padding_side='left' and pass pad_token_id=tok.pad_token_id to generate (verified). bos in tokenizer_config is <|im_start|> (id 1) while special_tokens_map says <|endoftext|>; the chat template never emits a bare BOS so this does not matter for chat.
- generation_config.json does not exist in the repo: generate() defaults to greedy unless you pass do_sample=True, temperature=1.0, top_p=0.7 (README/paper setting). eos_token_id=2 (<|im_end|>) comes from config.json; still pass it explicitly.
- NO OFFICIAL CODE REPO: README 'Code' link points to github.com/Ouro-LLM/Ouro (404); github.com/ByteDance/Ouro also 404; project page says 'Code (Coming Soon)'. The only official implementation is modeling_ouro.py on the Hub. rkstgr/LoopLM is an unofficial re-implementation with no documented HF-weight loading or MPS support.
- The Thinking-repo README is a near-verbatim copy of the 2.6B card; one raw fetch during this session even returned the 2.6B config/README (48 layers) for the 1.4B URL - transient Hub caching. Always verify after download: config num_hidden_layers must be 24 (safetensors header has model.layers.0-23, 269 tensors, 1,434,652,673 BF16 params).
- NO OLLAMA PATH: no GGUF for the 1.4B exists (only AXIOM-TECH/Ouro-2.6B-Thinking-IBNN-GGUF for the 2.6B, with no evidence llama.cpp implements the loop); llama.cpp has no 'ouro' architecture. Serve via `transformers serve` (OpenAI-compatible) or your own FastAPI wrapper on a second port beside Ollama.
- MLX PORT IS NOT RUNNABLE: mlx-community/Ouro-1.4B-Thinking-4bit ships quantized weights but mlx-lm has no models/ouro.py; `mlx_lm.utils._get_classes({'model_type':'ouro'})` on mlx-lm 0.29.1 raises 'ValueError: Model type ouro not supported' (verified). Its modeling_ouro.py is the PyTorch/transformers file, not MLX code.
- EVAL MISMATCH: the authors' Thinking-model numbers come from an in-house harness with LLM-as-judge (Table 17); the harness, judge model, prompt, max tokens for those tables, and number of samples averaged are NOT published, so exact replication is impossible. lm-eval's aime24/aime25 tasks use greedy decoding, max_gen_toks 32768, 'Question: ... Answer:' format, no chat template - different from the paper. Base-model reproduction issue #8: applying the chat template to base-model evals tanks scores; run base evals without chat template.
- Loop-count extrapolation: Table 12 shows AIME24 pass@1 65.0 at R4 but 60.7 at R5 and 38.7 at R8; R1 is ~0. Do not expect gains beyond R4-R5.
- The arXiv 2608.13987 report on a different looped model (Nanbeige4.2-3B) on Apple Silicon lists MPS pitfalls (zeroed RoPE inv_freq buffer after device move, 'mps.matmul contracting dimensions differ', cache API sentinels) and notes looped models roughly double prefill memory; Ouro's code registers inv_freq as a non-persistent buffer computed at init, which worked in my test, but watch for garbage output as a RoPE symptom.

**Official eval recipe:** Paper: arXiv 2510.25741 (v1 2025-10-29; v5 is current), Appendix C.1 Tables 16-17, Sections 5.1-5.3 and 7.1. BASE models (Table 16): lm-eval-harness [59] and evalplus [60]: MMLU logprobs 5-shot; MMLU-Pro strict-match 5-shot CoT; BBH strict-match 3-shot CoT; ARC-C logprobs 25-shot; HellaSwag logprobs 10-shot; Winogrande logprobs 5-shot; GSM8K strict-match 3-shot CoT; MATH500 strict-match 5-shot CoT (in-house); HumanEval/HumanEval+/MBPP/MBPP+ pass@1 via evalplus. Base greedy decoding with max_new_tokens=128 (Sec 7.1). Reproduction note (HF Ouro-1.4B discussion #8): scores match the paper only WITHOUT the chat template; using it drops GSM8K 78.9->60.8. THINKING models (Table 17 / Sec 5.2): 'a single in-house harness and identical prompting', LLM-as-judge with fixed rubric and tie-breaking, temperature=1.0, top_p=0.7 for all models; Sec 7.1: 'For Ouro Thinking models, we sample with temperature=1.0, top_p=0.7 with max_new_tokens=8192' (stated for the HEx-PHI safety eval; the max length for Table 9 is not stated). AIME24/25 reported as pass@1 / pass@10 (Table 9); number of samples for pass@1 not stated; judge model, prompt template and system prompt not published; harness code not released (project page 'Code (Coming Soon)'). Table 9 targets, Ouro-1.4B-Thinking-R4: AIME24 65.0 / pass@10 83.3; AIME25 46.3 / 73.3; OlympiadBench 71.6; BeyondAIME 34.0; HLE 5.21; SuperGPQA 47.4; GPQA 45.5 (baselines: Qwen3-1.7B AIME24 32.0, Qwen3-4B 61.3, DS-Distill-Qwen-1.5B 29.6, DS-Distill-Qwen-7B 57.3). MATH500 82.4 is the BASE Ouro-1.4B (Table 7), not the Thinking model. Table 12 (Thinking-1.4B vs recurrence T=1..8): OlympiadBench 2.22/59.70/70.67/71.55/72.30/69.48/69.04/66.81; SuperGPQA 2.03/33.07/44.50/47.37/48.73/46.15/45.29/42.88; AIME24 0.00/37.33/62.33/65.00/60.67/50.67/42.33/38.67; AIME25 0.33/25.00/43.33/46.30/47.00/43.00/41.00/38.00. Closest local approximation (lm_eval 0.4.13, HF backend on MPS): `lm_eval --model hf --model_args pretrained=ByteDance/Ouro-1.4B-Thinking,trust_remote_code=True,dtype=bfloat16,device=mps --tasks aime24,aime25,hendrycks_math500,gsm8k_cot --apply_chat_template --gen_kwargs do_sample=True,temperature=1.0,top_p=0.7,max_gen_toks=8192 --batch_size 1 --log_samples --output_path out/` repeated N seeds for pass@1; for the Qwen3 baselines run the identical command (lm-eval's aime tasks default to greedy/32768/no chat template, so override as above). Not verified end-to-end; lm-eval's exact-match extraction differs from the paper's LLM judge, so expect a few points' deviation.

**Python:** System Python 3.9 cannot be used: transformers 4.57.x requires Python >=3.10.0 and torch 2.14.0 requires >=3.10 (both from PyPI metadata). uv already has cpython-3.12.13 installed on this machine (`uv python list` shows /Users/patrickli/.local/share/uv/python/cpython-3.12-macos-aarch64-none). Recipe verified with: `uv venv --python 3.12 ~/ouro-venv && VIRTUAL_ENV=~/ouro-venv uv pip install torch transformers==4.57.6 accelerate safetensors huggingface_hub`. Installed venv size measured: ~950 MB (that included mlx-lm; ~800 MB without). A ready-made venv from this verification exists at /private/tmp/claude-501/-Users-patrickli-Documents-vibe/c637d19a-be90-4272-9fe5-c46d4bd25e1d/scratchpad/ouro-venv (session scratchpad; may be deleted).


---

### Huginn-0125 (RavenForCausalLM, recurrent-depth 3.5B: 2 prelude + 4 weight-tied recurrent + 2 coda layers, n_embd 5280, mean_recurrence 32)

**Repo:** tomg-group-umd/huginn-0125  
**Download:** 15.65 GB · **RAM:** ~8.5 GB

**Loop control:** num_steps (int, per call): model(input_ids, num_steps=N) and model.generate(input_ids, generation_config, num_steps=N, tokenizer=tok). It must be a call kwarg, never a GenerationConfig field (model card: 'num_steps and other model arguments CANNOT be included in the GenerationConfig'). If omitted in eval mode, randomized_iteration_sampler() returns config.mean_recurrence (=32), which can be overridden at load: from_pretrained(..., mean_recurrence=N) — this is how lm-eval's model_args mean_recurrence=32 works (verified: config.mean_recurrence becomes N). Trained mean 32 (poisson-lognormal sampling); model card: <4 steps = very coarse, gains up to ~64, no hard max; materialized params ~ num_steps*1.5B + 2B. Adaptive per-token depth: criterion in {entropy-diff, latent-diff, cosine, kl, minp-kl, argmax-stability, none}, exit_threshold (float or 'auto'; auto = 1e-3 entropy-diff / 0.03 latent-diff / 1e-3 kl / 1e-5 minp-kl / 5 steps argmax-stability), min_steps, do_not_exit_in_prefill, check_criterion_every_n_steps; KV sharing: cache_lookup_strategy in {'full','latest-m4','available-m4','always-last-m4','skip','randomized','compress-s16', 'latest-m4-compress-s16'}; continuous_compute=True warm-starts the latent state from the previous token.

**Thinking mode:** None. Huginn-0125 has no thinking/reasoning toggle, no <think> tokens, and no post-training (model card: 'not finetuned or post-trained, but ... natively understand its chat template'). The only 'reasoning dial' is latent depth: num_steps (fixed) or criterion/exit_threshold (adaptive early exit), plus continuous_compute=True ('continuous CoT' warm-start). For visible chain-of-thought, prompt for it (the authors' GSM8K eval uses lm-eval gsm8k_cot 8-shot CoT with system prompt 'You are a helpful assistant that can assist users with mathematical reasoning.').

**Dependencies:**

- torch==2.14.0 (PyPI latest, requires Python >=3.10; MPS backend built in)
- transformers==4.57.6 (MUST be <5: 5.x fails in tie_weights on the remote code's list-typed _tied_weights_keys; 4.57.6 is the last 4.x on PyPI, uploaded 2026-01-16)
- accelerate==1.14.0 (needed for device_map={'':'mps'} and lm-eval)
- safetensors==0.8.0
- huggingface_hub==0.36.2 (provides the `hf download` CLI)
- lm_eval==0.4.13 (lm-evaluation-harness, for the official eval recipe)
- datasets==5.0.1 (pulled by lm_eval)
- fastapi==0.141.1 + uvicorn==0.52.4 (only for the Ollama/OpenAI-compatible shim in serve/shim.py)

**Quantized / alternative runtimes:** none found. HF API search for 'huginn' (69 repos) returns only the original tomg-group-umd/huginn-0125, two plain safetensors mirrors (nikvisel/huginn-0125, Nemesispro/huginn-0125), JonasGeiping/huginn-0125-checkpoints, the tomg-group-umd/huginn_swa_* weight-averaged merges, and unrelated 2023-24 Llama merges (TheBloke/LoneStriker/mradermacher Huginn-13B, HuginnV5.x GGUF/GPTQ/AWQ/exl2). No mlx-community entry, no GGUF of the recurrent model. Authors (HF discussion #10) say quantization is untested and a separate inference stack would need custom architecture support; llama.cpp discussion #11934 (Feb 2025) never produced an implementation. Only route on the Mac: transformers 4.57.x on MPS behind an OpenAI/Ollama-compatible shim (e.g. /Users/patrickli/Documents/vibe/looplm/serve/shim.py), served on a second port next to `ollama serve`.

**Minimal load + generate:**

```python
# huginn_local.py  --  Huginn-0125 on Apple Silicon (MPS, CPU fallback), transformers 4.57.x
# Setup (once):
#   uv venv -p 3.12 .venv-huginn
#   uv pip install --python .venv-huginn/bin/python 'transformers==4.57.6' 'torch==2.14.0' accelerate safetensors huggingface_hub
# Disk: the HF repo is 4 float32 safetensors shards = 15.6 GB (config torch_dtype=float32, tied embedding stored twice).
#   Route A (needs ~16 GB free): let from_pretrained download into ~/.cache/huggingface (cast to bf16 at load, RAM ~7.5 GB).
#   Route B (needs ~12 GB free at peak, ends at ~7.8 GB on disk): run `python huginn_local.py convert` once; it downloads
#   one fp32 shard at a time, rewrites it as bf16 into ./huginn-0125-bf16 and deletes the fp32 blob, then loads from that dir.
import os, sys, time, json, shutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

REPO = "tomg-group-umd/huginn-0125"
LOCAL_BF16 = "huginn-0125-bf16"

def convert_to_bf16_streaming(out_dir=LOCAL_BF16):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    # small files (config, tokenizer, remote code, index) -> out_dir
    snap = snapshot_download(REPO, allow_patterns=["*.json", "*.py", "*.md", ".gitattributes"])
    for f in os.listdir(snap):
        if not f.endswith(".safetensors"): shutil.copy(os.path.join(snap, f), out_dir)
    idx = json.load(open(os.path.join(out_dir, "model.safetensors.index.json")))
    total = 0
    for shard in sorted(set(idx["weight_map"].values())):
        if os.path.exists(os.path.join(out_dir, shard)): continue
        p = hf_hub_download(REPO, shard)                       # ~4.8 GB fp32 shard
        t = load_file(p)
        t = {k: (v.to(torch.bfloat16) if v.is_floating_point() and k != "freqs_cis" else v) for k, v in t.items()}
        save_file(t, os.path.join(out_dir, shard), metadata={"format": "pt"})
        total += sum(v.numel() * v.element_size() for v in t.values())
        os.remove(os.path.realpath(p)); os.remove(p) if os.path.lexists(p) else None   # free the fp32 blob + symlink
        print("converted", shard, flush=True)
    idx["metadata"]["total_size"] = total
    json.dump(idx, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

def load(src):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(src)                     # tokenizer.json/tokenizer_config.json ship with the repo
    model = AutoModelForCausalLM.from_pretrained(
        src, trust_remote_code=True, dtype=torch.bfloat16,       # bf16: what the authors trained/evaluated in
        device_map={"": device},                                 # load straight onto MPS (avoids a CPU+MPS double copy)
        # mean_recurrence=32,                                    # optional: default num_steps when a call omits num_steps
    ).eval()
    return tok, model, device

def chat_prompt(tok, user, system="You are a helpful assistant."):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # -> '<|begin_text|><|begin_header|>system<|end_header|>\n\n...<|end_turn|><|begin_header|>user<|end_header|>\n\n...<|end_turn|><|begin_header|>Huginn<|end_header|>\n\n'
    return tok.encode(text, return_tensors="pt", add_special_tokens=False)   # template already adds <|begin_text|>

def gen_config(max_new_tokens=256):
    # greedy, as in the model card; eos 65505=<|end_text|>, 65508=<|end_turn|>, pad 65509
    return GenerationConfig(max_new_tokens=max_new_tokens, stop_strings=["<|end_text|>", "<|end_turn|>"],
                            use_cache=True, do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None,
                            return_dict_in_generate=True, eos_token_id=[65505, 65508], bos_token_id=65504, pad_token_id=65509)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        convert_to_bf16_streaming(); sys.exit()
    src = LOCAL_BF16 if os.path.isdir(LOCAL_BF16) else REPO
    tok, model, device = load(src)
    ids = chat_prompt(tok, "Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell in total? Think step by step.").to(device)
    cfg = gen_config(128)

    # 1) Fixed recurrence depth: num_steps is a per-call kwarg (NOT a GenerationConfig field). Try 4 / 8 / 16 / 32 / 64.
    for r in (8, 32):
        t0 = time.time()
        out = model.generate(ids, cfg, num_steps=r, tokenizer=tok)          # HF generate path + HuginnDynamicCache
        n = out.sequences.shape[1] - ids.shape[1]
        print(f"\n[num_steps={r}] {n} tok in {time.time()-t0:.1f}s ({n/(time.time()-t0):.2f} tok/s)\n", tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True))

    # 2) Per-token adaptive compute (zero-shot early exit). NOTE: this path ignores GenerationConfig.max_new_tokens;
    #    pass max_new_tokens= as a kwarg. Paper threshold for KL is 5e-4 (code's "auto" = 1e-3).
    out = model.generate(ids, cfg, num_steps=64, tokenizer=tok, max_new_tokens=128,
                         criterion="kl", exit_threshold=5e-4, cache_lookup_strategy="latest-m4")
    print("\n[adaptive kl]", tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True))

    # 3) Memory-saving KV cache (keeps only 16 recurrent-step caches) and/or warm-start latent state between tokens
    out = model.generate(ids, cfg, num_steps=32, tokenizer=tok, max_new_tokens=128,
                         continuous_compute=True, cache_lookup_strategy="compress-s16")
    print("\n[continuous_compute + compress-s16]", tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True))

    # 4) Plain forward: logits + final latent state at a chosen depth
    with torch.no_grad():
        o = model(ids, num_steps=16)          # omit num_steps -> config.mean_recurrence (32) in eval mode
    print("\nlogits", tuple(o.logits.shape), o.logits.dtype, "latents", tuple(o.latent_states.shape))
```

**Gotchas:**

- transformers must be <5 (pin 4.57.6): on 5.0.0 and 5.16.1 loading fails in tie_weights with AttributeError 'list' object has no attribute 'keys' because raven_modeling_minimal.py sets _tied_weights_keys = ['lm_head.weight'] (list); 4.56.2/4.57.6 verified working. The HF card itself warns the HuginnDynamicCache auto-injection 'may break with huggingface updates'. Modeling file last changed 2025-07-29; config says transformers_version 4.44.2, generation_config 4.53.3.
- Disk: repo is 15.65 GB of float32 shards (4,771,970,936 + 4,744,780,096 + 4,744,737,616 + 1,384,120,448 bytes) including a duplicated tied embedding (model card: 'stored on HF with 2 copies of the tied embedding'); df on this Mac shows 4.1 GB free -> the download cannot start. Free ~16 GB for the plain route, or ~12 GB peak for the shard-by-shard bf16 conversion in the snippet (final ~7.8 GB on disk).
- RAM: 3.565B params = ~7.1 GB bf16 weights + KV cache. Load with device_map={'':'mps'} (verified with the remote code + accelerate 1.14) so weights land on MPS directly; .to('mps') after a CPU load can transiently double memory on a 16 GB machine.
- KV cache is per recurrent step: (4 + 4*num_steps) layer caches x 2 x 5280 x 2 bytes = ~2.8 MB/token at r=32 (0.75 MB/token at r=8). An 8-shot GSM8K CoT prompt (~1.3k tokens) + 256 generated tokens at r=32 is ~4.5 GB of cache on top of 7.1 GB weights -> near the 16 GB ceiling. Use cache_lookup_strategy='compress-s16' (caps recurrent caches at 16 -> ~1.4 MB/token), fewer shots, or lower num_steps.
- Speed (estimate, NOT measured with real weights; nobody has published Apple-Silicon numbers): decode is bandwidth-bound and each recurrent step re-reads the 1.5B-param (3 GB bf16) block, so at r=32 ~100 GB/token -> roughly 1 tok/s on M4 (~120 GB/s), ~4 tok/s at r=8. GitHub issue #33 reports ~90 s per 200 tokens at default depth even on an A800. Full GSM8K (1319 items x ~256 tokens) at r=32 is on the order of days; use --limit / subsample.
- GenerationConfig.max_new_tokens is IGNORED by the custom paths generate_with_adaptive_compute and generate_minimal (any call with criterion=, exit_threshold=, exit_evaluator=, or continuous_compute=): _prep_generate_args only reads a max_new_tokens= kwarg or generation_config.max_length. With the model card's config and a prompt longer than max_length you silently get 0 new tokens (verified). Pass max_new_tokens=N as a kwarg (verified) or set max_length.
- The model-card call generate_with_adaptive_compute(..., cache_kwargs={'lookup_strategy': ...}) is stale: the current signature takes cache_lookup_strategy='...' (cache_kwargs is not a parameter). Also cache_implementation='static' (HuginnStaticCache, which hardcodes bf16) fails on transformers 4.57.6 with 'You should provide exactly one of layers or layer_class_to_replicate to initialize a Cache' -> leave the default dynamic cache.
- forward() hard-codes prepared_attn_mask = None (line 674, compile_mask call commented out), so left-padded batches attend to pad tokens and positions shift: batched vs single-prompt logits differ (verified diff on a padded prompt). For faithful eval numbers use batch_size=1 (the authors used batch_size=auto on GPUs; small discrepancies are expected either way).
- MPS-specific: torch.autocast('cuda', enabled=False) in the rotary code only emits warnings on MPS; RMSNorm autocast uses x.device.type so it is fine. SDPA is the normal attention path; flex_attention (torch.nn.attention.flex_attention) is only used in embed_inputs()/adaptive-compute prefill when pads are present. On torch 2.14 a padded adaptive-compute batch ran on MPS in the tiny test (likely the non-compiled fallback), but flex_attention on MPS is not something the authors test — keep batch_size=1 to stay on SDPA.
- bf16 on MPS works (torch 2.14) and is what the authors evaluated in ('All benchmarks were evaluated in pure bfloat16'); do not use fp16 (no author guidance, risk of overflow across 32+ iterations) and fp32 would be 14 GB of weights. HF discussion #16 (2026-09-05) documents that bf16 vs fp32 readout changes which step the KL exit fires ~32% of the time at K=64 with no measurable downstream token change, and that the code's 'auto' KL threshold is 1e-3 whereas paper/README say 5e-4 -> pass exit_threshold=5e-4 explicitly.
- Tokenizer: repo ships tokenizer.json + tokenizer_config.json (PreTrainedTokenizerFast, chat_template with roles system/user/Huginn; assistant is aliased to Huginn). apply_chat_template already prepends <|begin_text|> (65504), so encode with add_special_tokens=False as the model card does; for raw prompts use add_special_tokens=True. Stop on <|end_turn|> (65508) as well as <|end_text|> (65505) — the repo's generation_config.json lists eos_token_id [65505, 65508]. Pad is <|pad|> 65509; model card says the model was not post-trained, so expect base-model-style verbosity.
- Batched adaptive compute cannot be used to measure per-token step savings (docstring: 'For batches, on each token, we iterate until the entire batch finishes'). The vLLM plugin in the GitHub repo is CUDA-oriented (GPU memory utilization / tensor parallel; 'everything token-level adaptive is unimplemented') — not a Mac route.
- Ollama integration: no GGUF/MLX/Ollama build exists (authors, discussion #10: quantization untested, 'you'd have to write support for this custom architecture'; llama.cpp discussion #11934 has no implementation). Run it behind the existing OpenAI/Ollama-compatible shim (/Users/patrickli/Documents/vibe/looplm/serve/shim.py) in .venv-tf4 with a backend that forwards num_steps (and optional criterion/exit_threshold/cache_lookup_strategy) as generate kwargs; note the shim's generic 'hf' backend sets a config field, which does NOT change Huginn's depth unless the field is mean_recurrence and num_steps is omitted.
- Base model: trained from scratch (800B tokens on Frontier MI250X); it is NOT Qwen-derived, so it does not satisfy the 'built on Qwen' wish. vladotpad/looped-qwen3-huginn-fineweb is a 9M-parameter toy (best.pt, d=384) — not a usable Qwen-based looped LLM.

**Official eval recipe:** Harness: EleutherAI lm-evaluation-harness (`hf` model backend; version not stated by the authors — 0.4.13 is current and ran the recipe on MPS here), pure bfloat16, recurrence fixed at 32 via model_args mean_recurrence=32 (config override picked up because the model omits num_steps in eval mode). Code tasks in the paper were run with bigcode-evaluation-harness (paper: 'except for the code tasks, which are executed using bigcode'); the README also gives an lm-eval humaneval_instruct command. Exact commands from https://github.com/seal-rg/recurrent-pretraining README:
(1) HellaSwag 0-shot: `lm_eval --model hf --model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32 --tasks hellaswag --batch_size=auto --num_fewshot=0` (Table 1 'lm-eval-harness tasks zero-shot', normalized accuracy when provided: ARC-E 69.91, ARC-C 38.23, HellaSwag 65.21, MMLU 31.38, OBQA 38.80, PiQA 76.22, SciQ 93.50 at r=32).
(2) GSM8K CoT ('w/ sys. prompt' row): `lm_eval --model hf --model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32 --tasks gsm8k_cot --batch_size=auto --apply_chat_template=True --fewshot_as_multiturn --system_instruction="You are a helpful assistant that can assist users with mathematical reasoning."` — gsm8k_cot = 8-shot chain-of-thought (Wei et al. exemplars, 'Q: ...\nA: ... The answer is N.'), each shot a separate user/Huginn turn under the chat template, greedy (do_sample=false), until ['Q:', '</s>', '<|im_end|>'] plus EOS, HFLM default max_gen_toks=256, filters strict-match regex 'The answer is (\\-?[0-9\\.\\,]+).' and flexible-extract '(-?[$0-9.,]{2,})|(-?[0-9]+)'. Paper Table 2 (arXiv 2502.05171v2, strict/flexible): Ours w/ sys. prompt r=32: GSM8K 24.87/38.13, GSM8K CoT 34.80/42.08, Minerva MATH 11.24, MathQA 27.97; w/o sys. prompt r=32: 28.05/28.20, 32.60/34.57, 12.58, 26.60; OLMo-2-1124-7B: 66.72/66.79, 61.94/66.19, 19.08, 37.59. The 'w/o sys. prompt' row drops --system_instruction. MATH uses the Minerva rules (lm-eval minerva_math). Sweeping depth = repeat with mean_recurrence=4/8/16/32/64.
(3) HumanEval: `HF_ALLOW_CODE_EVAL=1 accelerate launch -m lm_eval --model hf --model_args pretrained=tomg-group-umd/huginn-0125,mean_recurrence=32,trust_remote_code=True,dtype=bfloat16 --tasks humaneval_instruct --batch_size=1 --num_fewshot=0 --output_path=outputs/heval --confirm_run_unsafe_code --apply_chat_template=True --gen_kwargs=do_sample=True,temperature=0.2,top_p=0.95` (paper: 23.17 pass@1 at r=32).
Adaptive-compute evals use evaluate_raven/misc_benchmark_variants/hf_eval_adaptive_compute.py (HuginnWrapper subclassing lm_eval HFLM; criterion/exit_threshold/lookup_strategy as constructor args, num_steps via gen_kwargs='num_steps=N'); that folder's README states 'These benchmark variants were not used in the paper! Use the normal benchmarks to replicate numbers from the paper!'. Paper's KL exit threshold: 5e-4.
Mac adaptation (verified to run end-to-end on MPS with a tiny model): add `--device mps --batch_size 1` (batch_size=auto/padded batches are unmasked in forward(), see gotchas), `--limit N` or a subsample, `--log_samples --output_path results/`, and drop `accelerate launch` (run `lm_eval` directly). An alternative for apples-to-apples with the Ollama baselines is the project's own eval/run_eval.py against the shim, which already implements the same 8-shot gsm8k_cot exemplars and strict 'The answer is' matching.

**Python:** Do NOT use the system Python 3.9.6 (/usr/bin/python3): torch 2.14.0 requires Python >=3.10 (PyPI requires_python). Homebrew python3 on this Mac is 3.14.7; avoid it too (newest wheels lag). Use the uv-managed CPython 3.12.13 that is already installed (~/.local/share/uv/python/cpython-3.12-macos-aarch64-none): `uv venv -p 3.12 .venv-huginn && uv pip install --python .venv-huginn/bin/python 'transformers==4.57.6' 'torch==2.14.0' accelerate safetensors huggingface_hub lm_eval`. The project already has /Users/patrickli/Documents/vibe/looplm/.venv-tf4 with exactly this stack (3.12.13 / torch 2.14.0 / transformers 4.57.6 / lm_eval 0.4.13 / accelerate 1.14.0 / fastapi / uvicorn), which is what every check in this report was run in. Keep the transformers-5 venv (.venv) for other models; huginn cannot load there.


---

### Recurrent-Llama-3.2-train-recurrence-16 (Retrofitted Recurrence / Huginn-Raven-style depth-recurrent Llama-3.2-1B; ~1.385B params; prelude 4 / recurrent 6 / coda 4 layers, source layers 4-5 dropped)

**Repo:** smcleish/Recurrent-Llama-3.2-train-recurrence-16  
**Download:** 5.56 GB · **RAM:** ~4 GB

**Loop control:** num_steps (kwarg to model(input_ids, num_steps=N) or model.generate(input_ids, num_steps=N, ...); int or torch.tensor([n_no_grad, n_with_grad]); must NOT be placed in GenerationConfig). Default when omitted: config.mean_recurrence = 16 (randomized_iteration_sampler returns mean_recurrence in eval mode). For lm_eval the same knob is set via --model_args mean_recurrence=N (a config override passed through from_pretrained). Verified on this Mac that changing num_steps changes the logits and that num_steps=8 generation with the custom KV cache equals use_cache=False generation.

**Thinking mode:** None. No chat template, no <think> tags, not instruction-tuned (card: "it is _not_ an instruction model"). The only "thinking" knob is latent depth recurrence via num_steps (more loops = more test-time compute; Table 5 shows GSM8K 13.1 -> 31.9 -> 42.2 -> 45.3 -> 45.4 -> 45.3 for num_steps 1/2/4/8/16/32). The Huginn-0125 card's claim that the model "natively understands chat templates" refers to Huginn's own tokenizer, not this Llama-3 tokenizer, whose tokenizer_config.json has no chat_template (apply_chat_template would fail). The adaptive-compute early-exit mode (generate(..., criterion="kl"/"argmax-stability"/...)) is BROKEN for this checkpoint: reproduced here on the Llama variant -> AttributeError: 'RavenForCausalLM' object has no attribute 'freqs_cis' (same as GitHub issue #1 for the OLMo sibling).

**Dependencies:**

- python==3.11 (via `uv python install 3.11 && uv venv -p 3.11 .venv`; verified 3.11.15 on this Mac)
- torch==2.8.0 (VERIFIED on this M4 with MPS; any torch>=2.5 is required because the modeling file imports torch.nn.attention.flex_attention at module top and calls SDPA with enable_gqa=True; macOS arm64 wheel, no CUDA)
- transformers==4.51.0 (the version pinned by the authors' GitHub README; VERIFIED on this Mac. 4.53.1 (the version stamped in config.json) and 4.57.1 ALSO passed the same forward+cached-generate smoke test here; transformers==5.16.1 (current default `pip install transformers`) FAILS at config load with AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings' -- do NOT use 5.x)
- accelerate>=0.26.0 (needed by lm_eval's hf backend; optional for plain inference)
- safetensors
- huggingface_hub
- lm_eval[math]==0.4.13 (only for replicating the paper's evals; pulls sympy, antlr4-python3-runtime==4.11, math_verify; requires Python>=3.10)
- NOT needed: sentencepiece (tokenizer.json is a fast Llama-3 tokenizer), bitsandbytes/flash-attn (CUDA-only), mlx, llama.cpp

**Quantized / alternative runtimes:** none found. HF API searches (search=Recurrent-Llama, 'Recurrent-Llama-3.2 GGUF', 'Recurrent-Llama-3.2 mlx', huginn_raven, retrofitting-recurrence, author=smcleish, filter=gguf/mlx + huginn) return only the official smcleish safetensors checkpoints (train-recurrence-4/8/16/32, -untrained, -2-4-2-untrained, Llama-3.2-non-recurrent-posttrained) and irafm-llm/Recurrent-Llama-3.2-1B (an independent bf16 safetensors retrofit, 2.77 GB, license llama3.2, updated 2026-06-29 -- NOT a conversion of the paper's checkpoint, so not a replication target). All GGUF 'Huginn' hits (TheBloke/Huginn-13B-GGUF etc.) are unrelated Huginn-13B merges. The huginn_raven architecture has no llama.cpp or mlx-lm implementation, so no Ollama/MLX run command exists; use the transformers+MPS path above.

**Minimal load + generate:**

```python
# Recurrent-Llama-3.2-train-recurrence-16 on Apple Silicon (MPS, CPU fallback)
# Verified env on an M4 MacBook: Python 3.11.15, torch==2.8.0, transformers==4.51.0
#   uv python install 3.11 && uv venv -p 3.11 .venv && source .venv/bin/activate
#   uv pip install "torch==2.8.0" "transformers==4.51.0" accelerate safetensors huggingface_hub
# Disk: first run downloads 5.54 GB of fp32 safetensors (+17 MB tokenizer) into ~/.cache/huggingface/hub.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "smcleish/Recurrent-Llama-3.2-train-recurrence-16"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# bf16 = ~2.8 GB weights (fast, what Huginn recommends). The paper's evals were run in float32
# (shells/eval.sh: dtype="float32"); use torch.float32 (~5.6 GB) for the most faithful replication.
dtype = torch.bfloat16

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    torch_dtype=dtype,          # transformers>=4.56 prefers `dtype=`; `torch_dtype` still works on 4.51-4.57
    trust_remote_code=True,     # custom code ships in-repo: raven_config_minimal.py / raven_modeling_minimal.py
).to(device).eval()

# --- REQUIRED FIX: config.json / generation_config.json carry Huginn-0125 token ids, not Llama-3 ids.
# bos 65504 = " skating", eos 65505 = " creek", pad 65509 = " mandated" in THIS tokenizer, so a stock
# generate() stops on the word "creek" and never on <|end_of_text|> (128001). Override them:
BOS, EOS = tok.bos_token_id, tok.eos_token_id          # 128000 <|begin_of_text|>, 128001 <|end_of_text|>
model.generation_config.bos_token_id = BOS
model.generation_config.eos_token_id = EOS
model.generation_config.pad_token_id = EOS
model.config.bos_token_id, model.config.eos_token_id, model.config.pad_token_id = BOS, EOS, EOS

# --- Prompting / "chat template": there is NONE. tokenizer_config.json has no chat_template and the card
# says "it is _not_ an instruction model". Use the paper's GSM8K format (eval_yamls/gsm8k-cot-sean.yaml):
#   "Q: {question}\n\nA: Let's think step by step."   with 1 in-context example (see official_eval_recipe).
prompt = ("Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
          "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n\n"
          "A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some "
          "more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n"
          "Q: Janet has 3 apples and buys 5 more, then eats 2. How many apples does she have?\n\n"
          "A: Let's think step by step.")
# add_special_tokens=True prepends <|begin_of_text|>; the authors evaluate with add_bos_token=True.
ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

# --- Loop / recurrence count: the `num_steps` kwarg, passed PER CALL to forward() or generate().
# It must NOT be put inside a GenerationConfig. If omitted, the model uses config.mean_recurrence (=16).
# Paper evaluates test recurrence 1, 2, 4, 8, 16, 32 (Table 5). No hard cap in code.
NUM_STEPS = 16

with torch.no_grad():
    out = model.generate(
        ids,
        max_new_tokens=256,
        do_sample=False,                 # greedy, as in the paper's harness config (do_sample: false)
        num_steps=NUM_STEPS,
        eos_token_id=EOS, pad_token_id=EOS,
        stop_strings=["Q:"], tokenizer=tok,   # the harness stops on "Q:" as well
    )
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

# Single forward pass with an explicit recurrence (int, or torch.tensor([steps_no_grad, steps_with_grad])):
with torch.no_grad():
    logits_r32 = model(ids, num_steps=32).logits          # (1, seq, 128256)
    logits_r4  = model(ids, num_steps=torch.tensor([4, 0])).logits

# Optional: once downloaded, keep a 2.77 GB bf16 local copy and delete the 5.54 GB fp32 HF cache:
#   model.to(torch.bfloat16).save_pretrained("./recurrent-llama-r16-bf16"); tok.save_pretrained("./recurrent-llama-r16-bf16")
#   then copy raven_config_minimal.py + raven_modeling_minimal.py from the HF snapshot dir into that folder
#   and fix bos/eos/pad ids in its config.json + generation_config.json; load with the same code, REPO="./recurrent-llama-r16-bf16".
```

**Gotchas:**

- DISK: the repo is fp32 only (2 safetensors shards, 5,540,913,152 B = 5.54 GB + 17 MB tokenizer.json). The Mac currently has ~2.9-3.5 GB free, so the user MUST free >=6 GB before the first from_pretrained. Afterwards save a bf16 copy (2.77 GB) with save_pretrained and delete the HF cache blob to reclaim ~2.8 GB (see snippet). There is no bf16/quantized upload of the official checkpoint.
- WRONG TOKEN IDS IN CONFIG (real bug, verified by reading tokenizer.json): config.json and generation_config.json set bos_token_id=65504, eos_token_id=65505, pad_token_id=65509 -- these are Huginn-0125 tokenizer ids; in this Llama-3 tokenizer they are the ordinary tokens ' skating', ' creek', ' mandated'. Stock generate() therefore stops on the word 'creek' and never on <|end_of_text|> (128001). Override generation_config/config ids (snippet) or pass eos_token_id=128001. The custom generate_minimal/adaptive paths also hard-code stop ids {65504,65505,65508}. Note the authors' lm_eval runs did not fix this; the harness relies on 'Q:' stop strings, so the effect on their numbers is negligible but non-zero.
- TRANSFORMERS VERSION: pin transformers==4.51.0 (authors' README: conversion code built for 4.51.0 'due to a KV-Cache breaking change in future versions'). Verified on this Mac: 4.51.0, 4.53.1 and 4.57.1 all pass forward + cached generate; transformers 5.16.1 (what `pip install transformers` gives today) FAILS at config load ('PreTrainedConfig' object has no attribute 'max_position_embeddings'). GitHub issue #1 reports >=4.54 removed DynamicCache._seen_tokens / changed update() -- HuginnDynamicCache re-implements both, which is why 4.57.1 still worked here; 5.x does not.
- TORCH: needs torch>=2.5 (module-level `from torch.nn.attention.flex_attention import ...` and SDPA enable_gqa=True with 32 query heads / 8 KV heads). Verified torch 2.8.0 macOS arm64: the flex_attention import succeeds, GQA SDPA runs on MPS in fp32/bf16/fp16, and MPS fp32 logits match CPU fp32 to 1e-4. flex_attention is only executed when a BlockMask is passed, which forward() never does (prepared_attn_mask is hard-coded to None).
- ATTENTION MASK IS IGNORED: forward() sets prepared_attn_mask = None (compile_mask commented out), so `attention_mask` / left-padding has no effect -- padded batch members attend to pad tokens. Batched generation (the authors used --batch_size 32) is therefore slightly lossy; for the cleanest replication on the Mac use --batch_size 1 (also avoids MPS memory spikes). Expect the HF warning 'The attention mask is not set...' -- harmless.
- MPS DTYPES: bf16, fp16 and fp32 all ran on MPS with torch 2.8.0. RMSNorm wraps its math in torch.autocast(enabled=False, device_type='mps'), which worked. Do not enable torch.autocast on MPS around SDPA (pytorch issue #141774 reports dtype-mismatch errors under MPS autocast). HuginnStaticCache hard-codes dtype=torch.bfloat16 -- do not pass cache_implementation='static'; the default HuginnDynamicCache is dtype-agnostic and verified. First MPS forward is slow (Metal kernel compile, ~3 s even for a tiny model) -- warm up before timing.
- ADAPTIVE COMPUTE / EARLY EXIT IS BROKEN for this checkpoint: generate(..., criterion=..., exit_threshold=...) -> generate_with_adaptive_compute -> embed_inputs raises AttributeError 'freqs_cis' (reproduced here; same as GitHub issue #1 for OLMo, still open, no maintainer reply). Only plain forward()/generate(num_steps=N) are supported; the README warns 'We only tested ... forward(), generate()'. `continuous_compute` (generate_minimal) is likewise untested.
- CONTEXT: trained at block_size 1024 (config block_size=1024; paper: context length 1024). RoPE metadata allows 131072 but the authors evaluate with max_length=1024; keep prompts (few-shot + generation) under 1024 tokens.
- RECURRENCE COST: each num_steps adds 6 transformer layers; r=16 -> 4+6*16+4 = 104 effective layers per token, r=32 -> 200. Wall-clock scales roughly linearly with num_steps; on M4/MPS expect single-digit tok/s at r=16 in bf16 (NOT measured -- estimate only). A full GSM8K test set (1319 items, ~150-250 generated tokens each) at r=32 will take hours on this laptop.
- PARAMS/LICENSE: ~1.385B params (5.54 GB / 4 B), untied embeddings (tie_embeddings=false). Repo tagged apache-2.0 but derives from meta-llama/Llama-3.2-1B (Llama 3.2 Community License; the independent irafm-llm retrofit lists license llama3.2) -- treat as Llama-licensed. The tokenizer/model files are NOT gated (api: gated=false), so no HF token is needed.
- OLLAMA: the huginn_raven architecture is not supported by llama.cpp/Ollama/mlx-lm and no GGUF/MLX conversion exists (HF API searches for 'Recurrent-Llama', 'huginn_raven', gguf/mlx filters return nothing relevant; 'Huginn' GGUF hits are unrelated Huginn-13B merges). To pair with Ollama, run this model behind a small OpenAI-compatible HTTP server (e.g. FastAPI /v1/completions wrapping the snippet's generate(), exposing num_steps as an extra request field) on another port next to Ollama's baseline models (e.g. `ollama pull llama3.2:1b` for the base). `transformers serve` is not available in 4.51.0.
- COMPARISON MODELS: the paper's control is smcleish/Llama-3.2-non-recurrent-posttrained (same 50B-token Nemotron-CC-Math post-training, standard Llama arch, loadable without custom code; if lm_eval errors with 'got multiple values for keyword argument tie_word_embeddings', delete that key from its config.json per the README). Sibling looped checkpoints: train-recurrence-4/8/32 (each also 5.54 GB fp32). Stock meta-llama/Llama-3.2-1B is gated on HF (needs license acceptance + token) and scores only 4.9 GSM8K / 4.3 MATH in the authors' 1-shot setup (Table 5).
- CORRECTION to earlier notes: the '~62% GSM8K at r=32' figure is not the Llama number; Table 5 gives 45.3 GSM8K / 28.4 MATH at test recurrence 32 for train-recurrence-16 (45.4 / 28.6 at r=16), vs 37.1 / 27.4 for the non-recurrent post-trained control.
- NOT CONFIRMED: an actual load of the real 5.54 GB weights on this Mac (disk too small during this task; only the custom code path was exercised with a tiny random-init config on MPS), real tokens/s, and whether bf16 vs fp32 changes the benchmark numbers on MPS. Also unconfirmed: any third-party report of this model on Apple Silicon (none found).

**Official eval recipe:** Harness: EleutherAI lm-evaluation-harness (`lm_eval --model hf`), per https://github.com/mcleish7/retrofitting-recurrence (README 'Evals' + shells/eval.sh). Exact official invocation for math (from shells/eval.sh, GPU flags removed): `lm_eval --model hf --model_args pretrained=<model>,mean_recurrence=<R>,add_bos_token=True,dtype="float32",trust_remote_code=True,max_length=1024 --tasks gsm8k_cot_sean,minerva_math --num_fewshot=1 --batch_size 32` (for the non-recurrent control simply drop mean_recurrence=...). General tasks: `--tasks lambada_openai,hellaswag,arc_easy,arc_challenge,mmlu,openbookqa,piqa,social_iqa,winogrande,asdiv --batch_size auto` with the same model_args minus max_length. Recurrence is set through mean_recurrence=<R> in model_args (a config override forwarded to from_pretrained by lm_eval's HFLM; num_steps is then None so the model runs exactly R loops in eval mode); the paper reports R = 1, 2, 4, 8, 16, 32 (Table 5). Prompt: custom task gsm8k_cot_sean = the harness gsm8k_cot YAML with ' Let\'s think step by step.' appended: doc_to_text "Q: {{question}}\n\nA: Let's think step by step.", fewshot_config sampler first_n over 8 hand-written CoT exemplars, but shells/eval.sh overrides with --num_fewshot=1 and the paper states 'we use a single shot example in context' (so exactly ONE exemplar: the 15-trees example). Decoding: generation_kwargs do_sample: false (greedy), until ['Q:', '</s>', '<|im_end|>'], max_length=1024 total (no explicit max_gen_toks -> harness default 256). Metric: the paper reports GSM8K 'flexible-extract' exact_match (regex (-?[$0-9.,]{2,})|(-?[0-9]+), group_select -1) rather than strict-match; MATH uses minerva_math with Math-Verify (lm_eval[math]). YAML file: https://raw.githubusercontent.com/mcleish7/retrofitting-recurrence/master/eval_yamls/gsm8k-cot-sean.yaml -- copy it next to lm_eval/tasks/gsm8k/gsm8k.yaml in the installed package or pass `--include_path <dir containing it>`. Mac adaptation: `--device mps --batch_size 1` (lm_eval supports device mps with torch>=2.1; attention_mask is ignored by the model so batch_size 1 is safest), dtype float32 to match the authors (bf16 halves memory; effect on scores unmeasured). Replication targets (arXiv 2511.07384 Table 5, 'Final step accuracy', 1-shot, flexible extract), Llama 4,6,4 Train Recurrence=16 at test recurrence 1/2/4/8/16/32: GSM8K 13.1/31.9/42.2/45.3/45.4/45.3, MATH 16.5/26.1/28.8/28.7/28.6/28.4; Arc-E 55.2/60.6/61.8/61.9/61.9/61.9, HellaSwag 41.7/44.6/45.9/45.8/45.8/45.8, MMLU 31.5/34.8/36.9/37.0/37.0/37.0. Llama Non-Recurrent (post-trained control): Arc-E 62.6, Arc-C 38.2, HS 45.8, WG 57.1, MMLU 38.7, PIQA 68.4, OBQA 33.4, GSM8K 37.1, MATH 27.4. Stock Llama-3.2-1B from HF in the same setup: GSM8K 4.9, MATH 4.3. Training context for interpretation: ~50B tokens of Nemotron-CC-Math-v1-4plus, context 1024, 1-sqrt recurrence curriculum for first 75% then constant, Muon optimizer, bf16 mixed precision, prelude layers [0,1,2,3], recurrent block source layers [6..11], coda [12..15] (Table 9). Offline val-loss across recurrences: multi_recurence_eval.py (`model(input_ids, num_steps=torch.tensor([num_rec,0]))`, fp32).

**Python:** System Python 3.9 is NOT sufficient: lm_eval 0.4.13 requires Python>=3.10, torch 2.8 wheels target 3.9-3.13 but the authors developed on 3.11 (GitHub README: "We developed in Python 3.11"). Use uv: `uv python install 3.11 && uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python "torch==2.8.0" "transformers==4.51.0" accelerate safetensors huggingface_hub "lm_eval[math]==0.4.13"`. This exact combo (3.11.15 + torch 2.8.0 + transformers 4.51.0) was installed and exercised on this M4 during this task (tiny random-init RavenConfig on MPS in fp32/bf16/fp16: forward with num_steps, generate() through HuginnDynamicCache producing identical tokens to use_cache=False, batched left-padded generate). The venv (~0.5 GB) and uv cache (~0.5 GB) were deleted afterwards to give the disk back.


---

### Nanbeige4.2-3B-Base (NanbeigeForCausalLM, model_type "nanbeige"; 22 shared layers x num_loops=2 = 44 effective layers; 4.17B total params BF16, ~3B non-embedding; vocab 166144; Apache-2.0; trained from scratch, not Qwen-derived)

**Repo:** Nanbeige/Nanbeige4.2-3B-Base  
**Download:** 8.35 GB · **RAM:** ~11 GB

**Loop control:** config attribute `num_loops` on NanbeigeConfig (config.json: num_loops=2, loop_loss_weights=[], skip_loop_final_norm=false, enable_double_loop_split default False). It is NOT a generate()/forward() kwarg. modeling_nanbeige.py line 2018 `_get_num_loops()`: returns 1 if enable_double_loop_split, else len(loop_loss_weights)+1 if loop_loss_weights non-empty, else config.num_loops; it is evaluated on every forward, so `model.config.num_loops = N` before each generate() call changes recurrence depth per call (verified on MPS: N=1,2,3 give different logits, N=3 generates fine; KV cache gets N*22 layer slots via _get_loop_cache_layer_idx = layer_idx + loop_idx*22; each pass ends with the final RMSNorm unless skip_loop_final_norm). Trained/published value is 2 only ('two-pass configuration provides the most favorable trade-off', report sec 2.1); StaticCache is rejected when num_loops>1 (line 2143). In GGUF the value is stored as metadata key `nanbeige.num_loops` (llama.cpp llama-arch.cpp LLM_KV_NUM_LOOPS "%s.num_loops", plus nanbeige.skip_loop_final_norm); `llama-cli --override-kv nanbeige.num_loops=int:1` should override it at load time but this was NOT tested. In stock mlx-lm (main branch nanbeige.py) it is ModelArgs.num_loops / effective_num_loops, also config-level.

**Thinking mode:** -Base: none (no <think> tokens in its tokenizer, vocab 166100 ends before the instruct-only special tokens; base chat_template is just '<s>' + content). Instruct Nanbeige/Nanbeige4.2-3B: thinking is ON by default - the chat template appends '<think>\n' after '<|im_start|>assistant\n'; pass `enable_thinking=False` to apply_chat_template to prefill '<think>\n\n</think>\n\n' (no-think). Special ids: <|im_start|>=166100 (bos), <|im_end|>=166101 (eos), <think>=166103, </think>=166104, <tool_call>=166105. generation_config.json of instruct: do_sample=true, temperature=0.6, top_k=20, top_p=0.95.

**Dependencies:**

- torch==2.14.0 (any torch>=2.2 with MPS; 2.14.0 is what resolved and was tested)
- transformers==4.45.1 (the pin from the instruct card; DO NOT use transformers 5.x - the custom code calls DynamicCache.from_legacy_cache/to_legacy_cache, Cache.get_max_length and rope_scaling['type'], all removed/changed in 5.x per arXiv 2608.13987)
- tokenizers==0.20.3 (auto-resolved by transformers 4.45.1)
- sentencepiece==0.2.2 (required: the -Base repo ships only tokenizer.model, class LlamaTokenizer, no tokenizer.json)
- protobuf==7.36.1 (required by slow LlamaTokenizer / fast conversion)
- accelerate==1.14.0 (only needed if you use device_map='auto'; optional)
- safetensors==0.8.0
- huggingface-hub==0.36.2
- numpy==2.5.3

**Quantized / alternative runtimes:** GGUF (exists, fits the disk): mradermacher/Nanbeige4.2-3B-Base-GGUF (static: Q2_K 1.76 GB, Q3_K_M 2.17 GB, IQ4_XS 2.4 GB, Q4_K_S 2.5 GB, Q4_K_M 2.57 GB, Q5_K_M 2.99 GB, Q6_K 3.42 GB, Q8_0 4.43 GB, f16 8.34 GB) and mradermacher/Nanbeige4.2-3B-Base-i1-GGUF (imatrix: IQ3_S 2.0 GB, IQ4_XS 2.38 GB, Q4_K_M 2.57 GB ...). Needs llama.cpp with the NANBEIGE arch: merged to mainline 2026-07-27 (ggml-org/llama.cpp PR #25994; first tag b10159, 2026-07-28); the semver release v0.4.0 (2026-09-04) and the Homebrew formula llama.cpp 0.4.0 both contain LLM_ARCH_NANBEIGE (verified in the v0.4.0 tag's src/llama-arch.cpp). Run: `brew install llama.cpp` then OpenAI-compatible server alongside Ollama: `llama-server -hf mradermacher/Nanbeige4.2-3B-Base-GGUF:Q4_K_M -c 8192 -ngl 99 --port 8081` (downloads ~2.57 GB to ~/Library/Caches/llama.cpp); quick test: `llama-cli -hf mradermacher/Nanbeige4.2-3B-Base-GGUF:Q4_K_M -p "Question: What is 12*7?\nAnswer:" -n 64 --temp 0`; loop ablation (untested): add `--override-kv nanbeige.num_loops=int:1`. Ollama (stock, untested): `ollama run hf.co/mradermacher/Nanbeige4.2-3B-Base-GGUF:Q4_K_M`; official path is the Nanbeige/ollama fork branch nanbeige42 (commit 1cea47f3 2026-07-20 'support nanbeige4.2 model', built on llama.cpp b9888 + an MLX runner x/models/nanbeige). MLX: NO -Base conversion exists on the Hub (all MercuriusDream/vote-for-pedro/jishnuvenugopal/sahilchachra/WaveCut/cudo528 'Nanbeige4.2-3B-mlx-*' repos are the instruct model - verified bos_token_id 166100 / rope_theta 7e7 on MercuriusDream 4bit; mlx-community/Nanbeige4.2-3B-OptiQ-4bit is instruct and needs mlx-optiq>=0.4.6). Stock mlx-lm DOES have mlx_lm/models/nanbeige.py, but only on the main branch (added 2026-08-29, PR #1597; latest PyPI/GitHub release 0.31.3 is from 2026-04-22), so install `pip install git+https://github.com/ml-explore/mlx-lm` and convert yourself once disk allows: `python -m mlx_lm.convert --hf-path Nanbeige/Nanbeige4.2-3B-Base -q --q-bits 4 --mlx-path ./nanbeige42-base-mlx-4bit` (needs the 8.34 GB BF16 download + ~2.5 GB output), then `python -m mlx_lm.server --model ./nanbeige42-base-mlx-4bit --port 8082` (OpenAI-compatible). mlx-lm's nanbeige.py rejects enable_double_loop_split/loop_share_kv configs (not used by -Base).

**Minimal load + generate:**

```python
# Nanbeige4.2-3B-Base on Apple Silicon (MPS, CPU fallback)
# Tested env: Python 3.12.13, torch 2.14.0, transformers 4.45.1, sentencepiece 0.2.2, protobuf 7.36.1 (macOS 26.6, M4).
# NOTE: the modeling code path below was verified on MPS with the repo's own modeling_nanbeige.py using a tiny
# random-weight config (bf16+fp16, sdpa+eager, generate, batched generate, save_pretrained, loop override);
# the real 8.34 GB weights could not be downloaded on the 3.4 GB-free target disk, so free >=9 GB first.
#
#   uv venv --python 3.12 .venv && source .venv/bin/activate
#   uv pip install torch "transformers==4.45.1" sentencepiece protobuf accelerate
#   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # optional: lift the MPS allocator cap on 16 GB machines
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL = "Nanbeige/Nanbeige4.2-3B-Base"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32   # bf16 verified on MPS; torch.float16 also works

# README uses the slow SentencePiece tokenizer (use_fast=False). Keep it: the fast tokenizer tokenizes
# "Question" as a different id (19051 slow vs 13467 fast, verified) -> eval numbers would shift.
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False, trust_remote_code=True)   # add_bos_token=True -> "<s>" prepended

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cfg.num_loops = 2            # trained value (config.json). This is THE loop/recurrence knob; there is no generate() kwarg.
# (effective loops = 1 if cfg.enable_double_loop_split else len(cfg.loop_loss_weights)+1 if cfg.loop_loss_weights else cfg.num_loops)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, config=cfg, torch_dtype=dtype, trust_remote_code=True,
    attn_implementation="sdpa",      # "eager" also works; flash_attention_2 is CUDA-only
    low_cpu_mem_usage=True,
).to(device).eval()

def generate(prompt: str, num_loops: int = 2, max_new_tokens: int = 256, **gen_kw) -> str:
    # Loop count is read from model.config at EVERY forward (NanbeigeModel._get_num_loops), so it can be changed
    # per call. Set it BEFORE generate(), never mid-generation (KV cache has num_loops*22 layer slots).
    model.config.num_loops = num_loops
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,      # ALWAYS set max_new_tokens: no generation_config.json, config max_length=131072
            do_sample=False,                           # greedy for evals
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **gen_kw,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

# BASE model => no real chat format. tokenizer.chat_template is literally "<s>" + concatenated message contents,
# so apply_chat_template([{"role":"user","content":"hi"}]) == "<s>hi". Use raw few-shot prompts instead:
prompt = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\n"
    "Answer: In April she sold 48 clips. In May she sold 48/2 = 24 clips. Altogether 48 + 24 = 72. The answer is 72.\n\n"
    "Question: What is 12 * 7 + 5?\nAnswer:"
)
print("loops=2:", generate(prompt, num_loops=2, max_new_tokens=64))
print("loops=1:", generate(prompt, num_loops=1, max_new_tokens=64))   # ablation only: single pass is an untrained regime
print("loops=3:", generate(prompt, num_loops=3, max_new_tokens=64))   # runs, but no published results for >2

# Authors' sampling for the INSTRUCT model (Appendix B.1 of the report): temperature=0.6, top_p=0.95, top_k=20
# -> generate(prompt, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)

# ---- If you later load the INSTRUCT model Nanbeige/Nanbeige4.2-3B instead (same code, real chat template) ----
# messages = [{"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}]
# text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)  # False -> no-think
# enc = tok(text, add_special_tokens=False, return_tensors="pt").to(device)
# out = model.generate(**enc, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, eos_token_id=166101)
```

**Gotchas:**

- DISK: the -Base repo is a single BF16 model.safetensors of 8,339,624,768 bytes (8.34 GB) + ~3 MB tokenizer/code; the target has ~3.4 GB free, so the transformers path needs >=9 GB freed first (HF cache keeps one copy, no extra duplicate). Only Q4-class GGUFs (2.4-2.6 GB) fit right now.
- TRANSFORMERS VERSION: pin transformers==4.45.1 (instruct card). arXiv 2608.13987 shows the shipped modeling code breaks on transformers 5.8.1 with 5 bugs (zeroed inv_freq RoPE buffer, rope_scaling['type'] KeyError, DynamicCache.from_legacy_cache removed, MPS position_ids matmul SIGABRT, save_pretrained tied-weights failure). With 4.45.1 + torch 2.14.0 on this M4 none of these reproduced (inv_freq non-zero, direct forward, generate with attention_mask, batched left-padded generate, save_pretrained all OK). Bug #2 (rope_scaling['type']) cannot trigger on -Base anyway because config.json has rope_scaling=null. If you must use transformers 5.x, use the patched johnhalloran/Nanbeige4.2-3B-mps-fix modeling file (instruct-only checkpoint; patch is model-agnostic per its README but untested on -Base).
- RAM ON 16 GB: BF16 weights 8.34 GB resident + KV cache ~180 KB/token (2 x 44 effective layers x 8 KV heads x 128 dim x 2 B) + prefill logits: modeling_nanbeige.py line 2514 does logits.float() over the FULL prompt and the 166,144-token vocab (0.66 MB/token fp32 + 0.33 MB/token bf16 transient) and the model does NOT support num_logits_to_keep. A 2K-token prompt costs ~2 GB transient just for logits; a 4K prompt ~4 GB. Expect ~10-11 GB peak for 1-2K prompts; keep few-shot prompts short or patch line 2514 to apply lm_head only to the last position during generation. Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 if you hit the MPS allocator cap; close other apps. Batch>1 multiplies KV+logits.
- SPEED: arXiv 2608.13987 (M2 Max 32 GB, instruct, transformers) measured decode 15.5 tok/s at short context falling to 2.1 tok/s at ~5K context; naive prefill aborts by ~8K tokens (chunked prefill reaches ~11K). Expect similar or slower on M4 16 GB. The GGUF/llama.cpp Metal path is far faster and lighter; use transformers only for loop-count ablations and exact-tokenizer parity.
- TOKENIZER: only tokenizer.model (SentencePiece, LlamaTokenizer) + tokenizer_config.json + special_tokens_map.json ship; no tokenizer.json. Needs sentencepiece + protobuf. use_fast=False (README) and use_fast=True tokenize differently (verified: 'Question' -> 19051 slow vs 13467 fast with the legacy prefix-space behaviour); use slow for evals. add_bos_token=True prepends <s> (id 1); eos </s> (id 2); pad <unk> (id 0). No generation_config.json: generate() defaults are greedy, and config max_length=131072 gets moved into the generation config, so ALWAYS pass max_new_tokens or it will run until 131K tokens/EOS.
- CHAT TEMPLATE: the -Base chat_template is '{% for message in messages %}{% if loop.first %}<s>{% endif %}{{ message['content'] }}{% endfor %}' i.e. no roles; it is a base model, prompt it with raw few-shot text. Do not use eos_token_id=166101 (that is the instruct <|im_end|>; -Base vocab is 166100 and has no such token).
- ATTENTION: config default _attn_implementation resolves to 'eager' in 4.45.1 unless you pass attn_implementation='sdpa' (both verified on MPS). flash_attention_2 is imported only if the flash_attn package exists (CUDA-only); do not install it. StaticCache / torch.compile static cache is explicitly rejected when num_loops>1 (line 2143) - use the default DynamicCache. In 4.45.1 the returned past_key_values is a legacy tuple (return_legacy_cache) which prints a deprecation warning - harmless.
- LOOP COUNT: only changeable via config (model.config.num_loops), read per forward. Changing it mid-generation corrupts the KV layout; set before generate(). No published results for num_loops != 2; loops=1 is an untrained regime. Report from the authors: sharing KV across loops (loop_share_kv) was rejected for quality, so the full 44-layer KV cache is intentional.
- device_map='auto' (README) requires accelerate and on Mac maps to mps; the README's .to('cuda') must be replaced by .to(device). torch_dtype='auto' gives bf16 which works on MPS with torch 2.14; use torch.float16 if an older torch throws bf16 op errors.
- OLLAMA: stock Ollama v0.33.3 (2026-09-02) vendors llama.cpp b10760 (LLAMA_CPP_VERSION file) which is newer than the NANBEIGE merge (PR #25994 merged 2026-07-27; first tag b10159 dated 2026-07-28), and the Nanbeige/ollama nanbeige42 fork's only Ollama-side changes are a KV-VRAM estimate (layers *= num_loops), a chat renderer/parser for the instruct model, and an MLX-runner Go model - none are load gating. So `ollama run hf.co/mradermacher/Nanbeige4.2-3B-Base-GGUF:Q4_K_M` is expected to work through Ollama's llama.cpp runner, with Ollama under-estimating KV memory by 2x. NOT confirmed by running (Ollama not installed here); no Ollama release note mentions nanbeige. If it fails, use llama-server (see mlx_or_gguf_path) on a separate port next to Ollama - it is OpenAI-compatible.
- GGUF quality caveat: llama.cpp PR #25994 reviewers noted num_loops is read as optional-with-default=1, so a GGUF converted with an old converter would silently run single-pass; mradermacher's files were made after the merge, but verify by checking `nanbeige.num_loops = 2` in the llama.cpp load log. Quantized KV (-ctk q8_0) was flagged untested in the PR.

**Official eval recipe:** Base model (what can be replicated): report arXiv 2607.22083 sec 2.3 / Table 1 (also the HF card) evaluates Nanbeige4.2-3B-Base on GSM8K 92.7, BBH 81.6, MBPP 67.6, MMLU-Pro 63.8, SuperGPQA 35.2, GPQA 53.3 against Qwen3.5-4B(-Base) (84.4/79.1/57.1/51.8/32.1/43.1), Gemma4-E4B(-Base) (61.8/62.5/53.5/37.6/23.3/27.5) and Nanbeige4-3B-Base (85.9/70.7/60.7/47.6/24.8/36.2). The report, the HF card, and the bundled Nanbeige42_report.pdf give NO harness, prompt format, few-shot count, decoding, or max-token setting for these base numbers (the PDF text contains zero occurrences of 'shot'); there is no public eval code in the Nanbeige GitHub org (only sglang/vllm/llama.cpp/ollama forks). The only Nanbeige-published few-shot convention is the Nanbeige4-3B technical report (arXiv 2512.06266) footnote: 'For GSM8k, Cmath, and BBH, we run 3-shot evaluation. For MMLU, CMMLU, and MMLU-Pro, we run 5-shot evaluation' - but that footnote is attached to a scheduler-ablation table (Table 2, 1B model), not to the base-model comparison, so treat 3-shot GSM8K/BBH and 5-shot MMLU-Pro as the best available guess, not an official recipe. Practical replication: run lm-evaluation-harness with identical settings for both models, e.g. `lm_eval --model hf --model_args pretrained=Nanbeige/Nanbeige4.2-3B-Base,trust_remote_code=True,dtype=bfloat16,use_fast_tokenizer=False --tasks gsm8k --num_fewshot 3 --batch_size 1 --device mps` vs `pretrained=Qwen/Qwen3.5-4B-Base`, greedy, and report your own settings; expect a gap vs 92.7 because the authors' prompt/extraction is unknown. Instruct model (Nanbeige/Nanbeige4.2-3B) settings ARE published (report Appendix B, HF card): temperature 0.6, top-p 0.95, top-k 20, 256k context (HF card: max_new_tokens up to 131072 for reasoning; temperature 1.0 and 65,536 max_new_tokens for agentic/tool use); code-agent benchmarks: OpenHands (SWE-bench Verified), SWE-agent (SWE-bench Pro), Harbor/Terminus-2 (Terminal-Bench 2.0) with 256k ctx, 32k max output, temperature 1.0, 250 turns, averaged over 8 runs; thinking retained during agent evals, content before </think> stripped for scoring; judges DeepSeek-V4-Pro / GLM-5.1 / Qwen3.7-Plus / DeepSeek-V4-Flash depending on benchmark. The instruct repo's .eval_results/*.yaml (gpqa diamond 87.4, MathArena hmmt_feb_2026 82.1) only record scores, not configs.

**Python:** System Python 3.9 will NOT work: torch 2.14.0 requires Python >=3.10 (PyPI metadata). Use uv: `uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install torch "transformers==4.45.1" sentencepiece protobuf accelerate`. This exact env (CPython 3.12.13, torch 2.14.0, transformers 4.45.1, MPS available=True) was built and tested on the target M4 during this task. Homebrew Python 3.14.7 is installed but torch 3.14 wheels were not tested; 3.12 is the safe choice. transformers 4.45.1 itself allows >=3.8 so 3.11/3.12/3.13 are all fine. The mps-fix repo (pyproject) requires >=3.10 as well.


---

### Ouro-2.6B (ByteDance Seed) — 48 physical layers looped total_ut_steps=4 (R4) with entropy-regularized exit gate; 2.668B params, Apache-2.0. Sibling: ByteDance/Ouro-2.6B-Thinking (reasoning SFT, same architecture).

**Repo:** ByteDance/Ouro-2.6B  
**Download:** 5.34 GB · **RAM:** ~9 GB

**Loop control:** total_ut_steps — a config field read once at __init__ (`self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)`) and looped as `for current_ut in range(self.total_ut_steps)` in OuroModel.forward. Change it per call by setting BOTH `model.model.total_ut_steps = N` and `model.config.total_ut_steps = N` before generate (the cache size is `num_hidden_layers * config.total_ut_steps`; mismatch -> `IndexError: Cache index ... exceeds configured max_cache_size`, verified). Verified layer-call counts: 4 steps=24, 2 steps=12, 6 steps=36 on a 2-layer tiny config. Per-call readout kwargs on OuroForCausalLM.forward that generate() forwards: `exit_at_step` (0-based), `exit_threshold` (cumulative exit-gate prob; config default early_exit_threshold=1.0 = last step), `use_weighted_exit` — all post-hoc (every step still computed). Trained depth is 4; the paper evaluates 1–8 (5–8 = extrapolation, mild degradation).

**Thinking mode:** Base Ouro-2.6B: none (plain ChatML template, no <think>). Ouro-2.6B-Thinking (ByteDance/Ouro-2.6B-Thinking, same 48-layer/4-step config, bos=1/eos=2 already fixed there): its tokenizer_config chat_template has `{%- if enable_thinking is defined and enable_thinking is true -%}{{- '<think>\n' }}` after the assistant header, so `tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)` opens a think block; authors' decoding: temperature=1.0, top_p=0.7 (README example max_new_tokens=512; paper safety eval 8192). Whether the Thinking model emits <think> unprompted was not confirmed.

**Dependencies:**

- uv venv --python 3.12  (torch 2.8.0 has no cp314 wheels; system python here is actually 3.14.7)
- torch==2.8.0  (MPS available; verified with transformers 4.54.1 on this Mac)
- transformers==4.54.1  (README-recommended '<4.56'; needs the discussion #3/#4 patch below — stock code crashes on 4.54.1, verified)
- accelerate  (optional; 1.14.0 current)
- safetensors
- huggingface_hub  (snapshot_download)
- ALT path (transformers 5.x): transformers==5.16.1 + KristianS7/Ouro-2.6B fork (verified on tiny config; use dtype= not torch_dtype=)
- lm_eval[hf]==0.4.13  (for the paper's base-model evals)
- evalplus  (HumanEval/+ and MBPP/+ ; pip install 'evalplus' or the git main)
- MLX path: 'mlx-lm @ git+https://github.com/kernelpool/mlx-lm@feature/ouro'  (pulls mlx 0.32.2 / mlx_lm 0.32.0; stock mlx-lm 0.31.3 has NO ouro model)

**Quantized / alternative runtimes:** MLX: mlx-community/Ouro-2.6B-4bit (1.5 GB, 4-bit affine g64, model_type 'ouro', total_ut_steps=4, bos=1/eos=2, chat_template.jinja included; converted with mlx-lm 0.28.4). Stock mlx-lm (0.31.3 on PyPI, main branch) has NO mlx_lm/models/ouro.py (raw URL 404) — you must install the still-open PR ml-explore/mlx-lm#599 branch (kernelpool:feature/ouro, last commit 4482cbc 2026-08-21, PR updated 2026-08-24, reviewer awni open to merging as experimental): `uv venv --python 3.12 .venvmlx && source .venvmlx/bin/activate && uv pip install "mlx-lm @ git+https://github.com/kernelpool/mlx-lm@feature/ouro"` (installs mlx 0.32.2 / mlx_lm 0.32.0). Run: `mlx_lm.generate --model mlx-community/Ouro-2.6B-4bit --prompt "What is 17*23?" --max-tokens 200` or Python: `from mlx_lm import load, generate; model, tok = load("mlx-community/Ouro-2.6B-4bit"); generate(model, tok, prompt=tok.apply_chat_template([{"role":"user","content":"hi"}], add_generation_prompt=True), max_tokens=200, verbose=True)`. Loop count in MLX: set `model.model.total_ut_steps = N` and `model.args.total_ut_steps = N` before generation (make_cache allocates total_ut_steps*num_layers KVCache slots; verified 8 -> 12 slots for 4 -> 6 steps on a tiny random config). `exit_at_step`/`exit_threshold` exist on Model.__call__ but mlx_lm.generate does not forward them — only via direct model(x, exit_at_step=...) calls. VERIFIED: the PR's ouro.py loads, loops, caches, exits, and extrapolates on a tiny random config on this Mac. NOT VERIFIED: loading the actual 4-bit weights (not downloaded due to disk) or tok/s. Other MLX repos: ArcadaLabs/Ouro-2.6B-mlx-bf16 (bundles its own ouro.py but mlx-lm still resolves model_type from the installed package), mlx-community/Ouro-2.6B-Thinking-4bit. GGUF: none found for base Ouro-2.6B; scpalmetto/Ouro-2.6B-Thinking-Fixed and AXIOM-TECH/Ouro-2.6B-Thinking-IBNN-GGUF exist for the Thinking variant only and run as a plain single-pass 48-layer net in llama.cpp/Ollama (no looped-architecture support; ollama/ollama#14252 open) — not the R4 model.

**Minimal load + generate:**

```python
# ouro_local.py -- Ouro-2.6B on Apple Silicon (MPS, CPU fallback)
# Verified env: uv python 3.12, torch==2.8.0, transformers==4.54.1  (patch below is REQUIRED on 4.54.1)
#   uv venv --python 3.12 .venv && source .venv/bin/activate
#   uv pip install torch==2.8.0 transformers==4.54.1 accelerate safetensors huggingface_hub
import pathlib, torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "ByteDance/Ouro-2.6B"           # 5.34 GB bf16 safetensors + ~5 MB tokenizer files
local = pathlib.Path(snapshot_download(REPO))

# 1) Patch the trust_remote_code file in the local snapshot (HF discussions #3 + #4, still unmerged upstream).
#    Without this, transformers 4.54.1 raises: AttributeError: property 'key_cache' of 'UniversalTransformerCache' has no setter
p = local / "modeling_ouro.py"; s = p.read_text()
if "def get_mask_sizes" not in s:
    s = s.replace("self.key_cache", "self.k_cache").replace("self.value_cache", "self.v_cache")   # #3
    s = s.replace("    @property\n    def is_compileable(self) -> bool:",                          # #4
        "    def get_mask_sizes(self, cache_position, layer_idx: int = 0):\n"
        "        q = cache_position if isinstance(cache_position, int) else cache_position.shape[0]\n"
        "        return self.get_seq_length(layer_idx) + q, 0\n\n"
        "    @property\n    def is_compileable(self) -> bool:")
    p.write_text(s)

# 2) Load (bf16 on MPS needs macOS>=14 -- this machine is macOS 26; fall back to float16 if bf16 errors)
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "mps" else torch.float32
tok   = AutoTokenizer.from_pretrained(local)
model = AutoModelForCausalLM.from_pretrained(local, trust_remote_code=True, torch_dtype=dtype,
                                             attn_implementation="sdpa").to(device).eval()

# 3) Loop / recurrence count.  NOT a generate() kwarg.  Set BOTH attributes (verified):
#    model.model.total_ut_steps drives `for current_ut in range(total_ut_steps)`;
#    model.config.total_ut_steps sizes the KV cache (48 layers * steps slots) -> IndexError if they disagree.
def set_loops(n: int):          # trained max = 4 (R4); 5-8 = extrapolation (paper Table 11)
    model.model.total_ut_steps = n
    model.config.total_ut_steps = n
set_loops(4)

# 4) Prompt: ChatML template ships in tokenizer_config.json (default system prompt "You are a helpful assistant.").
#    Base model also accepts raw text; for plain completion just use tok("The future of AI is", ...).
messages = [{"role": "user", "content": "What is 17 * 23? Think step by step."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                         eos_token_id=[0, 2],   # 0=<|endoftext|> (config eos) and 2=<|im_end|> (PR #5 unmerged)
                         pad_token_id=0,
                         # Optional per-call exit control -- these DO reach forward() via generate() (verified),
                         # but are post-hoc: all `total_ut_steps` are still computed, only the readout changes.
                         # exit_at_step=1,        # 0-based index -> read logits from hidden state after step 2
                         # exit_threshold=0.5,    # cumulative exit-gate prob; config default 1.0 == always last step
                         )
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

# --- Thinking variant (ByteDance/Ouro-2.6B-Thinking): same code, plus
#   prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)  # appends "<think>\n"
#   model.generate(..., do_sample=True, temperature=1.0, top_p=0.7, max_new_tokens=8192)   # authors' decoding settings
# --- transformers 5.x alternative: REPO="KristianS7/Ouro-2.6B" (already patched, bos=1/eos=2/pad=0 fixed), skip step 1,
#   pass dtype=dtype instead of torch_dtype=dtype.
```

**Gotchas:**

- Stock upstream modeling_ouro.py CRASHES on the README-recommended transformers 4.54.1 (AttributeError: property 'key_cache' of 'UniversalTransformerCache' object has no setter) — reproduced here on MPS with sdpa and eager. Apply the discussion #3 rename (key_cache->k_cache, value_cache->v_cache) + #4 get_mask_sizes patch to the local snapshot; both PRs are still unmerged as of last upstream commit 2026-01-18 (repo sha 1ed04250).
- Without the #4 get_mask_sizes patch, batch>1 generation is corrupted on eager and raises RuntimeError on sdpa (the only backends usable on Mac). With the patch, batch=2 works on MPS (verified). lm-eval loglikelihood tasks batch/pad, so patch before evals.
- Loop count is NOT a generate() kwarg; set model.model.total_ut_steps AND model.config.total_ut_steps (see loop_param). exit_at_step/exit_threshold only change which step's hidden state is read out — no compute is saved (upstream is 'post-hoc gating'; KristianS7 and HF Thinking discussion #12 both note this, no ByteDance reply).
- KV cache is 4x a normal 48-layer model: 48 layers x 4 steps = 192 K/V slots x 16 heads x 128 dim x bf16 = ~1.57 MB per token. 2k context ~3.1 GB, 4k ~6.3 GB on top of 5.34 GB weights. total_ut_steps=8 doubles it. Keep contexts short on 16 GB.
- Speed: every token runs 192 decoder-layer passes (~10.7B-dense-equivalent compute). No Apple-Silicon tok/s report exists anywhere; expect low single-digit tok/s in bf16 on an M4 (estimate, unverified). Full-benchmark evals (MMLU 14k items, MMLU-Pro CoT 2048 max tokens) will take many hours to days on this machine.
- Disk: 5.34 GB bf16 download vs 3.5–5 GB free — free space first (HF cache defaults to ~/.cache/huggingface). MLX 4-bit is 1.5 GB if you go the MLX route.
- bf16 on MPS requires macOS>=14 (PyTorch); this machine is Darwin 25 so bf16 works (verified on tiny model). If you hit 'BFloat16 is not supported on MPS', use torch.float16. RoPE is computed in fp32 with autocast disabled (code special-cases mps), so fp16 is safe for positions.
- Config bos/eos are both 0 (<|endoftext|>); the ChatML template ends turns with <|im_end|> (id 2). Pass eos_token_id=[0,2] to generate or chat outputs may run on (PR #5 unmerged). No pad_token_id in upstream config; use pad_token_id=0.
- Attention: use attn_implementation='sdpa' or 'eager'; flash-attn/flex are declared but unavailable on Mac. No .cuda() calls in the code, MPS/CPU works via ALL_ATTENTION_FUNCTIONS.
- transformers>=4.56 breaks the cache further; transformers 5.x only works with the KristianS7/Ouro-2.6B fork (verified on 5.16.1; it needs pad_token_id in config — the fork's config has it, upstream's does not; use dtype= not torch_dtype=). The 'Feb-2026 transformers-5 fixes' claim applies to that fork, not upstream.
- Ollama/llama.cpp cannot run the looped model: no GGUF for base Ouro-2.6B, Thinking-only GGUFs would execute a single pass (no looped-arch support; ollama/ollama#14252 open since 2026-02-14 with no maintainer response). To pair with Ollama baselines, serve Ouro yourself (transformers or `mlx_lm.server` from the PR branch, unverified for ouro) behind an OpenAI-compatible endpoint and point lm-eval `--model local-completions` at both it and Ollama's /v1.
- Official GitHub code is still 'Coming Soon' on ouro-llm.github.io; rkstgr/LoopLM is a third-party reimplementation that does not load Ouro weights.
- Thinking variant: base Ouro-2.6B has no thinking mode. Ouro-2.6B-Thinking's template supports enable_thinking=True (adds '<think>\n'); its README uses max_new_tokens=512 but the paper's safety eval uses 8192 with temp=1.0/top_p=0.7. Whether it reasons without the flag was not confirmed.

**Official eval recipe:** Paper (arXiv 2510.25741, v5 2026-07-01), Sec. 5.1 + Appendix C.1 Table 16, base models: 'All benchmarks are evaluated using lm-eval-harness and evalplus'. Settings: MMLU logprobs 5-shot (lm-eval); MMLU-Pro strict match 5-shot CoT (lm-eval); BBH strict match 3-shot CoT (lm-eval); ARC-C logprobs 25-shot; HellaSwag logprobs 10-shot; Winogrande logprobs 5-shot; GSM8K strict match 3-shot CoT (lm-eval); MATH500 strict match 5-shot CoT (IN-HOUSE harness — not exactly reproducible); HumanEval/HumanEval+/MBPP/MBPP+ pass@1 via evalplus (greedy). Table 8 targets, Ouro-2.6B R4 vs Qwen3-8B-Base vs Qwen3-4B-Base: MMLU 74.60/76.63/73.19; MMLU-Pro 55.73/53.72/51.40; BBH 80.46/77.65/71.14; ARC-C 66.40/66.10/63.65; HellaSwag 79.69/79.60/75.66; Winogrande 75.85/76.80/71.19; GSM8K 81.58/83.09/72.86; MATH500 90.85/62.30/59.60; HumanEval 78.70/84.80/77.70; HumanEval+ 70.70/75.30/70.70; MBPP 80.40/79.00/78.80; MBPP+ 66.60/67.90/65.90. Table 11 (Ouro-2.6B base by step, for R1..R8 replication): MMLU 51.55/67.63/73.57/74.60/74.43/73.79/72.92/72.24; ARC-C 47.95/62.37/65.36/66.38/65.36/65.02/65.44/64.76; HellaSwag 68.94/77.61/79.12/79.56/...; Winogrande 61.48/70.48/74.35/75.53/75.93/.... Replication commands (lm_eval 0.4.13; point `pretrained` at the PATCHED local snapshot, since lm-eval loads the remote code unpatched; sweep steps by editing total_ut_steps in $LOCAL/config.json between runs): `LOCAL=$(python -c "from huggingface_hub import snapshot_download as s;print(s('ByteDance/Ouro-2.6B'))")`; `lm_eval --model hf --model_args pretrained=$LOCAL,trust_remote_code=True,dtype=bfloat16 --device mps --batch_size 1 --tasks mmlu --num_fewshot 5`; `... --tasks gsm8k_cot --num_fewshot 3` (harness default is 8-shot with sampler first_n, so --num_fewshot 3 takes the first 3; report the 'strict-match' filter); `... --tasks mmlu_pro` (default 5-shot CoT generate_until, max_gen_toks 2048, greedy — matches); `... --tasks bbh_cot_fewshot` (3-shot CoT, max_gen_toks 1024, greedy — matches); `... --tasks arc_challenge --num_fewshot 25`; `... --tasks hellaswag --num_fewshot 10`; `... --tasks winogrande --num_fewshot 5`; MATH500 nearest proxies: `--tasks hendrycks_math500 --num_fewshot 5` ('Problem:/Answer:' format, exact_match) or `--tasks minerva_math500` (4-shot Minerva prompt, math_verify) — neither is the authors' in-house harness. Code: `evalplus.evaluate --model $LOCAL --dataset humaneval --backend hf --greedy --trust-remote-code --dtype bfloat16` and `--dataset mbpp` (evalplus auto-detects base vs chat prompting from the chat template; evalplus also has --backend ollama for Ollama baselines). Thinking models (Table 17): in-house harness, LLM-as-judge with fixed rubric, temperature=1.0, top_p=0.7, safety eval max_new_tokens=8192 (base: greedy, max_new_tokens=128) on AIME24/25, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE — not reproducible with public tooling. All generate tasks are greedy (do_sample=false). Not confirmed: whether the authors ran lm-eval with a chat template (assume no, base model), and lm-eval's `--device mps` end-to-end with the full 2.6B weights (harness docs call MPS 'early stages').

**Python:** Use `uv venv --python 3.12 .venv` (verified). The system Python on this machine is 3.14.7 (not 3.9): torch 2.8.0 publishes no cp314 macOS wheels, so a uv-managed 3.12 is required. transformers 4.54.1 requires Python >=3.9; latest torch 2.14.0 requires >=3.10 but was NOT tested with transformers 4.54.1 — pin torch==2.8.0. The transformers-5 path (KristianS7 fork) was verified on 3.12 with transformers 5.16.1 + torch 2.8.0.


---
