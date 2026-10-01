# Facts for the upstream llama.cpp PR (write the prose yourself)

**Status: opened as [ggml-org/llama.cpp#29823](https://github.com/ggml-org/llama.cpp/pull/29823) on 2026-10-01.**

llama.cpp forbids AI-written PR descriptions, commit messages, and reviewer replies
(CONTRIBUTING.md item 5; AGENTS.md "Prohibited AI Usage"). This file is a checked list of
facts, numbers, and links. Every sentence in the PR must be yours.

The PR template has three sections: **Overview**, **Additional information**, **Requirements**.
Keep the Requirements section; deleting it can get the PR rejected.

## Overview: what the PR adds

- New architecture `ouro` for ByteDance's Ouro looped language models.
- Covered checkpoints: `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, `ByteDance/Ouro-1.4B-Thinking`,
  `ByteDance/Ouro-2.6B-Thinking` (Hugging Face; paper linked from the model cards).
- Model shape: 24 physical layers (1.4B) or 48 (2.6B); the whole stack is applied
  `total_ut_steps = 4` times with shared weights; KV cache is indexed per (loop, layer);
  the shared final norm is applied at the end of every loop and its output feeds the next loop.
- Loads as `n_layer = 96` with 1.43 B parameters for the 1.4B model: logical depth expanded,
  weights stored once.

## Overview: how it fits existing code

- Reuses the looped-model machinery added for Nanbeige4.2 in ggml-org/llama.cpp#25994:
  GGUF keys `ouro.num_loops` and `ouro.skip_loop_final_norm`, unroll to
  `n_layer_phys * n_loops` logical layers, layer sharing across loop slots, per-slot KV.
  Ouro's cache index `current_ut * num_hidden_layers + layer_idx` is exactly the unrolled index.
- Between-loop norm: Ouro's schedule equals Nanbeige's `skip_loop_final_norm = false`; no new code.
- What differs from Nanbeige: the block is a Gemma-style sandwich norm with four RMSNorms per
  layer. The two output norms map onto the existing `ATTN_POST_NORM` and `FFN_POST_NORM`
  tensor types, so no new ggml op and no new tensor type.
- Conversion maps all four norms explicitly. Reason: Ouro's `post_attention_layernorm` is the
  pre-FFN norm (Llama naming), but the generic tensor map treats that name as the post-attention
  norm (gemma2 / olmo2 meaning). The generic map would wire two norms to the wrong place.
- Early-exit gate (`model.early_exit_gate*`) is not converted. It only selects which computed loop
  feeds the LM head. At the shipped `early_exit_threshold = 1.0` it never fires: 0 of 299 token
  positions exited early, maximum gate logit 0.504. Conversion warns if a checkpoint ships a
  threshold below 1.0.
- Depth at runtime, one file: `--override-kv ouro.num_loops=int:N` (1 to 4 for a 4-loop checkpoint,
  larger values also load). The loader consults overrides before the file.
- Rope type `LLAMA_ROPE_TYPE_NORM`; conversion sets `undo_permute = True` (Llama-style Q/K).
- `LLM_ARCH_OURO` added to the larger graph-node budget list in `llama-context.cpp`
  (same treatment as other deep-graph archs).

## Files changed (from `git diff --stat ouro-arch~1 ouro-arch`, base is upstream `f1cee99`)

| file | lines | what |
|---|---|---|
| `src/models/ouro.cpp` | +213 | hparams (num_loops, n_layer_all, 1.4B/2.6B type), tensor loading with sharing, graph |
| `src/models/models.h` | +15 | `struct llama_model_ouro` declaration |
| `src/llama-arch.h` / `src/llama-arch.cpp` | +1 / +1 | `LLM_ARCH_OURO`, name `"ouro"` |
| `src/llama-model.cpp` | +3 | factory case, rope type case |
| `src/llama-context.cpp` | +1 | graph-node budget |
| `gguf-py/gguf/constants.py` | +18 | `MODEL_ARCH.OURO`, tensor list incl. ATTN_POST_NORM / FFN_POST_NORM |
| `conversion/ouro.py` | +71 | `OuroModel(LlamaModel)`: keys, norm mapping, gate drop |
| `conversion/__init__.py` | +1 | `"OuroForCausalLM": "ouro"` |

9 files, 324 insertions, 0 deletions. No test files added (test-llama-archs enumerates all archs).

## Validation (all numbers measured in this repo; anchors were measured with the transformers
## reference before the port existed)

Greedy generation on the smoke prompt is token-identical to the transformers reference.

GSM8K 3-shot, same problems and prompt, F16 GGUF on Metal vs bf16 transformers on MPS, M4 16 GB:

| loops | llama.cpp F16 | transformers bf16 | n |
|---|---|---|---|
| 1 | 26.0 | 23.0 | 100 |
| 2 | 67.0 | 64.0 | 100 |
| 4 | 80.5 ± 2.8 | 80.0 ± 2.8 | 200 |

Every depth is within one standard error. Quantized, 4 loops: Q8_0 79.8 (n = 99), Q4_K_M 75.0 (n = 100).
Paper reports 78.92 for Ouro-1.4B on full GSM8K.

`tests/test-llama-archs` passes with `ouro` included, re-run on 2026-10-01 after rebasing onto
upstream `f1cee99`: exit 0, "all 506 test(s) passed". The three `ouro` backend rows report
1.60e-07, 1.72e-13 and 2.44e-13 (Apple M4, Accelerate, Apple M4). The Meta row is SKIP, as it is
for every architecture. Built with Apple clang 21 from the Command Line Tools. On the same build the
published Ouro-1.4B Q4_K_M file loads as `arch = ouro`, `n_layer = 96`, 1.43 B params and answers the
smoke prompt correctly.

Other checkpoints, smoke check only (Q4_K_M, prompt "What is 17 + 26?", expected 43): Ouro-2.6B passed,
Ouro-1.4B-Thinking passed, Ouro-2.6B-Thinking passed on 2026-09-30 using the rebased build (it needs a
token budget of a few hundred because it reasons before answering). No benchmark was run on these three.

Smoke prompt ("What is 17 + 26?", greedy), all four variants answer 43: Ouro-1.4B (full eval above),
Ouro-1.4B-Thinking and Ouro-2.6B (publish gate, 2026-09-07), Ouro-2.6B-Thinking Q4_K_M (re-checked
2026-09-30 on the rebased build, 13.3 tok/s, minimum free memory 49%). Only 1.4B has benchmark numbers.

Decode speed, F16, M4 16 GB Metal: 36.5 tok/s at 1 loop, 9.5 tok/s at 4 loops.

## Additional information: links

- Ollama request for Ouro, open since Feb 2026: ollama/ollama#14252
- Looped-model mechanism this reuses: ggml-org/llama.cpp#25994 (Nanbeige4.2)
- Converted GGUFs already published: `BrandeisPatrick/Ouro-1.4B-GGUF`, `BrandeisPatrick/Ouro-2.6B-GGUF`,
  `BrandeisPatrick/Ouro-1.4B-Thinking-GGUF`, `BrandeisPatrick/Ouro-2.6B-Thinking-GGUF`
- Eval harness and full write-up: https://github.com/BrandeisPatrick/loop-transformer (make public first)
- Disclosure wording you liked: ggml-org/llama.cpp#27591

## Requirements: AI disclosure, state these facts in your own words

- The C++ and Python in the diff were written with Claude Code (Anthropic) working under your direction.
- Design decisions were yours or approved by you: reuse the Nanbeige loop mechanism, map norms explicitly,
  drop the early-exit gate, keep the change to 9 files.
- You reviewed every line before submitting. Only true after the walkthrough; do the walkthrough first.
- The validation runs (GSM8K depth sweep, test-llama-archs, token comparison) were executed on your machine.
- The PR description, the commit message, and all replies to reviewers are written by you without AI.
- Do not reuse the "drafted with AI assistance" phrase from #27591 for the description; that is exactly
  what item 5 of CONTRIBUTING.md prohibits.

## Process rules from AGENTS.md that affect the next steps

- Commit message: you write it and you amend the commit yourself with `git commit --amend`
  inside `third_party/llama.cpp`.
- Push and PR creation must be done by you (AGENTS.md line 95). Commands:

```bash
gh repo fork ggml-org/llama.cpp --clone=false
```

```bash
cd third_party/llama.cpp && git remote add fork git@github.com:BrandeisPatrick/llama.cpp.git && git push fork ouro-arch
```

Then open the PR in the browser from `BrandeisPatrick:ouro-arch` against `ggml-org:master`.
