# Ouro for llama.cpp

Adds the **`ouro`** architecture to llama.cpp, so ByteDance's looped language models
([Ouro-1.4B / 2.6B and their Thinking variants](https://huggingface.co/collections/ByteDance/ouro),
arXiv 2510.25741) can run as GGUF — and therefore in **Ollama**, LM Studio, and anything else that
vendors llama.cpp. Before this, no GGUF runtime could load them: llama.cpp has no `ouro` architecture
and [ollama/ollama#14252](https://github.com/ollama/ollama/issues/14252) has sat unanswered since
February 2026.

## How it works

Ouro applies its **entire** decoder stack `total_ut_steps` times per token with shared weights, and
each layer caches K/V at `current_ut * num_hidden_layers + layer_idx` — so every (loop, layer) pair
owns an independent cache slot. llama.cpp already has exactly this machinery, added for Nanbeige4.2
in [PR #25994](https://github.com/ggml-org/llama.cpp/pull/25994): the generic `{arch}.num_loops` GGUF
key, an unroll to `n_layer_phys * n_loops` logical layers, and weight sharing across loop slots. This
port reuses all of it.

Ouro differs from Nanbeige in one respect: its block uses **sandwich norms**, normalising the
attention output and the FFN output before each residual add,

```python
h = x + input_layernorm_2(attn(input_layernorm(x)))          # modeling_ouro.py
h = h + post_attention_layernorm_2(mlp(post_attention_layernorm(h)))
```

which maps onto the existing `ATTN_POST_NORM` / `FFN_POST_NORM` tensors. Note the name trap: Ouro's
`post_attention_layernorm` is the **pre-FFN** norm (Llama convention), not the post-attention norm the
generic tensor map assumes for gemma2/olmo2 — so `conversion/ouro.py` maps the four norms explicitly.

The `early_exit_gate` is deliberately **not** converted. In the reference it only selects which
already-computed loop's hidden state is read out; it never changes what is computed, so a fixed-depth
graph is faithful to the published numbers.

## Build and convert

```bash
llamacpp/build.sh                                    # clone upstream, apply the patch, build with Metal
llamacpp/convert.sh ByteDance/Ouro-1.4B models Q4_K_M   # HF -> GGUF (+ optional quants)
third_party/llama.cpp/build/bin/llama-cli -m models/Ouro-1.4B-F16.gguf --single-turn -p "..."
```

Loading it prints `n_layer = 96` for the 1.4B (24 physical layers × 4 loops) at `1.43 B` parameters —
logical depth expanded, weights shared.

## Verification

| check | result |
|---|---|
| loads with the right shape | `arch = ouro`, `n_layer = 96`, `params = 1.43 B` |
| greedy output vs the transformers reference | token-identical on the smoke prompt |
| GSM8K, 200 problems, 3-shot CoT strict | see [../results/REPORT.md](../results/REPORT.md) |
| speed, Apple M4, F16 | ~12 s/item vs ~39 s/item for transformers on MPS |

The reference anchor this is validated against — Ouro-1.4B at 4 loops scoring **80.0%** on a fixed
200-problem GSM8K subset — was measured in this repo with the transformers implementation before the
port existed, so the port is checked against a number, not against an impression.

## Files

- `ouro.cpp` → `src/models/ouro.cpp`, the graph
- `ouro.py` → `conversion/ouro.py`, the HF→GGUF conversion
- `0001-model-add-Ouro.patch` — the whole change, including registration, against upstream `f1cee99`
- `build.sh`, `convert.sh`

## Upstreaming

Submitted upstream as [ggml-org/llama.cpp#29823](https://github.com/ggml-org/llama.cpp/pull/29823) (opened 2026-10-01, one commit, nine files).

llama.cpp closed the previous looped-model contribution
([PR #18680](https://github.com/ggml-org/llama.cpp/pull/18680), IQuest-Coder) for not following its
contribution guidelines on AI-generated code, so this PR discloses AI assistance up front, in the
description and in the commit trailers.
