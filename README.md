---
license: apache-2.0
language:
  - en
base_model: ByteDance/Ouro-1.4B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - looped-language-model
  - recurrent-depth
  - universal-transformer
  - ouro
  - math
model-index:
  - name: Ouro-1.4B-GGUF
    results:
      - task:
          type: text-generation
          name: Math word problems
        dataset:
          name: GSM8K (200-problem subset, 3-shot CoT, strict match)
          type: openai/gsm8k
          config: main
          split: test
        metrics:
          - name: accuracy (F16, 4 loops)
            type: accuracy
            value: 80.5
          - name: accuracy (Q8_0, 4 loops)
            type: accuracy
            value: 79.8
          - name: accuracy (Q4_K_M, 4 loops)
            type: accuracy
            value: 75.0
          - name: accuracy (F16, 1 loop)
            type: accuracy
            value: 26.0
        source:
          name: BrandeisPatrick/loop-transformer
          url: https://github.com/BrandeisPatrick/loop-transformer/blob/main/results/REPORT.md
---

# Ouro-1.4B — GGUF

GGUF conversions of [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), a **looped
language model**: the entire 24-layer decoder stack is applied 4 times per token with shared weights,
so a 1.4B-parameter model computes at an effective depth of 96 layers.

## What this repository is, and is not

| | |
|---|---|
| **Model weights** | ByteDance Seed's, unchanged. Trained by them, licensed Apache-2.0 by them. Nothing here was fine-tuned. |
| **What was added** | The `ouro` **architecture for llama.cpp** — the graph, the HF→GGUF conversion, and the registration — so the weights can be loaded by GGUF runtimes at all. Plus these conversions and their measured accuracy. |
| **Upstream status** | Submitted upstream as [ggml-org/llama.cpp#29823](https://github.com/ggml-org/llama.cpp/pull/29823). Until it merges, these files need the patched build linked below. |
| **Credit** | If you use the model, cite ByteDance's paper (below). If you use the port or the evaluation harness, link the GitHub repository. |

These are the first GGUFs of this architecture. llama.cpp had no `ouro` architecture, so no GGUF
runtime could load Ouro at all — the request on the Ollama tracker
([#14252](https://github.com/ollama/ollama/issues/14252)) has been open since February 2026. The
architecture was written for this release; the patch and the evaluation harness are at
[BrandeisPatrick/loop-transformer](https://github.com/BrandeisPatrick/loop-transformer).

> **These files need a patched llama.cpp today.** The `ouro` architecture is not yet in upstream
> llama.cpp, so stock llama.cpp, Ollama and LM Studio cannot load them **yet**. Build with the patch:
> `git clone https://github.com/BrandeisPatrick/loop-transformer && loop-transformer/llamacpp/build.sh`.
> Once the upstream PR merges and Ollama bumps its llama.cpp pin, these files will run unmodified.

## Files

| file | size | GSM8K | note |
|---|---|---|---|
| `Ouro-1.4B-F16.gguf` | 2.7 GB | 80.5% | reference precision |
| `Ouro-1.4B-Q8_0.gguf` | 1.4 GB | 79.8% | **recommended** — matches F16 within noise |
| `Ouro-1.4B-Q4_K_M.gguf` | 854 MB | 75.0% | smallest; costs ~5 points, see below |

GSM8K is a fixed 200-problem subset (100 for the quants), 3-shot chain of thought, greedy, strict
answer match — the protocol the Ouro paper specifies in its Table 16. The published figure for this
model is **78.92**.

## Verification against the reference implementation

The port was validated against numbers measured with the original transformers implementation on the
same machine *before* it existed, so this is a comparison to data rather than to an impression.

| loops | this GGUF (F16) | transformers bf16 | seconds/item |
|---|---|---|---|
| 1 | 26.0 | 23.0 | 2.4 vs 11.1 |
| 2 | 67.0 | 64.0 | ~6 vs 19.3 |
| 4 | **80.5 ± 2.8** | **80.0 ± 2.8** | 11.3 vs 38.6 |

Every depth is within one standard error, greedy output is token-identical on a smoke prompt, and the
model loads as `n_layer = 96` at `1.43 B` parameters — depth expanded, weights stored once. On an
Apple M4 the GGUF runs about 3.4x faster than the reference does on MPS.

## The loop count is a runtime dial

Unusually for a GGUF, the compute/accuracy trade-off is adjustable at load time from a single file,
because llama.cpp's loader consults key overrides before the file:

```bash
llama-cli -m Ouro-1.4B-Q8_0.gguf --override-kv ouro.num_loops=int:1   # 24 layers, ~36 tok/s
llama-cli -m Ouro-1.4B-Q8_0.gguf --override-kv ouro.num_loops=int:2   # 48 layers
llama-cli -m Ouro-1.4B-Q8_0.gguf                                      # 96 layers, ~9.5 tok/s, default
```

Accuracy follows depth: 26 / 67 / 80 percent on GSM8K at 1 / 2 / 4 loops. Note the model was trained
at 4 loops; the published ablation shows quality degrading beyond that.

## Quantization notes

Q4_K_M costs about 5 points, more than a dense model this size usually loses. A natural hypothesis is
that a looped model re-applies the same weight error once per loop, so the damage compounds with depth.
**That was tested and is false**: the Q4 penalty is 6.0 points at depth 1 and 5.5 at depth 4, a
difference of −0.5 against a combined standard error of 7.9. The cost is flat in depth. Q8_0 is
effectively lossless and is the recommended file.

Always quote the loop depth alongside a number from these files — the same file scores 20% or 75%
depending only on a load-time flag.

## Provenance

Converted from `ByteDance/Ouro-1.4B` at commit `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` with a
llama.cpp built from upstream `67672dc` plus the `ouro` architecture patch. The early-exit gate is
deliberately not converted: it selects which already-computed loop feeds the LM head rather than
changing what is computed, and at the shipped `early_exit_threshold = 1.0` it never fires (measured:
0 of 299 token positions exit early).

## Citation

```bibtex
@article{ouro2025,
  title  = {Scaling Latent Reasoning via Looped Language Models},
  author = {ByteDance Seed},
  journal= {arXiv:2510.25741},
  year   = {2025}
}
```
