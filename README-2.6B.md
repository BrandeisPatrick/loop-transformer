---
license: apache-2.0
base_model: ByteDance/Ouro-2.6B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - looped-language-model
  - recurrent-depth
  - ouro
---

# Ouro-2.6B — GGUF

GGUF conversions of [ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B), the larger
base model of the **Ouro** family of looped language models. The 48-layer decoder stack
is applied 4 times per token with shared weights, so a 2.7B-parameter model computes at an effective
depth of **192 layers**.

These are the first GGUFs of this architecture. llama.cpp had no `ouro` architecture, so no GGUF
runtime could load Ouro at all. The architecture was written for this release; the patch, the
evaluation harness and the validation data are at
[BrandeisPatrick/loop-transformer](https://github.com/BrandeisPatrick/loop-transformer).

> **Needs a patched llama.cpp today.** The `ouro` architecture is not yet upstream, so stock llama.cpp,
> Ollama and LM Studio cannot load these files **yet**. Build with the patch:
> `git clone https://github.com/BrandeisPatrick/loop-transformer && loop-transformer/llamacpp/build.sh`
> Once the upstream PR merges and Ollama bumps its pin, these run unmodified.

## Files

| file | size | note |
|---|---|---|
| `Ouro-2.6B-F16.gguf` | 5.3 GB | reference precision |
| `Ouro-2.6B-Q8_0.gguf` | 2.8 GB | **recommended** — Q8_0 was lossless within noise on the 1.4B |
| `Ouro-2.6B-Q4_K_M.gguf` | 1.65 GB | smallest; on the 1.4B, Q4_K_M cost ~5 points |

## What is verified here, and what is not

Being precise, because this variant was not benchmarked end-to-end:

**Verified on these files.** Loads as `arch = ouro`, `n_layer = 192` (48 physical × 4 loops) at
`2.67 B` parameters — depth expanded, weights stored once — and generates correct, coherent output
(~8.3 tok/s on an Apple M4, Q8_0).

**Verified on the same code path, using the smaller [Ouro-1.4B](https://huggingface.co/BrandeisPatrick/Ouro-1.4B-GGUF).**
The port reproduces the transformers reference across loop depths: 26.0 / 67.0 / 80.5 percent on GSM8K
at 1 / 2 / 4 loops against 23.0 / 64.0 / 80.0, every point within one standard error, with
token-identical greedy output. That is the evidence the architecture is correct.

**Not benchmarked here.** The paper reports this model at GSM8K 81.58 and MATH500 90.85 (3-shot and
5-shot CoT, strict match), but those runs were not repeated for this file, so no accuracy figure is
claimed for it. The 1.4B is the benchmarked one; on that model the port matched the published GSM8K
figure to within half a point.

## The loop count is a runtime dial

The compute/accuracy trade-off is adjustable at load time from a single file:

```bash
llama-cli -m Ouro-2.6B-Q8_0.gguf --override-kv ouro.num_loops=int:2   # 96 layers, ~2x faster
llama-cli -m Ouro-2.6B-Q8_0.gguf                                      # 192 layers, default
```

The model was trained at 4 loops. The paper's own ablation for this model gives MMLU 51.55 / 67.63 /
73.57 / 74.60 at depths 1-4, degrading beyond 4. On the 1.4B, a single loop collapses accuracy to 26% — always
quote the depth alongside any number from these files.

## Provenance

Converted from `ByteDance/Ouro-2.6B` with a llama.cpp built from upstream `67672dc` plus the
`ouro` architecture patch. The early-exit gate is deliberately not converted: it selects which
already-computed loop feeds the LM head rather than changing what is computed, and at the shipped
`early_exit_threshold = 1.0` it never fires.

## Citation

```bibtex
@article{ouro2025,
  title  = {Scaling Latent Reasoning via Looped Language Models},
  author = {ByteDance Seed},
  journal= {arXiv:2510.25741},
  year   = {2025}
}
```
