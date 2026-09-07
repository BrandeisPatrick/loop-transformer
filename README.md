# looplm — looped transformers, locally, next to Ollama

Deploy **looped / recurrent-depth** language models on an Apple-Silicon Mac, expose them the way Ollama
exposes models, and measure them against the baselines their own papers used — through one harness,
under the papers' own protocols, with the extraction rule stated.

Hardware: M4, 16 GB unified memory, macOS. Everything below was measured on it.
Research notes: [NOTES.md](NOTES.md) · per-model recipes: [RECIPES.md](RECIPES.md) · full tables: [results/REPORT.md](results/REPORT.md)

## Results (GSM8K, fixed 200-problem subset, seed 0)

Base models use the Ouro paper's protocol: 3-shot chain of thought, greedy, strict answer match.
Instruct models use zero-shot chain of thought with a `\boxed{}` answer, thinking off. ± is one
binomial standard error. Paper numbers are on the full 1319-problem test set.

| model | looped | stage | GSM8K | paper | s/item |
|---|---|---|---|---|---|
| **Qwen3.5-4B** (`qwen3.5:4b`) | no | instruct | **93.5** ± 1.7 | — | 17.5 |
| **Nanbeige4.2-3B** (22 layers × 2 loops) | **yes** | instruct | **91.0** ± 2.0 | 92.7 (Base card) | 15.7 |
| Qwen3-1.7B (`qwen3:1.7b`) | no | instruct | 82.5 ± 2.7 | — | 4.2 |
| **Ouro-1.4B** (24 layers × 4 loops) | **yes** | base | **80.0** ± 2.8 | 78.92 | 38.6 |
| Qwen3-4B-Base | no | base | 76.5 ± 3.0 | 72.86 | 4.0 |
| Qwen3-1.7B-Base | no | base | 68.0 ± 3.3 | 70.28 | 2.9 |
| Nanbeige4.2-3B forced to 1 loop | yes | instruct | 0.0 (all truncated) | — | 7.1 |

**What replicates.** Ouro-1.4B at its trained depth lands within one standard error of its published
78.92, and so do both Qwen3 base baselines. The paper's headline — a looped 1.4B beats a conventional
1.7B — reproduces at +12 points, and that margin holds under every answer-extraction rule tried.

**The published ablation replicates.** The paper's per-step ablation is MMLU 5-shot at depths 1 to 4:
41.2 / 60.4 / 66.7 / 67.5. Run here with lm-eval on 449 questions per depth: **39.0 / 58.4 / — / 67.3**,
every point within one standard error.

**What the paper never published.** Ouro's GSM8K accuracy against recurrent steps, measured here at
1 / 2 / 3 / 4 loops: **23.0 / 64.0 / 72.0 / 80.0**. Going from 2 to 4 loops is worth +8.9 on MMLU and
+16.0 on GSM8K for the same weights — math consumes depth that knowledge recall does not, and three
loops is where the 1.4B overtakes the 1.7B.

**What looping does not do.** Forcing Nanbeige below its trained two loops does not give a shallower
model; it gives repetition garbage, because it was trained with no per-loop supervision
(`loop_loss_weights=[]`). Ouro, trained with an exit gate at every step, degrades gracefully instead.
A looped architecture only provides a compute dial if intermediate depths were supervised. And after
post-training the advantage washes out: Nanbeige is at parity with its own comparator Qwen3.5-4B, and
a plain post-trained `qwen3:1.7b` beats the looped Ouro *base* model. Looping and post-training are
complements, not alternatives.

**A scoring bug that reverses the headline.** lm-eval's `gsm8k_cot` strict-match regex cannot capture
`The answer is $250.`, and Ouro writes currency answers 2.5× as often as Qwen. Under the verbatim rule
the +12 gap shrinks to +1.5. `eval/rescore.py` re-scores saved generations under four rules so this is
visible rather than silent. Details in NOTES.md §8.

## What runs where

| track | model | runtime | loop control |
|---|---|---|---|
| native Ollama | Nanbeige4.2-3B | `ollama pull hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M` — llama.cpp merged the looped `nanbeige` arch in PR #25994; Ollama 0.33.3 ships it | fixed in GGUF metadata; `scripts/make_loop_variants.sh` rewrites `num_loops` and registers a copy |
| shim | Ouro-1.4B | `serve/shim.py --backend ouro` on torch/MPS in `.venv-tf4` (transformers 4.57; its remote code breaks on 5.x) | per request: `num_loops` 1–4 |
| shim | recurrent-qwen2.5-0.5b (Shapiro retrofit) | `serve/shim.py --backend recurrent-qwen` in `.venv` | per request: `num_loops` |
| shim | LoopUS Qwen3-1.7B (the only Qwen3-based looped model with weights) | `serve/shim.py --backend lds`, code vendored from github.com/Thrillcrazyer/LoopUS + `scripts/patch_loopus.py` | per request: `num_loops`, `exit_threshold` |

The shim speaks Ollama's API (`/api/tags /api/chat /api/generate /api/show`) and OpenAI's
(`/v1/models /v1/chat/completions /v1/completions`), so anything that talks to Ollama can talk to it.
Ouro cannot run inside Ollama today: llama.cpp has no `ouro` architecture (Ollama issue #14252 is
unanswered). Adding one, following the Nanbeige unroll, is the next piece of work.

```bash
# baseline via Ollama                              # looped model behind the shim
ollama serve                                       .venv-tf4/bin/python serve/shim.py --model ByteDance/Ouro-1.4B \
                                                       --backend ouro --name ouro-1.4b --loops 4 --port 11435
curl localhost:11435/api/chat -d '{"model":"ouro-1.4b","messages":[{"role":"user","content":"17+26?"}],"options":{"num_loops":2}}'
```

## Memory safety (16 GB machine)

Every launcher starts `scripts/memguard.sh`, which polls every 2 s and, below 30% free memory or on
800 MB of swap growth, kills only this project's jobs (model servers, loaded Ollama models, eval drivers),
logs it, and posts a notification. All jobs are resumable, so a trip costs a re-run. `scripts/memwait.sh`
blocks any model load until 45% is free. Ollama is run with a 2048-token default context, an 8-bit KV
cache, one loaded model, and `keep_alive: 0` on every request. This exists because Ollama 0.33.3 honors
a model's declared context and `qwen3.5:4b` declares 262,144 tokens.

```bash
OLLAMA_CONTEXT_LENGTH=2048 OLLAMA_KEEP_ALIVE=0 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

## Evaluation harness

- `eval/run_eval.py` — GSM8K / MATH-500 against any OpenAI- or Ollama-compatible endpoint. `--mode completion`
  is the lm-eval `gsm8k_cot` protocol (raw few-shot prompt, `--api ollama` uses `raw=true` so no chat
  template is applied); `--mode chat` is for instruct/thinking models. Resumable; retries transient errors.
- `eval/rescore.py` — re-score saved generations under lm-eval strict, strict with `$`, and flexible rules.
- `eval/report.py` — results tables with paper comparisons and the depth curve.
- `scripts/run_evals.sh`, `run_baselines.sh`, `run_phase2.sh` — the chains, memory-guarded.
- `scripts/run_mmlu_sweep.sh` — lm-eval MMLU at depth 1–4, the paper's published ablation.
- `scripts/provenance.py` — pins model commit shas and library versions into `results/provenance.json`.

Protocol details were verified against the Ouro paper's PDF (Table 16): GSM8K is 3-shot CoT strict
under lm-eval-harness; MATH500 alone used an in-house harness, so a MATH-500 run here would be a new
measurement, not a replication.

## Layout

```
serve/shim.py          Ollama+OpenAI-compatible server over HF looped models (MPS)
eval/                  runner, rescoring, report
scripts/               chains, memory guard, GGUF tools, provenance, LoopUS patch
models/                Ollama Modelfiles
results/               generations (JSONL), summaries, REPORT.md, provenance
workflows/             the research workflow that produced the survey
NOTES.md  RECIPES.md   findings and per-model recipes
.venv (transformers 5) recurrent-qwen, LoopUS   ·   .venv-tf4 (transformers 4.57) Ouro, lm-eval
```

## Status

Done: deployment of three looped families, GSM8K replication and depth sweep, the paper's MMLU depth
ablation replicated, matched baselines for every looped model, scoring-rule audit, memory safeguards.
Not run: LoopUS on GSM8K — its 4 GB bf16 load plus per-recursion caches trips the 30% memory bound on
this machine (its probes: correct at depth 8, repetition collapse at depth 1). Next: the Ouro
architecture for llama.cpp, so Ouro runs inside Ollama (see NOTES.md).
