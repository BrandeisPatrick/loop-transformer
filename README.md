# looplm — looped transformers, locally, next to Ollama

Deploy **looped / recurrent-depth** language models on an Apple-Silicon Mac, expose them the way Ollama
exposes models, and replicate the published math evaluations against comparable base models.

Started 2026-09-05. Hardware: M4, 16 GB unified memory. Research notes: [NOTES.md](NOTES.md).

## What is running

| Track | Model | How it runs | Loop control |
|---|---|---|---|
| A. Native Ollama | **Nanbeige4.2-3B** (Nanbeige Lab, Apache-2.0, arXiv 2607.22083) — 22 layers × `num_loops=2`, 3B non-embedding params | `ollama pull hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M` (llama.cpp added the `nanbeige` looped arch in PR #25994; Ollama 0.33.3 bundles it) | fixed in GGUF metadata; `scripts/make_loop_variants.sh` writes copies with a different `nanbeige.num_loops` and registers them (`nanbeige4.2:loops1`) |
| B. Shim | **Ouro-1.4B** (ByteDance Seed, Apache-2.0, arXiv 2510.25741) — 24 layers × 4 recurrent steps, early-exit gate | `serve/shim.py --backend ouro` on torch/MPS in `.venv-tf4` (transformers 4.57; the model's remote code breaks on transformers 5) | per request: `num_loops` (sets `total_ut_steps`), `exit_threshold` |
| C. Qwen-derived | **recurrent-qwen2.5-0.5b** (Shapiro, arXiv 2608.11233) — Qwen2.5-0.5B-Instruct with layers 6-17 weight-tied and looped T times | `serve/shim.py --backend recurrent-qwen` in `.venv` | per request: `num_loops` → `generate(max_loops=T)` |

The shim speaks both APIs, so anything that talks to Ollama can talk to it:
`GET /api/tags · POST /api/chat · POST /api/generate · POST /api/show · GET /api/version` and
`GET /v1/models · POST /v1/chat/completions · POST /v1/completions`.

```bash
# Ollama baseline (port 11434)                    # looped model behind the shim (port 11435)
ollama serve                                       .venv-tf4/bin/python serve/shim.py --model ByteDance/Ouro-1.4B \
                                                       --backend ouro --name ouro-1.4b --loops 4 --port 11435
curl localhost:11435/api/chat -d '{"model":"ouro-1.4b","messages":[{"role":"user","content":"17+26?"}],"options":{"num_loops":2}}'
```

## Evaluation

`eval/run_eval.py` runs GSM8K / MATH-500 against any endpoint, in two protocols:

- `--mode completion --shots 3` — raw few-shot "Q:/A:" prompt, strict `The answer is N` match: the
  lm-evaluation-harness `gsm8k_cot` protocol the Ouro paper used for base models (`--api ollama` uses
  `/api/generate raw=true` so Ollama does not wrap the prompt in a chat template).
- `--mode chat` — chat template + "reason step by step, answer in `\boxed{}`", for instruct / thinking
  models (`--api ollama --think on|off` toggles Nanbeige/Qwen3 thinking).

`scripts/run_evals.sh` chains the runs; every run is resumable; `eval/summarize.py` prints a table of
`results/*.summary.json`. Published numbers we are replicating:

| Model | Benchmark (setting) | Paper |
|---|---|---|
| Ouro-1.4B, T=4 | GSM8K 3-shot CoT strict | 78.92 |
| Qwen3-1.7B-Base | GSM8K 3-shot CoT strict | 70.28 |
| Ouro-1.4B, T=1/2/3/4 | MMLU 5-shot | 41.21 / 60.43 / 66.71 / 67.45 |
| Nanbeige4.2-3B-Base | GSM8K (setting unpublished) | 92.7 (Qwen3.5-4B-Base 84.4) |

## Layout

```
serve/shim.py            Ollama+OpenAI compatible server over HF looped models (MPS)
eval/run_eval.py         GSM8K / MATH-500 runner (chat or few-shot completion; OpenAI or Ollama API)
eval/summarize.py        table of results/*.summary.json
scripts/run_evals.sh     the eval chain (resumable)
scripts/make_loop_variants.sh   GGUF num_loops rewrite -> ollama create
scripts/gguf_header.py   read GGUF metadata from a file or URL (range request)
scripts/test_ouro.py     Ouro MPS smoke test at several loop counts
workflows/               research workflow script (Claude Code Workflow tool)
results/                 eval outputs (jsonl + summary.json)
.venv (transformers 5)   recurrent-qwen, LoopUS; .venv-tf4 (transformers 4.57) Ouro, lm-eval
```
