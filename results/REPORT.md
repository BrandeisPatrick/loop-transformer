# looplm results (GSM8K)

Local numbers are on a fixed 200-problem subset (seed 0) unless n says otherwise; ± is one binomial standard error. Paper numbers are on the full 1319-problem test set.

| run | model | loops | n | acc | paper | trunc | err | s/item | tok/item | status |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_nanbeige4.2-3b_loops1_nothink | nanbeige4.2:loops1 |  | 30 | **0.0** ± 0.0 | — | 30 | 0 | 7.1 | 256 | done |
| gsm8k_nanbeige4.2-3b_loops2_nothink | hf.co/bartowski/Nanbeige_Nanbeige4.2-3B- |  | 201 | **91.0** ± 2.0 | 92.7 | 1 | 1 | 15.7 | 314 | done |
| gsm8k_nanbeige4.2-3b_loops2_think | hf.co/bartowski/Nanbeige_Nanbeige4.2-3B- |  | 50 | **92.0** ± 3.8 | 92.7 | 3 | 0 | 64.4 | 1264 | done |
| gsm8k_ouro-1.4b_3shot_T1 | ouro-1.4b | 1 | 100 | **23.0** ± 4.2 | — | 11 | 0 | 11.1 | 84 | done |
| gsm8k_ouro-1.4b_3shot_T2 | ouro-1.4b | 2 | 100 | **64.0** ± 4.8 | — | 0 | 0 | 19.3 | 100 | done |
| gsm8k_ouro-1.4b_3shot_T3 | ouro-1.4b | 3 | 100 | **72.0** ± 4.5 | — | 0 | 0 | 28.2 | 104 | done |
| gsm8k_ouro-1.4b_3shot_T4 | ouro-1.4b | 4 | 200 | **80.0** ± 2.8 | 78.9 | 2 | 0 | 38.6 | 106 | done |
| gsm8k_ouro-1.4b_llamacpp_F16_T4 | ouro-1.4b-gguf |  | 200 | **80.5** ± 2.8 | — | 2 | 0 | 11.3 | 106 | done |
| gsm8k_ouro-1.4b_llamacpp_L1 | ouro-gguf-L1 |  | 100 | **26.0** ± 4.4 | — | 13 | 0 | 2.4 | 88 | done |
| gsm8k_ouro-1.4b_llamacpp_L2 | ouro-gguf-L2 |  | 100 | **67.0** ± 4.7 | — | 0 | 0 | 5.5 | 102 | done |
| gsm8k_ouro-1.4b_llamacpp_Q4KM_L1 | ouro-q4-L1 |  | 100 | **20.0** ± 4.0 | — | 10 | 0 | 0.9 | 77 | done |
| gsm8k_ouro-1.4b_llamacpp_Q4KM_L4 | ouro-1.4b-q4 |  | 100 | **75.0** ± 4.3 | — | 0 | 0 | 4.5 | 102 | done |
| gsm8k_ouro-1.4b_llamacpp_Q8_L4 | ouro-q8 |  | 99 | **79.8** ± 4.0 | — | 0 | 0 | 7.3 | 102 | running |
| gsm8k_qwen3-1.7b-base_3shot | hf.co/mradermacher/Qwen3-1.7B-Base-GGUF: |  | 200 | **68.0** ± 3.3 | 70.3 | 17 | 0 | 2.9 | 133 | done |
| gsm8k_qwen3-1.7b_nothink | qwen3:1.7b |  | 200 | **82.5** ± 2.7 | — | 1 | 0 | 4.2 | 292 | done |
| gsm8k_qwen3-4b-base_3shot | hf.co/mradermacher/Qwen3-4B-Base-GGUF:Q4 |  | 200 | **76.5** ± 3.0 | 72.9 | 10 | 0 | 4.0 | 135 | done |
| gsm8k_qwen3.5-4b_nothink | qwen3.5:4b |  | 200 | **93.5** ± 1.7 | — | 8 | 0 | 17.5 | 420 | done |

## Same generations, four extraction rules

See NOTES.md §8: lm-eval's verbatim strict-match cannot capture a `$`-prefixed answer, which penalises the two models unequally.

| run | as-run | lm-eval strict | strict with `$` | lm-eval flexible | `$`-formatted |
|---|---|---|---|---|---|
| gsm8k_ouro-1.4b_3shot_T1 | 23.0 | 20.0 | 23.0 | 23.0 | 10.0% |
| gsm8k_ouro-1.4b_3shot_T2 | 64.0 | 52.0 | 64.0 | 67.0 | 16.0% |
| gsm8k_ouro-1.4b_3shot_T3 | 72.0 | 59.0 | 72.0 | 72.0 | 17.0% |
| gsm8k_ouro-1.4b_3shot_T4 | 80.0 | 60.5 | 79.0 | 80.0 | 23.0% |
| gsm8k_qwen3-1.7b-base_3shot | 68.0 | 59.0 | 65.5 | 71.5 | 9.5% |
| gsm8k_qwen3-4b-base_3shot | 76.5 | 67.0 | 74.0 | 81.0 | 11.0% |

## Ouro-1.4B: accuracy vs recurrent steps (GSM8K 3-shot strict)

| loops (T) | n | acc | truncated | acc on non-truncated | s/item |
|---|---|---|---|---|---|
| 1 | 100 | 23.0 ± 4.2 | 11 | 25.8 (n=89) | 11.1 |
| 2 | 100 | 64.0 ± 4.8 | 0 | 64.0 (n=100) | 19.3 |
| 3 | 100 | 72.0 ± 4.5 | 0 | 72.0 (n=100) | 28.2 |
| 4 | 200 | 80.0 ± 2.8 | 2 | 80.8 (n=198) | 38.6 |

Truncation matters at low depth: a run that hits the 256-token cap never emits "The answer is N" and is scored wrong. The last column removes those, separating "reasoned badly" from "never finished".

The paper only reports GSM8K at T=4 (78.92); its per-step ablation is on MMLU (41.21 / 60.43 / 66.71 / 67.45 at T=1..4).

## The llama.cpp port, against the transformers reference

Same 3-shot protocol and same problems, F16 GGUF on Metal vs bf16 transformers on MPS. Depth is set at load time with `--override-kv ouro.num_loops=int:N` from a single file.

| loops | llama.cpp | transformers | n | s/item (llama.cpp) | s/item (transformers) | speedup |
|---|---|---|---|---|---|---|
| 1 | **20.0** | 23.0 | 100 | 0.9 | 11.1 | 11.8x |
| 1 | **26.0** | 23.0 | 100 | 2.4 | 11.1 | 4.5x |
| 2 | **67.0** | 64.0 | 100 | 5.5 | 19.3 | 3.5x |
| 4 | **75.0** | 80.0 | 100 | 4.5 | 38.6 | 8.6x |
| 4 | **79.8** | 80.0 | 99 | 7.3 | 38.6 | 5.3x |
| 4 | **80.5** | 80.0 | 200 | 11.3 | 38.6 | 3.4x |

## Ouro-1.4B: MMLU 5-shot vs recurrent steps (lm-eval, the paper's published ablation)

~3% of each subject (all 57 subjects, ~420 questions), log-likelihood scoring, no chat template. Paper numbers are Table 10 on the full set; ± is lm-eval's reported standard error.

| loops (T) | MMLU (ours) | paper | humanities | other | social sci | STEM |
|---|---|---|---|---|---|---|
| 1 | **39.0** ± 2.2 | 41.21 | 31.1 | 47.5 | 43.9 | 37.5 |
| 2 | **58.4** ± 2.3 | 60.43 | 50.0 | 61.6 | 67.3 | 58.7 |
| 4 | **67.3** ± 2.2 | 67.45 | 63.5 | 64.6 | 77.6 | 65.4 |
