# looplm results (GSM8K)

Local numbers are on a fixed 200-problem subset (seed 0) unless n says otherwise; ± is one binomial standard error. Paper numbers are on the full 1319-problem test set.

| run | model | loops | n | acc | paper | trunc | err | s/item | tok/item | status |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_ouro-1.4b_3shot_T1 | ouro-1.4b | 1 | 100 | **23.0** ± 4.2 | — | 11 | 0 | 11.1 | 84 | done |
| gsm8k_ouro-1.4b_3shot_T2 | ouro-1.4b | 2 | 100 | **64.0** ± 4.8 | — | 0 | 0 | 19.3 | 100 | done |
| gsm8k_ouro-1.4b_3shot_T3 | ouro-1.4b | 3 | 2 | **50.0** ± 35.4 | — | 0 | 0 | 27.9 | 80 | running |
| gsm8k_ouro-1.4b_3shot_T4 | ouro-1.4b | 4 | 200 | **80.0** ± 2.8 | 78.9 | 2 | 0 | 38.6 | 106 | done |
| gsm8k_qwen3-1.7b-base_3shot | hf.co/mradermacher/Qwen3-1.7B-Base-GGUF: |  | 200 | **68.0** ± 3.3 | 70.3 | 17 | 0 | 2.9 | 133 | done |

## Same generations, four extraction rules

See NOTES.md §8: lm-eval's verbatim strict-match cannot capture a `$`-prefixed answer, which penalises the two models unequally.

| run | as-run | lm-eval strict | strict with `$` | lm-eval flexible | `$`-formatted |
|---|---|---|---|---|---|
| gsm8k_ouro-1.4b_3shot_T1 | 23.0 | 20.0 | 23.0 | 23.0 | 10.0% |
| gsm8k_ouro-1.4b_3shot_T2 | 63.4 | 51.2 | 63.4 | 63.4 | 17.1% |
| gsm8k_ouro-1.4b_3shot_T4 | 80.0 | 60.5 | 79.0 | 80.0 | 23.0% |
| gsm8k_qwen3-1.7b-base_3shot | 68.0 | 59.0 | 65.5 | 71.5 | 9.5% |

## Ouro-1.4B: accuracy vs recurrent steps (GSM8K 3-shot strict)

| loops (T) | n | acc | truncated | acc on non-truncated | s/item |
|---|---|---|---|---|---|
| 1 | 100 | 23.0 ± 4.2 | 11 | 25.8 (n=89) | 11.1 |
| 2 | 100 | 64.0 ± 4.8 | 0 | 64.0 (n=100) | 19.3 |
| 3 | 2 | 50.0 ± 35.4 | 0 | 50.0 (n=2) | 27.9 |
| 4 | 200 | 80.0 ± 2.8 | 2 | 80.8 (n=198) | 38.6 |

Truncation matters at low depth: a run that hits the 256-token cap never emits "The answer is N" and is scored wrong. The last column removes those, separating "reasoned badly" from "never finished".

The paper only reports GSM8K at T=4 (78.92); its per-step ablation is on MMLU (41.21 / 60.43 / 66.71 / 67.45 at T=1..4).
