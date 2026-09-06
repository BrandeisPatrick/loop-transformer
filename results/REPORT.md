# looplm results (GSM8K)

Local numbers are on a fixed 200-problem subset (seed 0) unless n says otherwise; ± is one binomial standard error. Paper numbers are on the full 1319-problem test set.

| run | model | loops | n | acc | paper | trunc | err | s/item | tok/item | status |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_ouro-1.4b_3shot_T4 | ouro-1.4b | 4 | 43 | **76.7** ± 6.4 | 78.9 | 0 | 0 | 39.4 | 106 | running |
| gsm8k_qwen3-1.7b-base_3shot | hf.co/mradermacher/Qwen3-1.7B-Base-GGUF: |  | 200 | **68.0** ± 3.3 | 70.3 | 17 | 0 | 2.9 | 133 | done |

## Ouro-1.4B: accuracy vs recurrent steps (GSM8K 3-shot strict)

| loops (T) | n | acc | s/item |
|---|---|---|---|
| 4 | 43 | 76.7 ± 6.4 | 39.4 |

The paper only reports GSM8K at T=4 (78.92); its per-step ablation is on MMLU (41.21 / 60.43 / 66.71 / 67.45 at T=1..4).
