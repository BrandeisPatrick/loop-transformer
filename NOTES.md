# NOTES — looped transformers: research state (2026-09-06)

Compiled from a 5-agent web sweep (all claims carry source URLs in `workflows/` transcripts) plus hands-on verification on this Mac. See README.md for what is deployed.

## 1. Field trends

SCOPE AND METHOD. All claims above were pulled live from arXiv abstract/HTML pages, Hugging Face model cards and config.json files, GitHub READMEs/PRs (via gh), and the curated list github.com/huskydoge/Awesome-Loop-Models (~152 entries, which I used to find Aug-Sep 2026 papers). The 2026 wave is large: roughly 80 loop-model papers dated 2026 on that list, so the 36 entries here are the ones with weights/code or that set the design consensus.

WHAT THE FIELD IS CONVERGING ON.
1) Which blocks loop. Two topologies dominate. (a) Whole-stack looping with deep supervision at every loop: Ouro (48 layers x4, https://arxiv.org/html/2510.25741) and Nanbeige4.2 (22 layers x2, config num_loops=2). (b) The 'sandwich' prelude / weight-tied core / coda: Huginn (2,4,2), Parcae (8/8/8), the UMD retrofitting checkpoints, Gated Recurrent Transformers (https://arxiv.org/abs/2608.15062), Hyperloop (only middle block looped), and the IFM 'Looped Models Done Right' ablation, which explicitly picks the Huginn-style sandwich for its 8B MoE. Retrofit-style work loops a contiguous MIDDLE block (ETD, Training-Free Looped Transformers, SMELT loops 'the middle half twice', LoopMDM loops early-middle layers). MixerLoop (https://arxiv.org/abs/2608.18230) goes further and loops only the attention/mixer sublayers, cutting recurrent projection FLOPs 45.9%.
2) Loop counts. Production has settled on 2. Nanbeige4.2 found extra passes gave little and cost training speed; LoopCoder-v2 shows two loops lift SWE-bench 43.0->64.4 while R=3 collapses to 27.6 (https://arxiv.org/abs/2606.18023); SMELT and IQuest-Coder-40B-Loop also use 2. Ouro trains at 4 and its own per-step curve saturates at T=4 (1.4B MMLU 41.2/60.4/66.7/67.5) and degrades at 5-8. Huginn trained with mean 32 but the Jacobian-lens paper finds it effectively uses a window of ~2 recurrences (https://arxiv.org/abs/2609.01924). Academic stability papers push to 8-15 loops (Parcae 8, Fully Looped 12, LoopMTP 15) but the iso-depth scaling law says each loop is worth phi=0.46 of a real layer at full-layer training cost (https://arxiv.org/abs/2604.21106).
3) Adaptive exit. Ouro's entropy-regularized learned gate + Q-exit threshold is the only shipped learned-exit mechanism, and two independent 2026 analyses (Diagnosing Halting Gates https://arxiv.org/abs/2607.20519; Huginn's zero-shot KL/entropy criteria) conclude that simple confidence/KL readouts on a well-trained trajectory match learned gates. Serving support for exit is the bottleneck: vLLM runs Ouro at full total_ut_steps, and Continuous Depth Batching (https://arxiv.org/abs/2608.09444, code released) is the first scheduler that recovers ~99% of the adaptive-depth speed-up. MoR-style per-token routing and CHASE for looped Mamba (https://arxiv.org/abs/2607.10110) are the other exit families.
4) KV sharing. Consensus: do not keep a KV cache per loop. Ouro 'last-step-only' KV gives 4x memory at <0.5% loss; PLT / LoopCoder-v2 use gated sliding-window attention with shared KV across loops (window 64); Huginn cycles a fixed KV budget (compress-s16); MoR caches only active tokens per recursion; llama.cpp's nanbeige.cpp instead allocates a KV slot per loop (memory grows with loops) so a 2-loop model is the practical cap there.
5) Training recipes. 2026's dominant theme is STABILITY: residual explosion and gradient oscillation appear once loops >4. Fixes converge on (i) scaling the recurrent residual by 1/N loops (https://arxiv.org/abs/2606.18524) or loop-aware DeepNorm (DeepLoop), (ii) spectral-norm-constrained input injection (Parcae), (iii) parameter-free signal distribution across layers (Fully Looped Transformer), (iv) fixed-point/pre-norm formulations (FPRM, Attractor Models), (v) hyper-connections / matrix residual streams (Hyperloop, Nanbeige 'mHC', iso-depth phi 0.46->0.65). Deep supervision at every loop is standard (Ouro, LOTUS supervising loops with CoT steps, LoopMTP supervising loop t with the token t steps ahead, LoopRPT adding RL on intermediate states); the Readout Blind Spot paper warns dense CE does not control hidden-state scale and norms can blow into the thousands (https://arxiv.org/abs/2606.24898). Random/stochastic loop counts during training improve extrapolation (Huginn, Stochastic Stopping https://arxiv.org/abs/2606.29983); truncated BPTT hurts (phi 0.46->0.38). Scaling laws now exist (Parcae: scale loops and data together; SMELT: 6.8-18% FLOP savings at up to 54B). Loop + MoE is the 2026 frontier (Loopie, LoopMoE, SMELT, IFM), and press reports say OpenAI's Astra uses recurrent depth with capped loops (Fortune, 2026-09-03: https://fortune.com/2026/09/03/reports-openais-astra-model-uses-a-new-more-efficient-ai-architecture-alarms-ai-safety-experts-who-worry-the-method-makes-models-harder-to-control/ ; secondary source, not a paper).
6) Retrofit vs scratch. A clear sub-track converts existing dense models: curriculum retrofitting (UMD, Llama/OLMo/TinyLlama 1B checkpoints released), ETD (OLMo-2 1B, +28% GSM8K), LoopUS, PoLar, Shapiro's 6M-param adapter on Qwen2.5-0.5B, and fully training-free mid-block looping on Qwen3-4B (+2.64pp MMLU-Pro). Nobody has released a looped Qwen checkpoint; Qwen-based looping so far is either training-free or small-scale.

LOCAL-DEPLOY REALITY CHECK (M4, 16 GB, 3.5 GB free, Ollama). Stock llama.cpp master supports exactly ONE looped architecture: LLM_ARCH_NANBEIGE with LLM_KV_NUM_LOOPS (src/llama-arch.h; src/models/nanbeige.cpp shares weights across loops; PR #25994 merged 2026-07-27). There is no Ouro, Huginn, Parcae or IQuest/PLT arch in llama.cpp, so no working GGUF path for them. mlx-lm main natively supports Nanbeige (PR #1597 merged 2026-08-29) and iquestloopcoder (2-loop PLT); Ouro support is an OPEN PR #599 (89 tok/s, ~1 GB peak for 1.4B-4bit) that you must install from the branch. Ollama's own registry has no looped model; the only Ollama route is a GGUF of Nanbeige4.2-3B (Abiray Q4_K_M 2.57 GB), with the caveat that Nanbeige's README still points to its own Ollama fork.

WHAT TO REPLICATE FIRST (in order).
1. Nanbeige4.2-3B (Q4_K_M via Ollama or mlx-lm) vs Qwen3.5-4B served by Ollama: replicate the base-model GSM8K 92.7 vs 84.4 and MMLU-Pro 63.8 vs 51.8 gaps, then MATH500/GPQA. Fits disk now. Caveat: it is fixed 2-pass, so there is no loop-count knob to sweep.
2. Ouro-1.4B/2.6B via the mlx-lm PR #599 branch (mlx_lm.server = OpenAI-compatible endpoint beside Ollama) vs Qwen3-1.7B/4B in Ollama: sweep total_ut_steps 1..4 and early_exit_threshold to reproduce the T=1..4 accuracy curve and the GSM8K 78.9 / MATH500 82.4 (1.4B) numbers from the paper HTML; use OuroTrace or lm-eval. Ouro-1.4B-4bit (~0.9 GB) fits disk; 2.6B-4bit is 1.5 GB.
3. Training-Free Looped Transformers on Qwen3-1.7B/4B-Instruct: implement damped mid-block sub-steps in mlx-lm or transformers and target +2.64pp MMLU-Pro. No training, no new weights, directly gives 'a looped model built on Qwen'. PoLar (code released) is the alternative with a tiny learned router.
4. If you free ~10 GB: Huginn-0125 in transformers on MPS (bf16 ~7 GB) sweeping num_steps 4/8/16/32 on GSM8K (38->42 flexible) as the canonical test-time-depth curve; and the UMD retrofitting scripts on Qwen3-0.6B at a small token budget if you want to train.

COULD NOT CONFIRM. (a) An official Ouro training/inference repo (project page says 'Coming Soon'; ByteDance/Ouro and ByteDance-Seed/Ouro 404). (b) Whether stock Ollama's bundled llama.cpp already includes the nanbeige arch, versus needing the Nanbeige ollama fork. (c) Whether mlx-lm's iquestloopcoder.py loads LoopCoder-V2 (model_type iquestpltcoder). (d) The arXiv ID of 'LoopCoder: Scaling Code Intelligence via Looped Language Models' (the HF card's 2512.22087 resolves to an unrelated paper). (e) A URL for the IFM 'Looped Models Done Right' report and for LoopCD (github.com/hoeng4/LoopCD exists, EMNLP 2026, but no arXiv found). (f) Code links for LoopUS, Attractor Models, Fully Looped Transformer and Loop-Think-Generalize (Awesome-list badges only). (g) Affiliations for most 2026 papers (arXiv abs pages omit them; those listed are inferred and marked). (h) The Ouro-2.6B model card summary said '24 layers' but config.json says 48 (paper: 1.4B=24, 2.6B=48); treat config.json as authoritative. (i) Loopie weights: none linked on HF as of 2026-09-05.

## 2. Papers (2025 → Sep 2026), weights-first

- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn-0125)** (University of Maryland (Goldstein/Geiping) + LLNL, 2025-02-07 (v2 2025-02-17)) — **weights** — HF: https://huggingface.co/tomg-group-umd/huginn-0125 (Apache-2.0; dataset https://huggingface.co/datasets/tomg-group-umd/huginn-dataset; X: https://x.com/jonasgeiping/status/1907463318530343042). https://arxiv.org/abs/2502.05171
  3.5B model, (2,4,2) prelude/recurrent/coda layers, hidden 5280, 800B tokens on 4096 AMD MI250X; recurrence sampled per step from log-normal Poisson with mean 32, truncated backprop k=8. Paper HTML (https://arxiv.org/html/2502.05171v2): at r=32 GSM8K strict 38.13 / flexible 42.08 (OLMo-2-1B 66.79), MATH 11.24 (19.08), ARC-C 38.23, MMLU 31.38, HellaSwag 65.21, HumanEval 23.17, MBPP 24.80; easy tasks saturate ~8 iterations, GSM8K keeps improving to 32+. Zero-shot adaptive exit via KL between successive steps (threshold 5e-4), KV-cache sharing by modulo over a fixed budget (compress-s16), continuous CoT warm-start. HF card: materialized params ~ num_steps*1.5B + 2B, min 4 steps for coherence, gains to ~64.

- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** (KAIST + Mila + Google (Bae et al.), 2025-07-14) — **weights** — HF: none (Google Drive links in repo). https://arxiv.org/abs/2507.10524
  Shared layer stack with lightweight routers assigning per-token recursion depth; recursion-wise KV caching only for active tokens. Equal-compute: 43.1% vs 42.3% few-shot with ~50% fewer params; equal-data: 19% less training time, 25% less peak memory; up to 2x inference throughput. Repo has expert-choice and token-choice routers; 360M checkpoints on Google Drive (not HF).

- **Scaling Latent Reasoning via Looped Language Models (Ouro)** (ByteDance Seed + UC Santa Cruz + Princeton et al. (33 authors; affiliations per HF search snippet, not on abs page), 2025-10-29 (v5 2026-07-01)) — **weights** — HF: https://huggingface.co/ByteDance/Ouro-2.6B (also Ouro-1.4B, Ouro-1.4B-Thinking, Ouro-2.6B-Thinking; MLX 4-bit: https://huggingface.co/mlx-community/Ouro-2.6B-4bit). https://arxiv.org/abs/2510.25741
  1.4B/2.6B LoopLMs pretrained on 7.7T tokens with all transformer blocks looped 4x (R4), entropy-regularized learned exit gate + Q-exit inference. Paper HTML tables (https://arxiv.org/html/2510.25741): Ouro-2.6B MMLU 74.60 vs Qwen3-8B 76.63; MMLU-Pro 55.73 vs 53.72; BBH 80.46 vs 77.65; GSM8K 81.58 vs 83.09; MATH500 90.85 vs 62.30. Ouro-1.4B: MMLU 67.35 vs Qwen3-4B 73.19; GSM8K 78.92 vs 72.86; MATH500 82.40 vs 59.60. Thinking-R4: AIME24 65.0 (1.4B) / 64.7 (2.6B). Per-step 1.4B MMLU: T=1 41.21, T=2 60.43, T=3 66.71, T=4 67.45, degrades at T=5-8. 'Last-step-only' KV cache = 4x memory reduction at <0.5% loss (GSM8K 78.85 vs 78.92). Config.json (https://huggingface.co/ByteDance/Ouro-2.6B/raw/main/config.json): model_type ouro, 48 layers, hidden 2048, vocab 49152, total_ut_steps 4, early_exit_threshold 1.0, requires trust_remote_code and transformers<4.56 (rec. 4.54.1). X thread: https://x.com/RidgerZhu/status/1983732551404679632

- **Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence** (University of Maryland (McLeish, Kirchenbauer, Geiping, Goldstein, Goldblum) + LLNL, 2025-11-10) — **weights** — HF: https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence (dataset smcleish/retrofitting-llama-fineweb-edu-tokenized). https://arxiv.org/abs/2511.07384
  Converts existing non-recurrent LMs into depth-recurrent ones; a curriculum of increasing recurrences preserves performance while reducing total compute; on math tasks the converted recurrent model beats post-training the original at equal compute. Released checkpoints: Recurrent-Llama-3.2 (1B), Recurrent-TinyLlama-3T (~1.1B), Recurrent-OLMo-2-0425 (1B) each with train-recurrence 4/8/16/32, plus untrained and non-recurrent-posttrained controls (collection page). Repo pins transformers==4.51.0 for KV-cache compatibility; conversion scripts in convert_pretrained_model/.

- **IQuest-Coder-V1 (40B-Loop-Instruct / 40B-Loop-Thinking) and the cited 'LoopCoder: Scaling Code Intelligence via Looped Language Models'** (IQuest Lab, 2025-12 (HF blog timestamp 2026-03-02)) — **weights** — HF: https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct ; AWQ 4-bit community: https://huggingface.co/cyankiwi/IQuest-Coder-V1-40B-Loop-Instruct-AWQ-4bit. https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Thinking
  40B, 80 effective layers = 2 iterations over shared weights; second iteration refines first-iteration hidden states via a global-local attention gate. Card claims SWE-Bench Verified 76.2%, BigCodeBench 49.9%, LiveCodeBench v6 81.1%. Transformers >=4.52.4 + trust_remote_code; vLLM with qwen3_coder tool parser. Custom 'iquestcoder' license.

- **Parcae: Scaling Laws For Stable Looped Language Models** (UC San Diego Sandy Research (Daniel Y. Fu) with Together AI blog (https://www.together.ai/blog/parcae), 2026-04-14) — **weights** — HF: https://huggingface.co/SandyResearch/parcae-1.3b (plus parcae-770m, -370m, -140m; https://huggingface.co/SandyResearch). https://arxiv.org/abs/2604.12946
  Recasts looping as a nonlinear time-variant dynamical system and constrains spectral norm of injection parameters via a discretized negative-diagonal parameterization. Up to 6.3% lower val perplexity than prior looped models; +2.99 CORE / +1.18 Core-Extended at 1.3B vs Transformer; 87.5% relative quality of a Transformer twice the size; first looped scaling laws: compute-optimal training scales loops and data in tandem. parcae-1.3b card: 8 prelude / 8 core / 8 coda layers, dim 1536, recurrence 8, FineWeb-Edu.

- **HRM-Text: Efficient Pretraining Beyond Scaling** (Sapient Intelligence (Guan Wang; X: https://x.com/makingAGI/status/2057065550682235020), 2026-05-20) — **weights** — HF: https://huggingface.co/sapientinc/HRM-Text-1B (HF transformers docs page model_doc/hrm_text exists). https://arxiv.org/abs/2605.20613
  Hierarchical Recurrent Model (slow H / fast L modules iterated over the same input, additive state injection) trained on 40B unique tokens for ~$1,500: MMLU 60.7, ARC-C 81.9, DROP 82.2, GSM8K 84.5, MATH 56.2 at 1B params; claims 100-900x fewer tokens than 2-7B baselines. Adjacent to looped transformers (dual-timescale recurrence rather than a single looped stack).

- **LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling** (IQuest Lab (Bryan Dai) + Beihang + Renmin + others (affiliations not on abs page), 2026-06-16) — **weights** — HF: https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2 (Apache-2.0). https://arxiv.org/abs/2606.18023
  Family of 7B Parallel-Loop-Transformer coders trained from scratch on 18T tokens with different loop counts; gain-cost framework shows two loops are optimal, 3+ loops regress (R=3 SWE-bench 27.6). Two-loop vs one-loop: SWE-bench Verified 43.0->64.4, Multi-SWE 14.0->31.0, Terminal-Bench 11.2->21.0, BFCL 32.2->40.1, HumanEval+ 81.1->84.1, LiveCodeBench 27.4->35.4. HF config: model_type iquestpltcoder, 14 shared layers, hidden 5120, plt_num_loops 2, plt_window_size [64,0], transformers 4.57.1.

- **Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model** (Nanbeige Lab (Kanzhun / BOSS Zhipin; contact nanbeige@kanzhun.com on HF card), 2026-07-24 (v2 2026-07-27)) — **weights** — HF: https://huggingface.co/Nanbeige/Nanbeige4.2-3B ; GGUF: https://huggingface.co/Abiray/Nanbeige4.2-3B-GGUF (Q4_K_M 2.57 GB ... Q8_0 4.43 GB), https://huggingface.co/WaveCut/Nanbeige4.2-3B-heretic-GGUF. https://arxiv.org/abs/2607.22083
  Looped Transformer: 22 physical layers executed twice (config.json: model_type nanbeige, num_hidden_layers 22, num_loops 2, skip_loop_final_norm false, hidden 3072, vocab 166144; https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/config.json). 3B non-embedding / 4B total params, 28T tokens; HF card also lists 'LoopSplit, mHC with depth attention, concatenated n-gram embeddings'. Base model (https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base): GSM8K 92.7 vs Qwen3.5-4B-Base 84.4; BBH 81.6 vs 79.1; MBPP 67.6 vs 57.1; MMLU-Pro 63.8 vs 51.8. Instruct (https://huggingface.co/Nanbeige/Nanbeige4.2-3B): GPQA-Diamond 87.4 vs Qwen3.5-9B 81.7; HMMT-Feb-2026 82.8 vs 69.6; SWE-bench Verified 63.6 vs 53.1; LiveCodeBench-v6 72.5. Raschka's architecture gallery says 2 passes retained ~75% token efficiency and more passes gave little benefit (https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/).

- **Reasoning with Latent Thoughts: On the Power of Looped Transformers** (Google Research (Saunshi et al.), 2025-02-24 (ICLR 2025)) — no weights. https://arxiv.org/abs/2502.17416
  A k-layer transformer looped L times nearly matches a kL-layer model on addition, p-hop induction and math, far above a k-layer model; looped models have an inductive bias toward reasoning even with worse perplexity — the theoretical anchor most 2026 papers cite.

- **Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts (ETD)** (Meta FAIR + University College London, 2025-10-08) — no weights. https://arxiv.org/abs/2510.07358
  Mid-training stage teaches the model to iterate over a small subset of reasoning-relevant middle layers; architecture/params unchanged. OLMo-2 1B Base: +28.4% relative on GSM8K, +36% on MATH; gains across 17 reasoning benchmarks by iterating selected layers at inference.

- **Parallel Loop Transformer for Efficient Test-Time Computation Scaling (PLT)** (ByteDance Seed (Bohong Wu et al.; affiliation not on abs page), 2025-10-28) — no weights. https://arxiv.org/abs/2510.24824
  Cross-Loop Parallelism computes different loops for different tokens in one pass (breaks sequential loop latency); Gated Sliding-Window Attention shares KV cache across loops so memory stays near non-looped cost. Claims accuracy of a deep looped model with almost no extra latency or memory. This is the architecture used by LoopCoder-v2 and IQuest-Coder-V1-Loop.

- **MoDR: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning** (Zhang, Wu, He, Shen, Lyu, Zhu (OpenReview; affiliations not captured), 2026 (ICLR 2026 poster, Apr 23 2026)) — no weights. https://iclr.cc/virtual/2026/poster/10011117
  Fine-tunes Huginn into N recurrent branches (shared recurrent block + per-branch LoRA) with hard-gate routing; +7.2% avg over original Huginn and +2.48% over a fine-tuned Huginn on math benchmarks; +21.21% / +1.52% on commonsense.

- **Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves (Dreamer)** (TUM / Bosch / TU Darmstadt (Knupp, Metzen, Bohn, Groh, Kersting; inferred), 2026-01-29) — no weights. https://arxiv.org/abs/2601.21582
  Combines sequence attention, depth attention (attending over previous loop states to beat the hidden-size bottleneck) and sparse expert attention; 2-8x fewer training tokens for same accuracy vs FLOP/param/memory-matched SOTA, beats ~2x larger models at equal tokens; 2-11x higher expert-selection diversity.

- **LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation** (University of Toronto / Vector (Jeddi, Ciccone, Taati; inferred), 2026-02-11) — no weights. https://arxiv.org/abs/2602.11451
  Conditions each loop on current time and step size and trains with shortcut-consistency so trajectories of different lengths agree, giving elastic depth under variable compute budgets on LM and reasoning benchmarks (ICLR 2026 submission).

- **Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training** (IMT Atlantique + Sony (Lys, Gripon, Pasdeloup, Marmoret, Mauch, Cardinaux, Boukli Hacene; affiliations inferred from author history, not on abs page), 2026-02-16 (v2 2026-03-02)) — no weights. https://arxiv.org/abs/2602.14759
  Re-applies a selected block range of off-the-shelf pretrained LMs at inference; reports 'modest but consistent' accuracy gains and latent-trajectory analysis showing more stable state evolution. Base models not stated on abs page.

- **LoopRPT: Reinforcement Pre-Training for Looped Language Models** (Harbin Institute of Technology (Bing Qin, Ming Liu group; affiliation inferred), 2026-03-20) — no weights. https://arxiv.org/abs/2603.19714
  Reframes next-token prediction as next-token reasoning with RL signals on intermediate loop representations using an EMA teacher and noisy latent rollouts; on the Ouro architecture across scales, improves per-step representation quality and is Pareto-dominant in accuracy vs compute, with largest gains on hard tokens.

- **Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers** (The Ohio State University, 2026-04-09 (v2 2026-08-11; COLM 2026)) — no weights. https://arxiv.org/abs/2604.07822
  Recurrent-depth transformers achieve systematic generalization via a three-stage grokking (memorization -> ID generalization -> systematic) and depth extrapolation (train 5-hop, test 10-hop) by increasing inference recurrence, limited by 'overthinking' at very deep recurrence.

- **A Mechanistic Analysis of Looped Reasoning Language Models** (Oxford (Bronstein, Dong) + Mila (Courville) + Google DeepMind (Castro) (inferred), 2026-04-13) — no weights. https://arxiv.org/abs/2604.11791
  Each layer in the cycle converges to its own fixed point; recurrent blocks learn inference stages mirroring feedforward models; studies how block size, input injection and normalization govern cyclic fixed-point emergence (39 pages).

- **How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models** (TUM (Schwethelm, Rueckert, Kaissis), 2026-04-22 (v3 2026-05-07)) — no weights. https://arxiv.org/abs/2604.21106
  Iso-depth pretraining (fixed 20-layer unrolled depth: 2 prelude + 16/r recurrent x r + 2 coda) over r in {1,2,4,8} and ~50x compute; recurrence-equivalence exponent phi = 0.46 (1 = a loop is worth a layer, 0 = no gain); truncated BPTT lowers phi to 0.38, hyperconnections raise it to 0.65; at r=4 a 410M looped model matches a 580M non-looped one but costs as much to train as 1B.

- **Hyperloop Transformers** (MIT (Zeitoun, Torroba-Hennigen, Yoon Kim), 2026-04-23 (v3 2026-07-02)) — no weights. https://arxiv.org/abs/2604.21254
  Begin/middle/end blocks with only the middle looped, plus hyper-connections giving matrix-valued residual streams; matches depth-matched Transformer and mHC Transformer baselines with ~50% fewer parameters; gains survive post-training weight quantization.

- **LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models** (Taekhyun Park, Yongjae Lee, Dohee Kim, Hyerim Bae (affiliation not on abs page), 2026-05-10) — no weights. https://arxiv.org/abs/2605.11011
  Post-training framework decomposing a pretrained LLM into encoder / looped reasoning block / decoder: block decomposition guided by representation dynamics, input-dependent selective gate against hidden-state drift, random deep supervision, confidence head for adaptive early exit. Claims improved reasoning and perplexity across scales with limited training budgets. Project page https://thrillcrazyer.github.io/LoopUS

- **Simply Stabilizing the Loop via Fully Looped Transformer** (Jilin University (Rao Fu ... Yi Chang; affiliation inferred), 2026-05-11 (v2 2026-05-25)) — no weights. https://arxiv.org/abs/2605.18797
  Identifies gradient oscillation and residual explosion as the two instability sources; two parameter-free fixes (Fully Looped Architecture distributing inter-loop signals across layers, Attention Injection reusing attention blocks) allow stable training to 12 loop iterations where baselines collapse, +13.2% average downstream. Uses the Qwen3 tokenizer and muP LR transfer across depths.

- **Training-Free Looped Transformers** (Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, Qiang Liu (UT Austin / Google per author history; affiliations not on abs page), 2026-05-22) — no weights. https://arxiv.org/abs/2605.23872
  Inference-time wrapper loops a contiguous mid-stack block of a frozen checkpoint with no fine-tuning, treating loops as damped ODE-style sub-steps. Qwen3-4B-Instruct +2.64 pp MMLU-Pro; Qwen3-30B-A3B-Instruct +1.14 pp CommonsenseQA; Moonlight-16B-A3B-Instruct +1.20 pp OpenBookQA; works for dense, MoE and MLA+MoE.

- **LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling** (Huawei Noah's Ark Lab (Yichun Yin, Lifeng Shang co-authors; inferred), 2026-06-03 (v2 2026-08-27; EMNLP 2026)) — no weights. https://arxiv.org/abs/2606.04438
  IterAdaLN (modulation conditioned on iteration index + hidden state) breaks weight-sharing symmetry; capacity balancing restores attention-to-FFN active-param ratio; first strictly controlled looped-MoE vs vanilla-MoE comparison at equal total params, per-token FLOPs and active ratio: >1 point at 3B, ~3 points at 9B across nine benchmarks.

- **Skip a Layer or Loop It? Learning Program-of-Layers in LLMs (PoLar)** (University of Maryland (Tianyi Zhou lab), 2026-06-04 (v2 2026-08-08)) — no weights. https://arxiv.org/abs/2606.06574
  A lightweight predictor emits a per-input execution program that skips or repeats layers of a frozen LLM; most inputs reach equal or better accuracy with shorter programs, some wrong answers get fixed with fewer layers, consistent math-reasoning gains that persist OOD.

- **On the Residual Scaling of Looped Transformers: Stability and Transferability** (same author group as SMELT (Wang, Li, Zhang, Huang, Yan, Li), 2026-06-16) — no weights. https://arxiv.org/abs/2606.18524
  Weight sharing correlates residual updates across iterations, so the usual 1/sqrt(L) depth scaling is insufficient; prescribes epsilon = 1/N for N loops and epsilon = lambda/(N*sqrt(L)) for multi-layer blocks; optimal LR depends only on unique layers L, enabling hyperparameter transfer across loop counts without retuning.

- **LOTUS: Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers** (UW-Madison (Kangwook Lee) + ETH (Anej Svete), 2026-06-30 (v2 2026-07-13)) — no weights. https://arxiv.org/abs/2606.31779
  Recurrent-depth Transformer with parallel supervision on latents: K latent blocks x R iterations with CE loss against gold CoT-step tokens; first latent-CoT method to match explicit CoT at 3B scale; 2.5-6.9x lower thought-phase latency; closes a gap that previously widened with scale.

- **Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts** (University of Cambridge (Popescu, Sáez de Ocáriz Borde, Liò), 2026-07-08) — no weights. https://arxiv.org/abs/2607.20519
  On Ouro-1.4B/2.6B, learned halting gates conflate exit selection with trajectory formation; fixed-prior depth supervision already yields difficulty-aware trajectories; simple confidence-based readouts match learned gates; gate failures stem from joint gate training, not gate capacity; latency tracks mean exit depth.

- **DeepLoop: Depth Scaling for Looped Transformers** (Princeton (Mengdi Wang) + UCLA (Quanquan Gu), 2026-07-15 (v2 2026-08-06)) — no weights. https://arxiv.org/abs/2607.13491
  Formalizes tied-depth effect via a visit-alignment coefficient; loop-aware Post-LN DeepNorm scaling alpha=(2N)^1/2, beta=(8N)^-1/2 for unrolled depth N; exponent must rise from 1/4 to 1/2 as loops grow; neutral without recurrence, helps with it at GPT-2 scale.

- **Loop the Loopies!** (IQuest Lab (per HF paper page), 2026-07-17 (v2 2026-07-20)) — no weights. https://arxiv.org/abs/2607.16051
  Loopie-20B (2B active) and Loopie-6B (0.6B active) MoE looped Transformers using a 'layer-loop' recurrence pattern; claims to substantially outperform vanilla Transformers at equal compute (incl. a 30B-A3B vanilla ablation) and gold-medal performance on 2025 IMO and IPhO without tools via a new post-training method.

- **Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets** (Mark Shapiro (single author; affiliation not on abs page), 2026-07-31) — no weights. https://arxiv.org/abs/2608.11233
  Splits Qwen2.5-0.5B-Instruct into Prelude / weight-tied Recurrent Block / Coda with an identity-preserving one-loop path and a trainable bridge re-injecting the Prelude representation on later loops. Installs at 6M trainable params (adapter over frozen base) or 180M (full block): 83.8% vs 84.0% on the paper's ARC-style task battery; extrapolates to ~1.5x supervised depth (70% accuracy through depth 18); 84% vs 72% for a scratchpad baseline and 7.6x faster; 53% vs 2.5% retained beyond depth 10.

- **Towards Looped Models Done Right — Part I: Topology, Input Injection, Recurrent-State Design** (IFM Research (MBZUAI Institute of Foundation Models: Huang, Shi, Chen, Wen, Liu, Eric Xing, Xuezhe Ma), 2026-07-31 (per search snippets)) — no weights. https://github.com/huskydoge/Awesome-Loop-Models
  Compute-matched ablations of recurrence topology, input injection and recurrent-state design at 730M dense; transfers the favored Huginn-style prelude/loop/coda sandwich to 8B total / 0.8B-active MoE and compares with Ouro and a feedforward baseline.

- **Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching** (TUM (Schwethelm, Rueckert, Kaissis), 2026-08-10) — no weights. https://arxiv.org/abs/2608.09444
  Scheduling at loop-iteration granularity with priority queues for the non-looped prelude/coda stages; on Ouro-1.4B and Huginn-3.5B achieves up to 99% of the theoretical adaptive-depth speed-up, 1.5-1.9x offline throughput, 45-90% lower normalized latency under dynamic load.

- **Looped Language Models Improve Compositional Tool Calling** (University of Cambridge (same authors), 2026-08-17) — no weights. https://arxiv.org/abs/2608.18171
  Matched looped vs non-looped training on API-Bank, BFCL, NESTful: recurrence helps multi-step, dependency-aware tool use with accuracy rising with recurrent depth, small gains on isolated calls; adaptive inference gives better compute-performance trade-off than fixed depth.

- **SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers** (Shaowen Wang, Ge Zhang, ... Shen Yan, Jian Li (ByteDance Seed + Tsinghua per author overlap with Ouro/PLT; not on abs page), 2026-09-01) — no weights. https://arxiv.org/abs/2609.01343
  Loops the middle half of layers twice inside an MoE while matching baseline on three compute metrics; scales four sizes up to 54B non-embedding params; architecture-specific scaling law shows loss drops faster with compute, saving 6.8-18.0% training FLOPs on the compute-optimal frontier.

- **Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?** (Wenlong Wang, Fergal Reid (affiliation not on abs page), 2026-09-01) — no weights. https://arxiv.org/abs/2609.01924
  Virtual-unrolled Jacobian analysis of Ouro-2.6B (48 layers x 4 iterations, deep supervision) and Huginn-0125 (4-layer core x 16) vs Qwen3.6-27B (64 untied layers): a workspace forms in the iterated part of each, but recurrence changes access; Huginn effectively operates with a sliding window of ~two recurrences.


## 3. Runtime support (verified)

- **llama.cpp**: STATUS (verified 2026-09-05 against ggml-org/llama.cpp master): NO support for Ouro or Huginn; generic "loop" plumbing exists and one looped LLM (Nanbeige4.2-3B) is merged.

1) Ouro (ByteDance, model_type "ouro", OuroForCausalLM, custom_code): not in llama.cpp. `gh` searches of issues+PRs for "ouro", "huginn", "recurrent depth", "looped" return nothing relevant; convert_hf_to_gguf.py / conversion/ package has no ouro/huginn/raven class (checked raw master). No GGUF of any ByteDance Ouro model exists on HF (HF API search "ouro" returns only ByteDance/* safetensors, mlx-community/* 4-bit MLX ports, and Firworks/Ouro-2.6B-Thinking-nvfp4). The Ouro model card's "Quantizations / use in llama.cpp, Ollama, LM Studio" sidebar is HF boilerplate; the README itself never mentions llama.cpp/GGUF (https://huggingface.co/ByteDance/Ouro-1.4B/raw/main/README.md). Ouro config: 24 layers (1.4B) / 48 layers (2.6B), hidden 2048, 16 heads x 128 head_dim (no GQA), total_ut_steps=4, early_exit_threshold=1.0, vocab 49152, 65536 ctx, bf16 (https://huggingface.co/ByteDance/Ouro-1.4B/raw/main/config.json). Its KV cache is per (loop, layer): cache index = current_ut * num_hidden_layers + layer_idx, i.e. 96 slots for 1.4B / 192 for 2.6B (modeling_ouro.py lines ~318-322, 543-549).

2) Huginn-0125 (tomg-group-umd, model_type "huginn_raven", RavenForCausalLM, 3.5B, 2 prelude + 4 recurrent + 2 coda layers, n_embd 5280, 55 heads x 96, mean_recurrence 32, vocab 65536, ctx 4096, weights stored fp32 = 15.65 GB): not in llama.cpp. Only artifact is Discussion #11934 "GGUF Support for Latent Reasoning Models" (2025-02-17) with zero maintainer replies and no PR (https://github.com/ggml-org/llama.cpp/discussions/11934). HF discussion "Can we quantize the model to GGUF or GPTQ?" is open with no author answer (https://huggingface.co/tomg-group-umd/huginn-0125/discussions). The old TheBloke "Huginn-13B" GGUFs are an unrelated Llama-2 merge.

3) What DOES exist in llama.cpp for loops: PR #25994 "[Model] Add support for Nanbeige4.2" by zqlcode, merged 2026-07-27 (merge b77d6467) (https://github.com/ggml-org/llama.cpp/pull/25994). It added generic GGUF keys `{arch}.num_loops` and `{arch}.skip_loop_final_norm` (gguf-py/gguf/constants.py lines 150-151; src/llama-arch.cpp lines 245-246) and implements the loop by UNROLLING: `hparams.n_layer_all = n_layer_phys * n_loops`, physical weights shared across loops, "each slot still has its own KV index" (src/models/nanbeige.cpp lines 6-31, 66-69). Nanbeige4.2-3B: 22 layers x num_loops=2, hidden 3072, 48 heads/8 KV heads x 128, vocab 166144, 256K ctx (https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/config.json). GGUFs: bartowski/Nanbeige_Nanbeige4.2-3B-GGUF built with llama.cpp release b10159 (Q4_K_M 2.68 GB, Q8_0 4.43 GB, bf16 8.34 GB) (https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF). Nanbeige's own README still points to a fork branch `nanbeige42` for llama.cpp/vLLM/SGLang/Ollama, which is now stale for llama.cpp (https://huggingface.co/Nanbeige/Nanbeige4.2-3B). Note: Nanbeige is trained from scratch (28T tokens, arXiv 2607.22083), NOT built on Qwen.

4) Other looped archs in flight: IQuest-Coder-V1-40B-Loop (model_type iquestloopcoder, 80 layers, loop_num=2) PR #18680 was CLOSED unmerged 2026-01-08 over undisclosed AI-generated code (https://github.com/ggml-org/llama.cpp/pull/18680); community GGUFs (Avarok, DevQuasar) require that custom branch and the model is 40B anyway. PR #27625 "add support for HrmTextForCausalLM (DFM Mimir 1B)" (a Danish hierarchical/recurrent model) opened 2026-08-23, still open (https://github.com/ggml-org/llama.cpp/pull/27625).

5) Feasibility of adding Ouro to llama.cpp yourself: the Nanbeige unroll pattern maps directly (24 phys layers x 4 loops = 96 logical layers with per-loop KV slots, final RMSNorm applied after each loop, early_exit_gate ignored = fixed-depth 4 steps, which is exactly what the vLLM port did). The `LLAMA_MAX_LAYERS` assert (n_layer_phys*n_loops <= LLAMA_MAX_LAYERS) would need checking for 2.6B (192). Nobody has published this as of today. Early-exit / variable depth per token is not expressible in a static ggml graph without a custom op (the Nanbeige PR discussion notes the "looping by duplicating layers" design as the intended approach).

- **Ollama**: STATUS: Ollama cannot run Ouro or Huginn today; it CAN (very likely, unverified end-to-end) run Nanbeige4.2-3B GGUF because its pinned llama.cpp now contains the arch.

- Ollama v0.33.3 (released 2026-09-02, https://github.com/ollama/ollama/releases) pins llama.cpp tag `b10760` via the repo-root file LLAMA_CPP_VERSION (fetched at both main and v0.33.3; llama/README.md explains the pin; cmake/local.cmake reads it). Tag b10760 (commit 2026-09-02) contains src/models/nanbeige.cpp (HTTP 200 at that ref), and PR #25994 merged 2026-07-27, so `ollama run hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M` should work per HF's Ollama docs (https://huggingface.co/docs/hub/en/ollama). Caveats I could NOT confirm: (a) no end-to-end test was run; (b) Ollama's Go code has zero mentions of "nanbeige" (gh code search total_count 0), so family/template detection may be generic; (c) the `nanbeige/nanbeige4.2:3b-Q4_K_M` name in Nanbeige's README returns 404 on ollama.com; only community uploads exist (ollama.com/ndavat/Nanbeige4.2-3B). Nanbeige's README also offers a fork `git clone -b nanbeige42 https://github.com/Nanbeige/ollama.git` with a Go MLX implementation ("Full native build (MLX Metal on macOS arm64)") and notes "For llama-server (GGUF) backend only, the official ollama repo also works".

- Ouro: open feature request "Support for looping Model like Ouro" #14252 (2026-02-14, label model, zero replies) (https://github.com/ollama/ollama/issues/14252). Nothing on ollama.com: search "ouro" returns only `zoecohn4/Ouro`, an unrelated 4.7 GB "snarky sidekick" assistant with 8K ctx. Huginn: no issues, no library entries. IQuest-Coder 40B/40B-Loop request #13606 (2026-01-02) open, no replies.

- Ollama's two engines and why neither helps: (1) GGUF path = llama.cpp (Metal) -> needs a llama.cpp arch (see llama_cpp_support). (2) MLX path (safetensors) = Ollama's own Go MLX runner: docs/development.md says "The MLX engine enables running safetensor based models. On macOS arm64, MLX is enabled by default"; the runner only has Go implementations under x/models/: cohere2_moe, dflash, gemma4, glimmer, glm4_moe_lite, laguna, llama, nemotron_h, qwen3, qwen3_5, qwen3_5_moe, qwen4_exp (gh api repos/ollama/ollama/contents/x/models). No looped model, and there is no plugin/Python hook to add one. The MLX preview blog (2026-03-30) said it "will expand the list of supported architectures" and required >32 GB unified memory for its launch model (https://ollama.com/blog/mlx); the 2026-06-11 follow-up (https://ollama.com/blog/mlx-performance) adds NVFP4 but no arbitrary-arch support.

- Baselines for evals ARE in the library: ollama.com/library/qwen3 has tags qwen3:1.7b, qwen3:1.7b-fp16, qwen3:4b, qwen3:4b-fp16, qwen3:4b-instruct, qwen3:8b; library/qwen3.5 and library/gemma4 exist (HTTP 200) — these are exactly the baselines in the Ouro paper tables (Qwen3-4B/8B) and Nanbeige paper (Qwen3.5-9B, Gemma4-12B).

- **Importing unsupported architectures into Ollama**: No. Ollama cannot import an architecture its runtimes don't implement, for any input format.

- Safetensors import ("FROM /path/to/safetensors" in a Modelfile + `ollama create`): docs.ollama.com/import states the supported architectures for Safetensors models and adapters are "Llama (including Llama 2, Llama 3, Llama 3.1, and Llama 3.2); Mistral (including Mistral 1, Mistral 2, and Mixtral); Gemma (including Gemma 1 and Gemma 2); Phi3" (https://docs.ollama.com/import). The actual converter, convert/convert.go on main, switches on config.json architectures[0] with an explicit whitelist (LlamaForCausalLM, MllamaForConditionalGeneration, Llama4..., Mistral3..., Ministral3..., MixtralForCausalLM, Gemma/Gemma2/Gemma3/Gemma3n/Gemma4..., Phi3ForCausalLM, Qwen2ForCausalLM, Qwen2_5_VL..., Qwen3VL..., Olmo3ForCausalLM, BertModel, NomicBert..., CohereForCausalLM, GptOssForCausalLM, DeepseekOCR..., DeepseekV3ForCausalLM, Glm4MoeLite..., LagunaForCausalLM, GlmOcr..., Lfm2..., Qwen3Next/Qwen3_5..., NemotronH...) and otherwise returns `fmt.Errorf("unsupported architecture %q", p.Architectures[0])` (convert.go ~lines 297-360). "OuroForCausalLM", "RavenForCausalLM"/huginn_raven, "NanbeigeForCausalLM" and "IQuestLoopCoderForCausalLM" are absent, so `ollama create` from their safetensors fails immediately. Remote Python modeling code (auto_map/trust_remote_code) is never executed by Ollama.

- GGUF import ("FROM model.gguf" or `ollama run hf.co/<user>/<repo>:<quant>`): the file is accepted, but at load time the runner must know `general.architecture`; otherwise you get "unknown model architecture: '<name>'" — see issues #15508 (gemma4), #14512/#15499/#15747 (qwen35moe), #16664 (diffusion-gemma), #3638; the fix in each case was Ollama bumping its llama.cpp pin (https://github.com/ollama/ollama/issues/15508). Ollama v0.33.3 pins llama.cpp b10760 (LLAMA_CPP_VERSION), which knows `nanbeige` (num_loops unrolled) but not ouro/huginn. So: Nanbeige4.2 GGUF -> should import; Ouro/Huginn -> no GGUF can exist yet, and even a hand-made one would fail with unknown architecture.

- MLX/safetensors runner: Ollama's Go MLX engine only implements the architectures under x/models (cohere2_moe, dflash, gemma4, glimmer, glm4_moe_lite, laguna, llama, nemotron_h, qwen3, qwen3_5, qwen3_5_moe, qwen4_exp); the blog says architectures will be expanded over time but there is no user-extensible mechanism (https://ollama.com/blog/mlx; docs/development.md "MLX Engine (Optional)").

Conclusion: to "pair with Ollama", a looped HF/MLX model must be served by a separate process that speaks Ollama's API (see bridge_options), or you use Nanbeige4.2-3B GGUF which stock Ollama can already load via llama.cpp.

- **MLX / mlx-lm**: STATUS: Best-supported local path on this Mac, but Ouro support lives ONLY in an unmerged mlx-lm PR; Nanbeige is on mlx-lm main but not on PyPI yet.

- Ouro in mlx-lm: PR #599 "Add Ouro" by kernelpool, head `kernelpool/mlx-lm:feature/ouro`, opened 2025-11-09, still OPEN (mergeable=null, last update 2026-08-24). Adds mlx_lm/models/ouro.py (+304 lines) + tests. Maintainer awni (2025-12-03): "I've generally been quite skeptical of models with early exit as it doesn't play very well with GPUs... less efficient than just running the model in full". kernelpool 2026-08-21: "I've updated this branch and re-tested it against the 4 different Ouro models... as another looping model implementation, next to mlx_lm/models/iquestloopcoder.py" (https://github.com/ml-explore/mlx-lm/pull/599). The implementation (https://raw.githubusercontent.com/kernelpool/mlx-lm/feature/ouro/mlx_lm/models/ouro.py) exposes ModelArgs.total_ut_steps (default 4), early_exit_step, early_exit_threshold, and per-call kwargs use_weighted_exit / exit_at_step / exit_threshold; KV cache = total_ut_steps * num_layers entries (cache_idx = current_ut * num_layers + layer_idx), mirroring the HF reference. mlx-lm main has NO ouro.py (404 at main and at tag v0.28.4); gh commit history for that path is empty.

- Ready-made MLX weights: mlx-community/Ouro-1.4B-4bit (0.81 GB), Ouro-2.6B-4bit (1.5 GB), Ouro-1.4B-Thinking-4bit, Ouro-2.6B-Thinking-4bit, all uploaded 2025-11-09, card says "converted ... using mlx-lm version 0.28.4" — but 0.28.4 has no ouro class, so they were made with the PR branch; loading them requires `pip install git+https://github.com/kernelpool/mlx-lm@feature/ouro` (untested here). Their config keeps model_type "ouro", total_ut_steps 4, early_exit_threshold 1.0, 4-bit affine gs64 with early_exit_gate at 8-bit (https://huggingface.co/mlx-community/Ouro-2.6B-Thinking-4bit/raw/main/config.json).

- Nanbeige4.2 in mlx-lm: merged to main 2026-08-29, "Add Nanbeige (Nanbeige4.2 looped transformer) model support (#1597)"; models/nanbeige.py has num_loops, skip_loop_final_norm, loop_share_kv, effective_num_loops, and "Each loop has its own KV cache entries". NOT in any PyPI release: latest mlx-lm on PyPI is 0.31.3 (2026-04-22) and v0.31.3 tag lacks nanbeige.py -> install from git main. mlx-community/Nanbeige4.2-3B-OptiQ-4bit (3.15 GB) instead uses the third-party OptiQ >=0.4.6 package that "registers itself with mlx-lm on import optiq" and provides `optiq serve` (OpenAI-compatible) (https://huggingface.co/mlx-community/Nanbeige4.2-3B-OptiQ-4bit).

- Huginn: no MLX port on HF (API search "huginn-0125" -> only tomg-group-umd original + 2 mirrors + JonasGeiping checkpoints), nothing in mlx-lm.

- Other looped model in mlx-lm main: iquestloopcoder.py (40B only). mlx-lm also ships mlx_lm.server: OpenAI-compatible /v1/chat/completions, /v1/completions, /v1/models on :8080, "not recommended for production" (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md). For evals on MLX: chimezie/lm-evaluation-harness-mlx (https://github.com/chimezie/lm-evaluation-harness-mlx); upstream lm-eval has no MLX backend.

- Environment facts: PyPI mlx 0.32.2 (>=3.10), mlx-lm 0.31.3 (>=3.8); this Mac has uv-managed CPython 3.12.13 already installed (python3 is actually 3.14.7, not 3.9).

- **vLLM / SGLang**: vLLM: Ouro was ADDED then REMOVED. PR #27794 "[Model][Ouro] Support Ouro Model" merged 2025-10-30 (adapted from Qwen2 code; "we not use early exit in vllm. It's hard to handle kv cache here" — always runs full total_ut_steps; PP broken) (https://github.com/vllm-project/vllm/pull/27794). PR #49786 "[Model] Remove Ouro" by hmellor merged 2026-07-25 ("~1 year old... See very little usage in vLLM... marked for deletion") (https://github.com/vllm-project/vllm/pull/49786). registry.py now lists `"OuroForCausalLM": "0.26.0"` in _PREVIOUSLY_SUPPORTED_MODELS, and loading raises "Model architecture OuroForCausalLM was supported in vLLM until v0.26.0, and is not supported anymore" (registry.py ~lines 814, 1165-1173). Latest vLLM is v0.28.0 (2026-08-26). Issue #37668 "Early Stopping for Ouro models" closed as stale/not planned. The Ouro model card still says "vLLM does not currently support the adaptive exit feature... the model will always execute the full number of total_ut_steps". IQuestLoopCoderForCausalLM is still registered. Huginn: an experimental vLLM-v1 plugin lives in seal-rg/recurrent-pretraining/vllm (raven_vllm.py; `vllm serve tomg-group-umd/huginn-0125 --trust-remote-code --dtype bfloat16 --max-model-len 4096`, recurrence via hf_overrides {"mean_recurrence": 32}; "Token-level adaptive computation remains unimplemented") (https://github.com/seal-rg/recurrent-pretraining/tree/main/vllm). On macOS vLLM is CPU-only, build-from-source, "supports FP32 and FP16" (https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html); the vllm-metal MLX plugin lists Qwen3/3.5/3.6/3.8, Gemma 3/4, Llama 3, Mistral, GPT-OSS, GLM, Phi, SmolLM3 etc. but no Ouro/Nanbeige/looped model (https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md). None of this is practical on a 16 GB M4.

SGLang: no native Ouro model file (python/sglang/srt/models/ouro.py 404). PR #32894 "fix(transformers): support ByteDance/Ouro-1.4B via Transformers backend fallback" (closed 2026-07-30; merge status unclear from the page) patched transformers-v5 incompatibilities but the author reports KV reuse "degrades into garbage after ~2 tokens" because UniversalTransformerCache isn't supported — i.e. not usable for real generation (https://github.com/sgl-project/sglang/pull/32894). SGLang does document Apple Silicon: `uv pip install -e "python[all_mps]"`, `SGLANG_USE_MLX=1` MLX path (macOS 14+, PyTorch 2.13.x, MLX 0.32+), and a PyTorch-MPS fallback that claims "any HF model" with reduced features (https://docs.sglang.io/docs/hardware-platforms/apple_metal) — but given the cache bug above, Ouro via SGLang on Mac is not a viable route. Nanbeige provides SGLang fork branch `nbg42` and vLLM branch `nanbeige42` (README). Huginn's HF card summary mentions vLLM/SGLang but the README contains no SGLang instructions; the psych0v0yager/sglang fork has no README changes — unconfirmed.

- **Apple Silicon notes**: MACHINE AS OBSERVED (not as stated in the task): Apple M4, 16 GB (hw.memsize 17179869184), macOS 26.5.2 (Darwin 25F84); `python3` is 3.14.7 (task said 3.9); uv 0.11.19 with CPython 3.12.13 already installed under ~/.local/share/uv; free disk is ~1.9-2.0 GB on the Data volume (task said ~3.5 GB). Nothing among torch/mlx/ollama is installed. Disk is the first blocker: ByteDance/Ouro-1.4B bf16 = 2.87 GB, Ouro-2.6B = 5.34 GB, Nanbeige4.2-3B bf16 = 8.36 GB, huginn-0125 = 15.65 GB (stored fp32) — only mlx-community/Ouro-1.4B-4bit (0.81 GB) and Ouro-2.6B-4bit (1.5 GB) fit right now; Nanbeige Q4_K_M GGUF (2.68 GB) / OptiQ-4bit (3.15 GB) need space freed; Huginn needs ~16 GB download + bf16 re-save.

WHEELS: PyPI torch 2.14.0 ships macosx_14_0_arm64 wheels for cp310-cp314; transformers latest is 5.16.1. Ouro's README says "Please use transformers<4.56.0 ... recommend transformers==4.54.1", but that is stale: ByteDance merged the community UniversalTransformerCache fix and "verified ... enables Ouro to run with transformers>=4.56.0" (HF discussion #3, 2025-11-16), an lm-eval reproduction used transformers 4.57.6 (discussion #8), and transformers 5.x breaks the remote code (SGLang PR #32894 lists v5 API incompatibilities; HF PR #13 "Fix for transformers 5.x.x" is an empty draft). Recommendation: `uv venv -p 3.12`, torch 2.14, transformers ~=4.57, trust_remote_code=True. Huginn's remote code targets transformers' DynamicCache API and was last modified 2025-07-29 — compatibility with transformers 4.57/5.x is UNCONFIRMED.

TORCH MPS + trust_remote_code: MPS runs any pure-PyTorch custom model; Ouro's modeling_ouro.py and Huginn's raven_modeling_minimal.py use standard ops (RMSNorm, RoPE, SDPA/eager attention, sigmoid gate), so they should run with device_map="mps"/.to("mps") — but I found NO published report of either on MPS (searches for Ouro + MPS/Apple Silicon returned nothing). Set PYTORCH_ENABLE_MPS_FALLBACK=1 for any missing op (HF docs: "MPS doesn't support all PyTorch operations yet"). HF docs: "MPS requires the entire model to fit in unified memory, so device_map="auto" can't offload layers to the CPU"; "Loading weights to MPS is faster and uses less memory with safetensors 0.8.0 and PyTorch 2.9 or later" (https://huggingface.co/docs/transformers/main/en/perf_train_special). Huginn's model card: "the number of materialized parameters is num_steps * 1.5B + 2B" (compute, not memory) and "trained with bfloat16-mixed precision, so we recommend using bfloat16".

BF16 ON MPS: supported on Apple Silicon; HF docs state "MPS supports both bf16 and fp16 mixed precision (bf16 requires macOS 14.0 or later)" — macOS 26 qualifies. pytorch/pytorch#141864 ("BFloat16 is not supported on MPS") concerns Intel/AMD Macs, closed not-planned; #139386 (bf16 autocast on MPS) is still open but autocast is irrelevant for plain bf16 inference. Claims that bf16 is "up to 10x slower than fp16 on MPS" come from an unofficial guide (mattmireles/kokoro-coreml pytorch-mps.md) and are UNVERIFIED; if bf16 decode is slow, try float16 (Ouro card lists torch_dtype bf16; Huginn authors note benchmarks were "evaluated in pure bfloat16").

MPS MEMORY LIMITS: PyTorch MPS allocator hard limit PYTORCH_MPS_HIGH_WATERMARK_RATIO default 1.7 and soft limit PYTORCH_MPS_LOW_WATERMARK_RATIO default 1.4 on unified memory, both multiples of Metal's device.recommendedMaxWorkingSetSize (https://docs.pytorch.org/docs/2.5/_sources/mps_environment_variables.rst.txt); on a 16 GB Apple Silicon Mac llama.cpp reports recommendedMaxWorkingSetSize = 10922.67 MB (https://github.com/abetlen/llama-cpp-python/issues/687). Plan for <=11-12 GB of model+KV+activations with the OS and Ollama's baseline model unloaded (keep_alive 0 on the baseline when not in use).

MEMORY ARITHMETIC (computed from config.json + cache code; bf16, batch 1): 
- Ouro-1.4B: weights 2.87 GB; KV per token per slot = 2*16*128*2 B = 8 KB; slots = 24 layers x 4 steps = 96 -> 768 KB/token -> 1.5 GB @2k ctx, 3.0 GB @4k, 6.0 GB @8k. bf16 total @4k ≈ 6 GB + activations: fits. MLX 4-bit weights 0.81 GB (KV stays 16-bit).
- Ouro-2.6B: weights 5.34 GB; 48 x 4 = 192 slots -> 1.5 MB/token -> 6 GB @4k. bf16 @4k ≈ 11.5 GB: at the edge of the ~11 GB Metal working set; use MLX 4-bit (1.5 GB) or keep ctx <=2k in bf16.
- Nanbeige4.2-3B: weights 8.36 GB bf16 / 2.68 GB Q4_K_M; 8 KV heads -> 4 KB/slot; 22 x 2 = 44 slots -> 176 KB/token -> 0.7 GB @4k, 5.6 GB @32k.
- Huginn-0125: weights ~7 GB in bf16 (3.5B; on-disk fp32 15.65 GB); HuginnDynamicCache keys caches by step index (prelude 2 + 4 x num_steps + coda 2 slots), 2*55*96*2 B = 20.6 KB/slot -> at num_steps=32: 132 slots = 2.8 MB/token -> 2.9 GB @1k, 5.7 GB @2k, 11.4 GB @4k (ctx max 4096). So Huginn r=32 on 16 GB is only feasible at <=1.5-2k tokens, or with its "compress-s<k>" lookup_strategy cache modes (raven_modeling_minimal.py lines 148-273) or lower num_steps (r=8/16 still give most of the gains per the paper's Table 1).

SPEED (MLX vs MPS): peer-reviewed comparison on an M2 Ultra: "MLX achieves the highest sustained generation throughput ... PyTorch MPS remains limited by memory constraints on large models and long contexts" (arXiv 2511.05502). Secondary sources report MLX ~230 tok/s vs PyTorch MPS ~7-9 tok/s for small Qwen2.5 models and MLX 1.4-1.8x llama.cpp-Metal decode (medium.com/@michael.hannecke; yage.ai) — not independently verified. Expect MLX 4-bit to be the only way to get interactive speed for looped models here: every generated token re-reads the shared block once per loop (Ouro: 4x ~1.2B non-embedding params; Huginn r=32: 32x 1.5B), so decode is bandwidth-bound at loop_count x weight_bytes per token. Rough, UNCONFIRMED estimate on an M4 (base) at ~120 GB/s: Ouro-1.4B bf16 ≈ 10 tok/s ceiling, 4-bit ≈ 30-40 tok/s ceiling; Huginn r=32 bf16 ≈ 1 tok/s ceiling. No published Ouro/Huginn Apple Silicon numbers were found. mlx-lm's Ouro branch keeps all UT-step hidden states for early exit ("less efficient than just running the model in full" per awni), so use exit_at_step / fixed depth for benchmarking.

EVAL REPRODUCTION TARGETS (for the 'vs base' study): Ouro paper Table 7 (Ouro-1.4B R4 vs Qwen3-4B): MMLU 67.35/73.19, MMLU-Pro 48.62/51.40, BBH 71.02/70.95, GSM8K 78.92/72.86, MATH500 82.40/59.60, HumanEval 74.40/77.40; Table 8 (Ouro-2.6B R4 vs Qwen3-8B): GSM8K 81.58/83.09, MATH500 90.85/62.30, MMLU 74.60/76.63; Table 10/11 depth sweep (1.4B MMLU: 41.21/60.43/66.71/67.45 at steps 1-4; 2.6B: 51.55/67.63/73.57/74.60); Table 12 (1.4B-Thinking AIME24 0.0/37.33/62.33/65.00 at steps 1-4) (https://arxiv.org/html/2510.25741). Settings that reproduce them: lm-eval-harness, NO chat template, 5-shot MMLU, 3-shot CoT BBH/GSM8K; a community re-run with transformers 4.57.6 + lm-eval 0.4.11 got MMLU 67.46 / BBH 71.06 / GSM8K 79.38 (https://huggingface.co/ByteDance/Ouro-1.4B/discussions/8). Huginn: lm-eval with `--model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32`; paper Table 1 (r=4/8/16/32): ARC-C 27.99/35.15/37.71/38.23, HellaSwag 43.46/58.54/64.67/65.21, MMLU 23.39/25.29/31.25/31.38; GSM8K r=32 flexible/strict 28.20/38.13 with system prompt, GSM8K-CoT 34.80/42.08 (https://arxiv.org/html/2502.05171v2). Nanbeige4.2-3B model card compares to Qwen3.5-9B/Gemma4-12B (GPQA-Diamond 87.4/81.7/78.8, HMMT-Feb-2026 82.8/69.6/51.5, LiveCodeBench-V6 72.5/65.6/72.0) but lists no GSM8K/MATH500 numbers.

NOT CONFIRMED: no Qwen-based looped model with public weights exists — the Qwen2.5-0.5B recurrent-depth retrofit (arXiv 2608.11233, 2026-07-31) publishes no weights on its abstract page; Relaxed Recursive Transformers (Gemma-based) never released weights; Mixture-of-Recursions official checkpoints are 360M-only via Google Drive (raymin0223 README) plus an unofficial HF repo sudeshmu/mixture-of-recursions-360m. Ouro itself uses Qwen2-style modules (vLLM port "adapting architecture from the Qwen2 model") but is pretrained from scratch on 7.7T tokens, so the paper's Qwen3-1.7B/4B/8B comparisons are the closest "vs Qwen" baseline.


### Bridge options considered

- **Ollama-API shim in Python over mlx-lm (kernelpool feature/ouro branch) — recommended for Ouro on this 16 GB M4** — uv venv -p 3.12; pip install mlx==0.32.x and `git+https://github.com/kernelpool/mlx-lm@feature/ouro` (adds mlx_lm/models/ouro.py; PR #599, retested 2026-08-21 against all 4 Ouro models). Load mlx-community/Ouro-1.4B-4bit (0.81 GB) or Ouro-2.6B-4bit (1.5 GB) with mlx_lm.load; write a FastAPI/Starlette server on :11435 implementing GET /, GET /api/version, GET /api/tags, POST /api/show, GET /api/ps, POST /api/chat, POST /api/generate, GET /v1/models, POST /v1/chat/completions, using the exact JSON shapes in ollama_api_spec (NDJSON for /api/*, 'data: ...\n\n' + 'data: [DONE]' SSE for /v1/*, durations in ns). Map options.total_ut_steps / options.exit_at_step / options.exit_threshold to the model call kwargs (exit_at_step/exit_threshold/use_weighted_exit exist on the MLX Model.__call__). Optionally reverse-proxy unknown model names to the real Ollama on :11434 and merge /api/tags so Open WebUI/ollama-python see one server.
  - pros: Fastest option on Apple Silicon (MLX, 4-bit); per-request control of loop depth for depth-sweep replication; full Ollama + OpenAI surface so any Ollama client works; both Ouro and baseline Qwen3 models visible in one endpoint; kernelpool's implementation mirrors the HF reference cache layout.
  - cons: Unmerged fork (may drift from mlx-lm main; untested with mlx 0.32.2); MLX 4-bit weights are not bf16 (accuracy deltas vs paper must be measured — convert bf16->MLX yourself with mlx_lm.convert --dtype bfloat16 if disk allows); mlx-lm keeps all UT hidden states for early exit (extra memory/time); no Huginn port on MLX; you must fake /api/show model_info fields.
  - effort: 1-2 days for a solid shim (there are reference translators: regismesquita/ollama_proxy, eyalrot/ollama_openai, olegshulyakov/ollama-openai-proxy)

- **mlx_lm.server (OpenAI-compatible) + off-the-shelf Ollama->OpenAI translating proxy** — `mlx_lm.server --model mlx-community/Ouro-1.4B-4bit --port 8080` (with the kernelpool branch installed) exposes /v1/chat/completions, /v1/completions, /v1/models; run an Ollama-emulating proxy in front (e.g. github.com/regismesquita/ollama_proxy which maps /api/chat->/v1/chat/completions and /api/tags->/v1/models; or eyalrot/ollama_openai; olegshulyakov/ollama-openai-proxy).
  - pros: Near-zero code; OpenAI endpoint natively available for lm-eval's `local-chat-completions` backend.
  - cons: No way to pass loop-depth knobs per request (mlx_lm.server doesn't forward custom kwargs) — you get fixed total_ut_steps only; proxies vary in /api/show fidelity (Open WebUI needs it); mlx_lm.server is 'not recommended for production'.
  - effort: hours

- **Run Nanbeige4.2-3B GGUF directly in stock Ollama (no bridge) as the 'looped model in Ollama' track** — `ollama run hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M` (2.68 GB; Ollama v0.33.3 pins llama.cpp b10760 which contains src/models/nanbeige.cpp from PR #25994). Ablation: copy the GGUF and rewrite the `nanbeige.num_loops` metadata key to 1 with gguf-py (llama.cpp reads it at load and unrolls layers accordingly) to get a same-weights no-loop baseline; compare against qwen3:4b / qwen3.5 in the Ollama library.
  - pros: Zero bridge code; everything (looped model + baselines) inside Ollama; llama.cpp Metal is fast; small download; also runs in mlx-lm main (models/nanbeige.py, git install) and via OptiQ (mlx-community/Nanbeige4.2-3B-OptiQ-4bit).
  - cons: Not Ouro/Huginn and not a published 'latent-reasoning depth-scaling' paper (Nanbeige's report is agentic-focused; only 2 loops, no early exit); loop count is a load-time GGUF constant, not per request; end-to-end loading in Ollama 0.33.3 is unverified (Ollama code has no 'nanbeige' string; the README's nanbeige42 Ollama fork predates upstream support); num_loops=1 ablation quality is untested.
  - effort: minutes to run; 1-2 hours for the num_loops ablation

- **Ollama-API shim over HF transformers on torch MPS (bf16) — for exact-paper fidelity or Huginn** — uv venv -p 3.12; torch 2.14 (macosx_14_0_arm64 cp312 wheel), transformers~=4.57 (<5), accelerate; AutoModelForCausalLM.from_pretrained('ByteDance/Ouro-1.4B', trust_remote_code=True, torch_dtype=torch.bfloat16).to('mps'); pass exit_at_step/exit_threshold/use_weighted_exit via generate kwargs; for Huginn: 'tomg-group-umd/huginn-0125', trust_remote_code=True, bf16, model.generate(..., num_steps=r) or generate_with_adaptive_compute(), and HuginnDynamicCache(lookup_strategy='compress-s16') to bound KV. Same shim server as option 1 (swap backend). Set PYTORCH_ENABLE_MPS_FALLBACK=1.
  - pros: Bit-for-bit the reference implementation and dtype used in the papers (bf16), so lm-eval numbers are directly comparable; only path that runs Huginn locally; lm-eval `--model hf ... device=mps` works out of the box.
  - cons: Slowest (PyTorch MPS; unverified bf16 kernel speed on M4); Ouro-2.6B bf16 + 4k KV (~11.5 GB) is at the Metal working-set limit; Huginn needs a 15.65 GB fp32 download (disk!) and ~2.8 MB/token KV at r=32; no published MPS reports for either model; transformers 5.x breaks Ouro remote code.
  - effort: 1-2 days (shim) + disk cleanup

- **Write an `ouro` architecture for llama.cpp (then it works in Ollama, LM Studio, everything)** — Follow PR #25994's pattern: new LLM_ARCH_OURO with hparams from GGUF keys (reuse `{arch}.num_loops`=total_ut_steps), unroll 24 physical layers x 4 loops into 96 logical layers sharing tensors but with separate KV slots (2.6B = 192; check LLAMA_MAX_LAYERS), apply the final RMSNorm between loops (modeling_ouro.py applies self.norm after every UT step), drop early_exit_gate (fixed depth = what vLLM did); add conversion/ouro.py (Qwen2-like tensor names, 49152 vocab, tokenizer from vocab.json); then `ollama create` from the GGUF or upstream the PR and wait for Ollama to bump LLAMA_CPP_VERSION.
  - pros: Native Ollama support with Metal speed and quantization; benefits the community; Nanbeige/HRM PRs show maintainers accept unrolled-loop archs.
  - cons: C++/Python work of several days; no variable depth or early exit (static graph); GGUF quantization changes accuracy vs paper; Ollama picks it up only after its next llama.cpp pin bump (or you build Ollama from source).
  - effort: days to 2 weeks

- **SGLang on macOS (SGLANG_USE_MLX=1 or PyTorch-MPS fallback) or vLLM-metal / vLLM CPU** — SGLang: clone, `uv pip install -e "python[all_mps]"`, `SGLANG_USE_MLX=1 python -m sglang.launch_server --model-path ByteDance/Ouro-1.4B --trust-remote-code --disable-cuda-graph`. vLLM: only <=v0.26.0 had OuroForCausalLM (removed in PR #49786); macOS build is CPU-only fp32/fp16; vllm-metal has no Ouro.
  - pros: OpenAI-compatible server for free if it worked.
  - cons: Not viable: SGLang's Transformers-backend fallback for Ouro produces garbage after ~2 tokens due to the unsupported UniversalTransformerCache (PR #32894 author); vLLM dropped Ouro and has no Metal path; Huginn's vLLM plugin is CUDA-oriented and experimental.
  - effort: n/a — do not pursue on this machine


### Ollama API surface a shim must implement

Authoritative sources: server/routes.go (route table), api/types.go (JSON field names), middleware/openai.go (OpenAI SSE), docs/api.md and docs.ollama.com/api/*. All durations are nanoseconds; timestamps RFC3339 with nanos; server default http://localhost:11434.

ROUTE TABLE (routes.go lines 1887-1940, main): HEAD/GET "/" -> text "Ollama is running"; HEAD/GET /api/version -> {"version":"0.33.3"}; GET /api/status; POST /api/pull, /api/push; HEAD/GET /api/tags; POST /api/show; DELETE /api/delete; POST /api/create; POST+HEAD /api/blobs/:digest; POST /api/copy; GET /api/ps; POST /api/generate; POST /api/chat; POST /api/embed, /api/embeddings; OpenAI: POST /v1/chat/completions, /v1/completions, /v1/embeddings, GET /v1/models, GET /v1/models/:model, POST /v1/responses; also /v1/messages (Anthropic-style) and /v1/audio/transcriptions. A minimal shim that satisfies Open WebUI / ollama-python / most clients: GET /, GET /api/version, GET /api/tags, POST /api/show, GET /api/ps, POST /api/chat, POST /api/generate, GET /v1/models, POST /v1/chat/completions (and optionally /v1/completions, /api/embed).

GET /api/tags -> {"models":[{"name":"qwen3:4b","model":"qwen3:4b","modified_at":"2025-10-03T23:34:03.409490317-07:00","size":9608350245,"digest":"<sha256 hex>","details":{"parent_model":"","format":"gguf","family":"gemma4","families":["gemma4"],"parameter_size":"8.0B","quantization_level":"Q4_K_M"}}]} (docs.ollama.com/api/tags.md; ListModelResponse also allows optional "capabilities"; ModelDetails allows optional context_length/embedding_length).

POST /api/show request {"model":"...","verbose":false} -> {"license","modelfile","parameters":"num_keep 24\nstop \"<|...|>\"","template":"{{ if .System }}...","system","details":{parent_model,format,family,families,parameter_size,quantization_level},"model_info":{"general.architecture":"...","general.parameter_count":N,"<arch>.context_length":N,"<arch>.embedding_length":N,"<arch>.block_count":N,"tokenizer.ggml.model":"...", ...},"capabilities":["completion","tools","thinking"],"modified_at":"..."} (docs.ollama.com/api-reference/show-model-details.md; ShowResponse struct). Clients read details.family and model_info["general.architecture"] / "<arch>.context_length" — populate these consistently (e.g. architecture "ouro", ouro.context_length 65536).

GET /api/ps -> {"models":[{"name","model","size","digest","details":{...},"expires_at":"...","size_vram":N,"context_length":4096}]}.

POST /api/chat request: {"model" (req), "messages":[{"role":"system|user|assistant|tool","content":"...","thinking"?:"...","images"?:[b64],"tool_calls"?:[...],"tool_name"?,"tool_call_id"?}], "tools"?:[...], "format"?: "json"|<JSON schema>, "options"?:{temperature,top_p,top_k,min_p,num_predict,num_ctx,seed,stop:[...],repeat_penalty,repeat_last_n,presence_penalty,frequency_penalty,typical_p,num_keep,num_batch,num_gpu,num_thread,...}, "stream"?: bool (default true), "keep_alive"?: "5m"|0, "think"?: bool|"high"|"medium"|"low"|"max", "logprobs"?, "top_logprobs"?, "truncate"?, "shift"?}. Streaming response: Content-Type application/x-ndjson, one JSON object per line: {"model":"m","created_at":"2023-08-04T08:52:19.385406455-07:00","message":{"role":"assistant","content":"The"},"done":false} ... optional {"message":{"role":"assistant","thinking":"..."}} chunks ... final: {"model":"m","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"|"length"|"load","total_duration":4883583458,"load_duration":1334875,"prompt_eval_count":26,"prompt_eval_duration":342546000,"eval_count":282,"eval_duration":4535599000}. Non-streaming ({"stream":false}): single JSON with full message.content plus the same metrics. Tool calls: message.tool_calls:[{"id"?,"function":{"index":0,"name":"...","arguments":{...object...}}}]. (docs.ollama.com/api/chat.md, docs.ollama.com/api/streaming.md, raw docs/api.md, ChatRequest/ChatResponse/Message/Metrics structs.)

POST /api/generate request: {"model","prompt","suffix"?,"system"?,"template"?,"context"?:[ints] (legacy),"stream"?,"raw"?,"format"?,"images"?,"options"?,"keep_alive"?,"think"?,"logprobs"?,"top_logprobs"?}. Stream chunk: {"model","created_at","response":"The","done":false}; final: {"model","created_at","response":"","thinking"?,"done":true,"done_reason":"stop","context":[1,2,3],"total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"} (GenerateResponse struct; docs.ollama.com/api/generate.md no longer documents "context" but the server still emits it).

Other: POST /api/embed {"model","input":str|[str]} -> {"model","embeddings":[[...]],"total_duration","load_duration","prompt_eval_count"}; POST /api/pull {"model","stream"} streams {"status":"pulling manifest"}... {"status":"success"}; DELETE /api/delete {"model"}; HEAD /api/blobs/sha256:<hex> -> 200/404; POST /api/create {"model","from"|"files",...}. keep_alive: "Controls how long the model will stay loaded into memory following the request (default: 5m)", 0 unloads.

OpenAI-compatible (docs.ollama.com/api/openai-compatibility.md; middleware/openai.go): GET /v1/models -> {"object":"list","data":[{"id":"qwen3:4b","object":"model","created":<unix>,"owned_by":"library"}]}; GET /v1/models/{id} -> single Model. POST /v1/chat/completions supports model, messages, temperature, top_p, max_tokens, stream, tools, response_format, seed, stop, reasoning_effort ("high"/"medium"/"low"/"max"/"none"); unsupported: logprobs, tool_choice. Non-stream: Content-Type application/json {"id":"chatcmpl-...","object":"chat.completion","created":..,"model":..,"choices":[{"index":0,"message":{"role":"assistant","content":".."},"finish_reason":"stop"}],"usage":{"prompt_tokens","completion_tokens","total_tokens"}}. Stream: Content-Type text/event-stream, each event written as "data: {json}\n\n" with object "chat.completion.chunk" and choices[].delta, terminated by "data: [DONE]\n\n" (openai.go lines 89, 123-157). /v1/completions: prompt string only; no best_of/echo/logit_bias. /v1/embeddings: input str|[str], encoding_format, dimensions. Model aliasing: `ollama cp llama3.2 gpt-3.5-turbo`.

SHIM DESIGN NOTES: run on :11434 only if Ollama isn't running (or run Ollama on another port via OLLAMA_HOST and let the shim reverse-proxy every non-looped model name to it — the shim then merges /api/tags from both). Expose loop knobs through the free-form "options" map (e.g. options.total_ut_steps, options.exit_at_step, options.exit_threshold, options.num_steps for Huginn) — Ollama passes options as map[string]any so clients will forward unknown keys.


## 4. Published evaluation numbers (replication targets)


### Ouro-1.4B (ByteDance, base LoopLM, 24 layers x 4 recurrent steps, 1.4B params, 7.7T tokens)

- **GSM8K** — lm-eval-harness, strict match, 3-shot CoT (paper Table 16). Base models, no chat template. Ouro at T=4 (R4).

  - Ouro-1.4B R4: 78.92

  - Qwen3-1.7B-Base: 70.28

  - Qwen2.5-1.5B: 60.73

  - Llama3.2-1B: 7.05

  - Gemma3-1B: 2.05

  - Qwen2.5-3B: 74.6

  - Llama3.2-3B: 67.2

  - Qwen3-4B-Base: 72.86

  - Gemma3-4B: 68.69

- **MATH500** — In-house harness, strict match, 5-shot CoT (Table 16).

  - Ouro-1.4B R4: 82.4

  - Qwen3-1.7B-Base: 25.8

  - Qwen2.5-1.5B: 17.6

  - Llama3.2-1B: 7.4

  - Gemma3-1B: 41 (Printed as 41.00 in arXiv HTML Table 7; looks anomalous (Gemma3-1B GSM8K is 2.05). Not confirmed.)

  - Qwen2.5-3B: 42.6

  - Llama3.2-3B: 40.8

  - Qwen3-4B-Base: 59.6

  - Gemma3-4B: 68.6

- **MMLU** — lm-eval-harness, logprobs, 5-shot (Table 16).

  - Ouro-1.4B R4: 67.35 (Table 10 (per-step table) lists 67.45 at T=4 for the same model.)

  - Qwen3-1.7B-Base: 62.46

  - Qwen2.5-1.5B: 60.99

  - Llama3.2-1B: 45.46

  - Gemma3-1B: 39.85

  - Qwen2.5-3B: 65.62

  - Llama3.2-3B: 59.69

  - Qwen3-4B-Base: 73.19

  - Gemma3-4B: 58.37

- **MMLU-Pro** — lm-eval-harness, strict match, 5-shot CoT.

  - Ouro-1.4B R4: 48.62

  - Qwen3-1.7B-Base: 37.27

  - Qwen2.5-1.5B: 29.11

  - Llama3.2-1B: 11.8

  - Gemma3-1B: 11.31

  - Qwen2.5-3B: 37.87

  - Llama3.2-3B: 33.34

  - Qwen3-4B-Base: 51.4

  - Gemma3-4B: 34.61

- **BBH** — lm-eval-harness, strict match, 3-shot CoT.

  - Ouro-1.4B R4: 71.02 (Community repro (HF discussion #8) got 71.06 once chat template was removed; 60.77 with chat template.)

  - Qwen3-1.7B-Base: 53.51

  - Qwen2.5-1.5B: 43.66

  - Llama3.2-1B: 30.72

  - Gemma3-1B: 30.26

  - Qwen2.5-3B: 55.37

  - Llama3.2-3B: 39.45

  - Qwen3-4B-Base: 70.95 (Table 8 lists 71.14 for the same model.)

  - Gemma3-4B: 66.32

- **ARC-C** — lm-eval-harness, logprobs, 25-shot.

  - Ouro-1.4B R4: 60.92

  - Qwen3-1.7B-Base: 55.72

  - Qwen2.5-1.5B: 54.44

  - Llama3.2-1B: 41.98

  - Gemma3-1B: 39.25

  - Qwen2.5-3B: 55.46

  - Llama3.2-3B: 52.47

  - Qwen3-4B-Base: 63.65

  - Gemma3-4B: 60.92

- **HellaSwag** — lm-eval-harness, logprobs, 10-shot.

  - Ouro-1.4B R4: 74.29

  - Qwen3-1.7B-Base: 67.09

  - Qwen2.5-1.5B: 67.73

  - Llama3.2-1B: 59.35

  - Gemma3-1B: 56.12

  - Qwen2.5-3B: 74.54

  - Llama3.2-3B: 73.09

  - Qwen3-4B-Base: 75.66

  - Gemma3-4B: 75.58

- **Winogrande** — lm-eval-harness, logprobs, 5-shot.

  - Ouro-1.4B R4: 72.3

  - Qwen3-1.7B-Base: 66.3

  - Qwen2.5-1.5B: 66.77

  - Llama3.2-1B: 62.75

  - Gemma3-1B: 58.72

  - Qwen2.5-3B: 70.17

  - Llama3.2-3B: 69.14

  - Qwen3-4B-Base: 71.19

  - Gemma3-4B: 71.07

- **HumanEval / HumanEval+** — evalplus pass@1 (HumanEval score, then HumanEval+ in note).

  - Ouro-1.4B R4: 74.4 (HumanEval+ 67.40)

  - Qwen3-1.7B-Base: 66.5 (HumanEval+ 59.80)

  - Qwen2.5-1.5B: 52.4 (HumanEval+ 46.30)

  - Llama3.2-1B: 19.5 (HumanEval+ 17.40)

  - Gemma3-1B: 6.7 (HumanEval+ 5.50)

  - Qwen2.5-3B: 68.9 (HumanEval+ 62.20)

  - Llama3.2-3B: 29.9 (HumanEval+ 26.20)

  - Qwen3-4B-Base: 77.4 (HumanEval+ 70.70 (Table 8 lists HumanEval 77.70))

  - Gemma3-4B: 34.8 (HumanEval+ 29.30)

- **MBPP / MBPP+** — evalplus pass@1 (MBPP score, MBPP+ in note).

  - Ouro-1.4B R4: 73 (MBPP+ 62.70)

  - Qwen3-1.7B-Base: 68 (MBPP+ 58.50)

  - Qwen2.5-1.5B: 60.3 (MBPP+ 50.00)

  - Llama3.2-1B: 35.7 (MBPP+ 29.10)

  - Gemma3-1B: 12.4 (MBPP+ 10.10)

  - Qwen2.5-3B: 63 (MBPP+ 54.20)

  - Llama3.2-3B: 50.3 (MBPP+ 39.70)

  - Qwen3-4B-Base: 78.8 (MBPP+ 65.90)

  - Gemma3-4B: 60.6 (MBPP+ 51.10)

- **MMLU (5-shot avg) vs recurrent step T (Table 10)** — lm-eval-harness logprobs; trained at T=4; T=5..8 is extrapolation.

  - Ouro-1.4B T=1: 41.21

  - Ouro-1.4B T=2: 60.43

  - Ouro-1.4B T=3: 66.71

  - Ouro-1.4B T=4: 67.45

  - Ouro-1.4B T=5: 66.64

  - Ouro-1.4B T=6: 65.77

  - Ouro-1.4B T=7: 65.28

  - Ouro-1.4B T=8: 64.49

- **ARC-C (25-shot) vs T (Table 10)** — lm-eval-harness logprobs.

  - T=1: 37.63

  - T=2: 54.86

  - T=3: 59.47

  - T=4: 60.92

  - T=5: 58.96

  - T=6: 59.73

  - T=7: 58.96

  - T=8: 58.19

- **ARC-E (8-shot) vs T (Table 10)** — lm-eval-harness logprobs.

  - T=1: 63.85

  - T=2: 80.3

  - T=3: 83.33

  - T=4: 83.96

  - T=5: 82.91

  - T=6: 82.58

  - T=7: 81.99

  - T=8: 82.07

- **CommonsenseQA (10-shot) vs T (Table 10)** — lm-eval-harness logprobs.

  - T=1: 44.64

  - T=2: 67.98

  - T=3: 74.37

  - T=4: 75.43

  - T=5: 75.35

  - T=6: 74.94

  - T=7: 74.28

  - T=8: 73.55

- **HellaSwag (10-shot) vs T (Table 10)** — lm-eval-harness logprobs.

  - T=1: 55.24

  - T=2: 71.15

  - T=3: 74.07

  - T=4: 74.29

  - T=5: 73.72

  - T=6: 72.77

  - T=7: 72.35

  - T=8: 71.6

- **Winogrande (5-shot) vs T (Table 10)** — lm-eval-harness logprobs.

  - T=1: 56.99

  - T=2: 66.69

  - T=3: 71.35

  - T=4: 72.3

  - T=5: 70.32

  - T=6: 71.03

  - T=7: 70.09

  - T=8: 69.3

- **KV-cache sharing at decode (Table 14; GSM8K / MATH-500)** — Ouro-1.4B at T=4; GSM8K score listed, MATH-500 in note; memory reduction 4x for non-full strategies.

  - Full 4x cache: 78.92 (MATH-500 82.40)

  - Last-step-only cache: 78.85 (MATH-500 80.40)

  - Averaged cache: 78.73 (MATH-500 78.52)

  - First-step-only cache: 18.73 (MATH-500 8.43 (collapse))

- **Community lm-eval reproduction (HF discussion #8)** — lm-evaluation-harness; initial run mistakenly applied chat template to base model.

  - Ouro-1.4B MMLU (with chat template): 66.74 (paper 67.35)

  - Ouro-1.4B BBH (with chat template): 60.77 (paper 71.02; without chat template 71.06)

  - Ouro-1.4B GSM8K (with chat template): 60.8 (paper 78.92; matched paper after removing chat template (exact number not posted))

- Loop-count ablation: config.total_ut_steps sets T (modeling_ouro.py loops `for current_ut in range(self.total_ut_steps)`; early_exit_threshold/exit_at_step only select among already-computed steps, they do not save compute). Trained at T=4. Table 10: MMLU 41.21/60.43/66.71/67.45 at T=1/2/3/4, then 66.64/65.77/65.28/64.49 at T=5..8; ARC-C 37.63/54.86/59.47/60.92; HellaSwag 55.24/71.15/74.07/74.29; Winogrande 56.99/66.69/71.35/72.30. GSM8K/MATH500 vs T for the base model are NOT tabulated in the paper (only at T=4). Static-exit baseline steps 1-4 on MMLU appear only in Figure 5; Ponder gate with adaptive-exit training reaches ~66% MMLU at avg exit round 2.5 vs ~64% for the standard gate. Weights 2.87 GB bf16 safetensors.

- Eval code: No official Ouro repo exists: github.com/ByteDance/Ouro and github.com/Ouro-LLM/Ouro (linked from the Thinking model card) both return 404 and ouro-llm.github.io says 'Code (Coming Soon)'. Evals were lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) + evalplus; settings in paper Table 16. Reproduction recipe in https://huggingface.co/ByteDance/Ouro-1.4B/discussions/8

- Baselines available via Ollama: qwen3:1.7b (Ollama; NOTE this is post-trained Qwen3-1.7B, paper used Qwen3-1.7B-Base; import mradermacher/Qwen3-1.7B-Base-GGUF for a faithful baseline), qwen3:4b (same caveat; Base GGUF needed for exact match), qwen2.5:1.5b, qwen2.5:3b (instruct tags on Ollama; not verified this session), llama3.2:1b, llama3.2:3b (instruct tags; not verified this session), gemma3:1b, gemma3:4b (instruct tags; not verified this session)


### Ouro-2.6B (ByteDance, base LoopLM, 48 layers x 4 recurrent steps, 2.6B params)

- **GSM8K** — lm-eval-harness, strict match, 3-shot CoT; Ouro at T=4.

  - Ouro-2.6B R4: 81.58

  - Qwen2.5-3B: 74.6

  - Llama3.2-3B: 67.2

  - Qwen3-4B-Base: 72.86

  - Gemma3-4B: 68.69

  - Qwen2.5-7B: 81.5

  - Llama3.1-8B: 78.17

  - Qwen3-8B-Base: 83.09

  - Gemma3-12B: 77.18

- **MATH500** — In-house, strict match, 5-shot CoT.

  - Ouro-2.6B R4: 90.85

  - Qwen2.5-3B: 42.6

  - Llama3.2-3B: 40.8

  - Qwen3-4B-Base: 59.6

  - Gemma3-4B: 68.6

  - Qwen2.5-7B: 61.2

  - Llama3.1-8B: 52.9

  - Qwen3-8B-Base: 62.3

  - Gemma3-12B: 83.2

- **MMLU** — lm-eval-harness, logprobs, 5-shot.

  - Ouro-2.6B R4: 74.6

  - Qwen2.5-3B: 65.62

  - Llama3.2-3B: 59.69

  - Qwen3-4B-Base: 73.19

  - Gemma3-4B: 58.37

  - Qwen2.5-7B: 74.2

  - Llama3.1-8B: 73.02

  - Qwen3-8B-Base: 76.63

  - Gemma3-12B: 72.14

- **MMLU-Pro** — lm-eval-harness, strict match, 5-shot CoT.

  - Ouro-2.6B R4: 55.73

  - Qwen2.5-3B: 37.87

  - Llama3.2-3B: 33.34

  - Qwen3-4B-Base: 51.4

  - Gemma3-4B: 34.61

  - Qwen2.5-7B: 43.55

  - Llama3.1-8B: 43.24

  - Qwen3-8B-Base: 53.72

  - Gemma3-12B: 49.21

- **BBH** — lm-eval-harness, strict match, 3-shot CoT.

  - Ouro-2.6B R4: 80.46

  - Qwen2.5-3B: 55.37

  - Llama3.2-3B: 39.45

  - Qwen3-4B-Base: 71.14

  - Gemma3-4B: 66.32

  - Qwen2.5-7B: 53.72

  - Llama3.1-8B: 71.56

  - Qwen3-8B-Base: 77.65

  - Gemma3-12B: 78.41

- **ARC-C** — lm-eval-harness, logprobs, 25-shot.

  - Ouro-2.6B R4: 66.4

  - Qwen2.5-3B: 55.46

  - Llama3.2-3B: 52.47

  - Qwen3-4B-Base: 63.65

  - Gemma3-4B: 60.75

  - Qwen2.5-7B: 63.65

  - Llama3.1-8B: 60.75

  - Qwen3-8B-Base: 66.1

  - Gemma3-12B: 72.44

- **HellaSwag** — lm-eval-harness, logprobs, 10-shot.

  - Ouro-2.6B R4: 79.69

  - Qwen2.5-3B: 74.54

  - Llama3.2-3B: 73.09

  - Qwen3-4B-Base: 75.66

  - Gemma3-4B: 75.58

  - Qwen2.5-7B: 79.98

  - Llama3.1-8B: 81.97

  - Qwen3-8B-Base: 79.6

  - Gemma3-12B: 83.68

- **Winogrande** — lm-eval-harness, logprobs, 5-shot.

  - Ouro-2.6B R4: 75.85

  - Qwen2.5-3B: 70.17

  - Llama3.2-3B: 69.14

  - Qwen3-4B-Base: 71.19

  - Gemma3-4B: 71.27

  - Qwen2.5-7B: 76.48

  - Llama3.1-8B: 77.11

  - Qwen3-8B-Base: 76.8

  - Gemma3-12B: 77.74

- **HumanEval / HumanEval+** — evalplus pass@1 (HumanEval+ in note).

  - Ouro-2.6B R4: 78.7 (HumanEval+ 70.70)

  - Qwen2.5-3B: 68.9 (HumanEval+ 62.20)

  - Llama3.2-3B: 29.9 (HumanEval+ 26.20)

  - Qwen3-4B-Base: 77.7 (HumanEval+ 70.70)

  - Gemma3-4B: 34.8 (HumanEval+ 29.30)

  - Qwen2.5-7B: 79.3 (HumanEval+ 70.60)

  - Llama3.1-8B: 38.4 (HumanEval+ 31.10)

  - Qwen3-8B-Base: 84.8 (HumanEval+ 75.30)

  - Gemma3-12B: 46.3 (HumanEval+ 37.20)

- **MBPP / MBPP+** — evalplus pass@1 (MBPP+ in note).

  - Ouro-2.6B R4: 80.4 (MBPP+ 66.60)

  - Qwen2.5-3B: 63 (MBPP+ 54.20)

  - Llama3.2-3B: 50.3 (MBPP+ 39.70)

  - Qwen3-4B-Base: 78.8 (MBPP+ 65.90)

  - Gemma3-4B: 60.6 (MBPP+ 51.10)

  - Qwen2.5-7B: 73.8 (MBPP+ 63.50)

  - Llama3.1-8B: 62.4 (MBPP+ 51.60)

  - Qwen3-8B-Base: 79 (MBPP+ 67.90)

  - Gemma3-12B: 73.5 (MBPP+ 66.10)

- **Per-step sweep (Table 11): ARC-C / ARC-E / C-QA / HellaSwag / MMLU / Winogrande** — 25/8/10/10/5/5-shot logprobs; MMLU score listed, others in note.

  - Ouro-2.6B T=1: 51.55 (ARC-C 47.95, ARC-E 72.39, C-QA 57.58, HellaSwag 68.94, Winogrande 61.48)

  - Ouro-2.6B T=2: 67.63 (62.37, 85.23, 76.90, 77.61, 70.48)

  - Ouro-2.6B T=3: 73.57 (65.36, 87.33, 79.77, 79.12, 74.35)

  - Ouro-2.6B T=4: 74.6 (66.38, 86.95, 81.65, 79.56, 75.53)

  - Ouro-2.6B T=5: 74.43 (65.36, 86.83, 81.24, 79.57, 75.93)

  - Ouro-2.6B T=6: 73.79 (65.02, 86.74, 81.08, 79.63, 75.37)

  - Ouro-2.6B T=7: 72.92 (65.44, 86.57, 80.75, 79.59, 75.77)

  - Ouro-2.6B T=8: 72.24 (64.76, 86.49, 81.08, 79.50, 74.59)

- Loop-count ablation: Table 11 (trained at T=4): MMLU 51.55/67.63/73.57/74.60 at T=1..4, 74.43/73.79/72.92/72.24 at T=5..8 (degrades more gently than 1.4B under extrapolation). ARC-C 47.95/62.37/65.36/66.38; HellaSwag 68.94/77.61/79.12/79.56. GSM8K/MATH500 vs T not tabulated. Weights 5.34 GB bf16; ~1.5 MiB/token KV in bf16 with 4-step full cache (Raschka gallery).

- Eval code: Same as Ouro-1.4B: no official repo (404s); lm-evaluation-harness + evalplus per paper Table 16.

- Baselines available via Ollama: qwen3:4b, qwen3:8b (Ollama; post-trained, paper used Base), qwen2.5:3b, qwen2.5:7b, llama3.2:3b, llama3.1:8b, gemma3:4b, gemma3:12b


### Ouro-1.4B-Thinking (SFT reasoning model, T=4)

- **AIME24 pass@1 / pass@10** — In-house harness, LLM-as-judge, temp=1.0, top_p=0.7 (Table 17). Number of samples for pass@1 and max generation length NOT stated in paper; model card quick-start uses max_new_tokens=512, temp 1.0, top_p 0.7; SFT context 32K.

  - Ouro-1.4B-Thinking-R4: 65 (pass@10 83.3)

  - Ouro-2.6B-Thinking-R4: 64.7 (pass@10 90.0)

  - Qwen3-1.7B (thinking): 32 (pass@10 55.6)

  - Qwen3-4B (thinking): 61.3 (pass@10 75.0)

  - Qwen3-8B (thinking): 73 (pass@10 86.7)

  - DeepSeek-R1-Distill-Qwen-1.5B: 29.6 (pass@10 66.7)

  - DeepSeek-R1-Distill-Qwen-7B: 57.3 (pass@10 83.3)

- **AIME25 pass@1 / pass@10** — Same protocol as AIME24.

  - Ouro-1.4B-Thinking-R4: 46.3 (pass@10 73.3)

  - Ouro-2.6B-Thinking-R4: 50.3 (pass@10 76.7)

  - Qwen3-1.7B: 22 (pass@10 33.3)

  - Qwen3-4B: 51.3 (pass@10 63.3)

  - Qwen3-8B: 66.7 (pass@10 81.3)

  - DeepSeek-R1-Distill-Qwen-1.5B: 23 (pass@10 43.33)

  - DeepSeek-R1-Distill-Qwen-7B: 36 (pass@10 73.3)

- **OlympiadBench / BeyondAIME / HLE / SuperGPQA / GPQA (Table 9)** — In-house harness, LLM-as-judge, temp 1.0, top_p 0.7. OlympiadBench score listed; others in note in that order.

  - Ouro-1.4B-Thinking-R4: 71.6 (BeyondAIME 34.0, HLE 5.21, SuperGPQA 47.4, GPQA 45.5)

  - Ouro-2.6B-Thinking-R4: 76.4 (39.0, 5.58, 53.7, 52.7)

  - Qwen3-1.7B: 56.4 (15.0, 4.13, 35.9, 34.0)

  - Qwen3-4B: 73.2 (31.0, 5.21, 51.9, 54.5)

  - Qwen3-8B: 75.3 (38.0, 2.22, 48.0, 59.1)

  - DeepSeek-R1-Distill-Qwen-1.5B: 56.44 (9.0, 4.2, 26.5, 33.2)

  - DeepSeek-R1-Distill-Qwen-7B: 72 (30.0, 5.14, 46.6, 51.0)

- **Per-step sweep (Table 12): AIME24 score listed; OlympiadBench / SuperGPQA / AIME25 in note** — Trained at T=4; peaks at T=4-5.

  - T=1: 0 (OlympiadBench 2.22, SuperGPQA 2.03, AIME25 0.33)

  - T=2: 37.33 (59.70, 33.07, 25.00)

  - T=3: 62.33 (70.67, 44.50, 43.33)

  - T=4: 65 (71.55, 47.37, 46.30)

  - T=5: 60.67 (72.30, 48.73, 47.00)

  - T=6: 50.67 (69.48, 46.15, 43.00)

  - T=7: 42.33 (69.04, 45.29, 41.00)

  - T=8: 38.67 (66.81, 42.88, 38.00)

- Loop-count ablation: Table 12: AIME24 0.00/37.33/62.33/65.00 at T=1..4, then 60.67/50.67/42.33/38.67 at T=5..8; OlympiadBench 2.22/59.70/70.67/71.55/72.30/69.48/69.04/66.81; SuperGPQA 2.03/33.07/44.50/47.37/48.73/46.15/45.29/42.88; AIME25 0.33/25.00/43.33/46.30/47.00/43.00/41.00/38.00. T=1 is near-zero (model is unusable at one pass). No MATH500/GSM8K reported for the Thinking models.

- Eval code: None released (in-house harness). Model card: https://huggingface.co/ByteDance/Ouro-1.4B-Thinking (links github.com/Ouro-LLM/Ouro which 404s).

- Baselines available via Ollama: qwen3:1.7b (thinking mode; Ollama), qwen3:4b / qwen3:4b-thinking, qwen3:8b, deepseek-r1:1.5b, deepseek-r1:7b (Qwen distills; tags not verified this session)


### Ouro-2.6B-Thinking (SFT reasoning model, T=4)

- **Per-step sweep (Table 13): AIME24 score listed; OlympiadBench / SuperGPQA / AIME25 in note** — In-house harness, LLM-as-judge, temp 1.0, top_p 0.7; trained at T=4.

  - T=1: 3 (OlympiadBench 18.96, SuperGPQA 15.66, AIME25 2.00)

  - T=2: 52 (68.59, 48.58, 40.67)

  - T=3: 70.33 (75.56, 56.70, 50.67)

  - T=4: 64.7 (76.44, 53.68, 50.30)

  - T=5: 57 (71.85, 56.45, 49.33)

  - T=6: 56.33 (69.19, 55.44, 46.00)

  - T=7: 49.67 (57.63, 53.32, 38.00)

  - T=8: 39 (39.26, 46.84, 24.33)

- **Table 9 headline (see Ouro-1.4B-Thinking entry for baselines)** — Same protocol.

  - Ouro-2.6B-Thinking-R4 AIME24 pass@1: 64.7

  - Ouro-2.6B-Thinking-R4 AIME25 pass@1: 50.3

  - Ouro-2.6B-Thinking-R4 OlympiadBench: 76.4

  - Ouro-2.6B-Thinking-R4 GPQA: 52.7

  - Ouro-2.6B-Thinking-R4 SuperGPQA: 53.7

- Loop-count ablation: Table 13: AIME24 3.00/52.00/70.33/64.70/57.00/56.33/49.67/39.00 for T=1..8 (note peak at T=3, not the trained T=4); OlympiadBench 18.96/68.59/75.56/76.44/71.85/69.19/57.63/39.26; SuperGPQA 15.66/48.58/56.70/53.68/56.45/55.44/53.32/46.84. Extrapolation beyond T=4 collapses faster than for the 1.4B model.

- Eval code: None released; model card https://huggingface.co/ByteDance/Ouro-2.6B-Thinking (SFT data: ~8.3M examples, OpenThoughts3/AceReason etc.; recommends transformers==4.54.1 which discussion #14 says is broken, use >=4.57.1).

- Baselines available via Ollama: qwen3:4b (thinking), qwen3:8b, deepseek-r1:7b (not verified this session)


### Huginn-0125 (tomg-group-umd; 3.5B depth-recurrent: 2 prelude + 4 recurrent + 2 coda layers, n_embd 5280, 55 heads; 0.8T tokens; mean_recurrence=32)

- **ARC-E / ARC-C / HellaSwag / MMLU / OBQA / PiQA / SciQ / WinoGrande (Table 1, zero-shot lm-eval-harness)** — Zero-shot, lm-eval-harness, pure bf16. ARC-C score listed; the others in note in the order ARC-E, HellaSwag, MMLU, OBQA, PiQA, SciQ, WinoGrande.

  - Huginn r=1 (Table 7): 24.06 (ARC-E 34.89, HellaSwag 29.34, MMLU 23.60, OBQA 26.80, PiQA 55.33, SciQ 47.10, WinoGrande 49.41)

  - Huginn r=4: 27.99 (49.07, 43.46, 23.39, 28.20, 64.96, 80.00, 55.24)

  - Huginn r=8: 35.15 (65.11, 58.54, 25.29, 35.40, 73.45, 92.10, 55.64)

  - Huginn r=16: 37.71 (69.49, 64.67, 31.25, 37.60, 75.79, 93.90, 57.77)

  - Huginn r=32: 38.23 (69.91, 65.21, 31.38, 38.80, 76.22, 93.50, 59.43)

  - OLMo-2-1124-7B (4T): 57.42 (82.79, 80.50, 60.56, 46.20, 81.18, 96.40, 74.74)

  - OLMo-7B-0724 (2.75T): 43.43 (74.28, 77.76, 50.18, 41.60, 80.69, 95.70, 67.17)

  - OLMo-7B-0424 (2.05T): 45.05 (75.13, 77.24, 47.46, 41.60, 80.09, 96.00, 68.19)

  - OLMo-7B (2.5T): 40.27 (68.81, 75.52, 28.39, 42.20, 80.03, 88.50, 67.09)

  - OLMo-1B (3T): 30.72 (57.28, 63.00, 24.33, 36.40, 75.24, 78.70, 59.19)

  - Amber 7B (1.2T): 37.2 (65.70, 72.54, 26.77, 41.00, 78.73, 88.50, 63.22)

  - Pythia-2.8b: 32.51 (58.00, 59.17, 25.05, 35.40, 73.29, 83.60, 57.85)

  - Pythia-6.9b: 34.64 (60.48, 63.32, 25.74, 37.20, 75.79, 82.90, 61.40)

  - Pythia-12b: 34.64 (63.22, 66.72, 24.01, 35.40, 75.84, 84.40, 63.06)

  - Fixed-depth baseline 0.18T (Table 7): 26.96 (ARC-E 46.42, HellaSwag 37.34, MMLU 24.16, OBQA 29.60, PiQA 64.47, SciQ 73.20, WinoGrande 51.78, GSM8K-CoT 1.82/2.20)

  - Huginn early ckpt 0.18T r=32 (Table 7): 29.18 (ARC-E 53.62, HellaSwag 48.80, MMLU 25.59, GSM8K-CoT 9.02/10.24)

- **GSM8K (flexible/strict) and GSM8K-CoT (flexible/strict), Minerva MATH, MathQA (Table 2)** — lm-eval-harness; GSM8K zero-shot and 8-shot CoT; CoT run with chat template and 8 few-shot examples as multi-turn; 'w/ sys prompt' = system message 'You are a helpful assistant that can assist users with mathematical reasoning.' (GitHub README). Score listed = GSM8K-CoT strict; full numbers in note.

  - Huginn r=32 w/ sys prompt: 42.08 (GSM8K 24.87/38.13; GSM8K-CoT 34.80/42.08; Minerva MATH 11.24; MathQA 27.97)

  - Huginn r=32 w/o sys prompt: 34.57 (GSM8K 28.05/28.20; GSM8K-CoT 32.60/34.57; Minerva MATH 12.58; MathQA 26.60)

  - Huginn r=1: 0 (GSM8K-CoT 0.00/0.00 (Table 7))

  - Huginn EMA checkpoint r=64 (text, Sec. 4): 38.59 ('47.23% flexible (38.59% strict)' at r=64; paper does not say whether this is the plain or CoT GSM8K variant)

  - OLMo-2-1124-7B: 66.19 (GSM8K 66.72/66.79; CoT 61.94/66.19; Minerva 19.08; MathQA 37.59)

  - OLMo-7B-0724: 28.89 (GSM8K 28.66/28.73; CoT 28.89/28.89; Minerva 5.62; MathQA 27.84)

  - OLMo-7B-0424: 26.23 (GSM8K 27.07/27.29; CoT 26.23/26.23; Minerva 5.56; MathQA 28.48)

  - OLMo-7B: 7.28 (GSM8K 4.02/4.09; CoT 6.07/7.28; Minerva 2.12; MathQA 25.26)

  - OLMo-1B: 2.58 (GSM8K 1.82/2.27; CoT 1.59/2.58; Minerva 1.60; MathQA 23.38)

  - Amber 7B: 5.16 (GSM8K 3.94/4.32; CoT 3.34/5.16; Minerva 1.94; MathQA 25.26)

  - Pythia-12b: 4.62 (GSM8K 3.49/4.62; CoT 3.34/4.62; Minerva 2.56; MathQA 25.80)

- **MBPP / HumanEval pass@1 (Table 3)** — HumanEval zero-shot no chat template; repo README eval used humaneval_instruct with do_sample, temperature 0.2, top_p 0.95. MBPP score listed, HumanEval in note.

  - Huginn r=32: 24.8 (HumanEval 23.17)

  - starcoder2-3b: 43 (HumanEval 31.09)

  - starcoder2-7b: 43.8 (HumanEval 31.70)

  - OLMo-2-1124-7B: 21.8 (HumanEval 10.36)

  - OLMo-7B-0724: 25.6 (HumanEval 20.12)

  - OLMo-7B-0424: 21.2 (HumanEval 16.46)

  - OLMo-7B: 15.6 (HumanEval 12.80)

  - Amber: 19.6 (HumanEval 13.41)

  - Pythia-6.9b: 7.92 (HumanEval 5.60)

- Loop-count ablation: Recurrence set via model arg num_steps (generate) or mean_recurrence (config/lm-eval). Table 1: ARC-C 27.99/35.15/37.71/38.23 and HellaSwag 43.46/58.54/64.67/65.21 at r=4/8/16/32; r=1 gives ARC-C 24.06, HellaSwag 29.34, GSM8K-CoT 0.00 (Table 7). GSM8K-CoT/HumanEval vs r curves are only shown in Figure 7 (no table); HellaSwag saturates by r=8, GSM8K and HumanEval keep improving to r=32; README says gains continue to about r=64 then flatten; EMA checkpoint at r=64 reaches GSM8K 47.23 flexible / 38.59 strict. Materialized params = num_steps*1.5B + 2B. HF repo is 15.65 GB of fp32 safetensors (with a duplicated tied embedding per README); run in bf16 (~7 GB RAM). Cache options: lookup_strategy 'compress-s16' to cap KV memory, 'latest-m4' for adaptive compute; adaptive exit criteria kl/entropy-diff/latent-diff/argmax-stability with exit_threshold=5e-4.

- Eval code: https://github.com/seal-rg/recurrent-pretraining (evaluate_raven/: local_lm_eval.py, saturation_eval_dist.py, record_steps_in_bench.py; README gives lm_eval commands, e.g. `lm_eval --model hf --model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32 --tasks hellaswag --batch_size=auto --num_fewshot=0`)

- Baselines available via Ollama: olmo2:7b (Ollama library; = OLMo-2-1124-7B), starcoder2:3b, starcoder2:7b (Ollama library; not verified this session), Pythia and Amber are not on Ollama


### Nanbeige4.2-3B / Nanbeige4.2-3B-Base (Nanbeige; looped transformer, 22 layers x num_loops=2, 4B total / 3B non-embedding, 28T tokens; the only looped model with merged llama.cpp support)

- **Base-model table (tech report Table 1 / Base model card): GSM8K score listed; BBH / MBPP / MMLU-Pro / SuperGPQA / GPQA in note** — Few-shot settings and harness NOT stated on the card or in the report.

  - Nanbeige4.2-3B-Base: 92.7 (BBH 81.6, MBPP 67.6, MMLU-Pro 63.8, SuperGPQA 35.2, GPQA 53.3)

  - Qwen3.5-4B-Base (5B total): 84.4 (79.1, 57.1, 51.8, 32.1, 43.1)

  - Nanbeige4(.1)-3B-Base: 85.9 (70.7, 60.7, 47.6, 24.8, 36.2)

  - Gemma4-E4B-Base (8B total): 61.8 (62.5, 53.5, 37.6, 23.3, 27.5)

- **Instruct/thinking model reasoning rows (model card): GPQA-Diamond listed; HMMT-Feb-2026 / IMO-Answer-Bench / LiveCodeBench-V6 / HLE w/o search / SciCode / SWE-Bench Verified in note** — Thinking mode, preserve_thinking=true, temperature 0.6, max new tokens 131,072 (reasoning); agentic tasks temp 1.0, 65,536 tokens.

  - Nanbeige4.2-3B: 87.4 (HMMT 82.8, IMO-AB 67.3, LCB-V6 72.5, HLE 17.8, SciCode 35.6, SWE-V 63.6)

  - Qwen3.5-9B: 81.7 (69.6, 56.3, 65.6, 12.5, 32.7, 53.1)

  - Qwen3.5-4B: 78.2 (60.6, 46.8, 55.8, 6.8, 22.7, 38.8)

  - Gemma4-12B: 78.8 (51.5, 54.5, 72.0, 14.8, 38.2, 44.2)

  - Gemma4-E4B: 60.6 (24.2, 24.0, 55.3, 4.0, 24.9, 14.0)

- Loop-count ablation: No per-loop accuracy numbers published. Tech report only says the two-pass configuration 'retains approximately 75% of the token efficiency' of a standard transformer and that more passes give marginal gains; from-scratch looped training beat upcycling (no numbers). config.json: num_loops=2, loop_loss_weights=[], skip_loop_final_norm=false. Loops are fixed in the GGUF metadata; whether editing num_loops at runtime works is unconfirmed.

- Eval code: None released. Deployment: llama.cpp PR #25994 merged 2026-07-27 (num_loops read from GGUF metadata); README documents convert_hf_to_gguf + Q4_K_M + Ollama Modelfile; mlx-lm has nanbeige.py.

- Baselines available via Ollama: Qwen3.5-4B / Qwen3.5-9B and Gemma4 tags on Ollama (availability not verified this session), Community Ollama uploads of Nanbeige4.2-3B exist (ndavat/Nanbeige4.2-3B, 1,548 pulls) but there is no official library entry; Nanbeige's own ollama fork branch nanbeige42 is documented instead


### Other looped models with weights (survey; none are good local targets)

- **Mixture-of-Recursions (raymin0223) — few-shot avg from paper (approx. 118M non-emb scale row; harness lm-eval, LAMBADA/HellaSwag/PIQA/WinoGrande/ARC-E/ARC-C/MMLU)** — Only 360M Vanilla/Recursive/MoR checkpoints released, on Google Drive, not HF format. Numbers below are from the paper's main table as summarized by WebFetch; scale mapping not fully verified.

  - MoR (expert-choice, Nr=3): 42.6 (val NLL 2.7925)

  - Recursive baseline (Nr=3): 41.5 (val NLL 2.8466)

  - Vanilla 315M: 42.3 (val NLL 2.7824)

- **LoopFormer (armenjeddi, ICLR 2026)** — 17 HF checkpoints of 0.1-1B (NanoGPT fork; e.g., LoopFormer-3block-8iterations 0.3B, NanoGPT-Base24 1B). No benchmark numbers on the collection page.

  - LoopFormer-3block-8iterations (0.3B): 0 (score not published on HF collection page; research checkpoint only)

- **IQuest-Coder-V1-40B-Loop-Instruct (dual-iteration LoopCoder)** — 40B params: too large for 16 GB. mlx-lm has iquestloopcoder.py.

  - IQuest-Coder-V1-40B-Loop-Instruct SWE-Bench Verified: 76.2 (LiveCodeBench v6 81.1, BigCodeBench 49.9 (from search snippet; not verified against the card))

- **LoopRPT (arXiv 2603.19714)** — Fine-tunes Ouro-1.4B/2.6B on Omni-Math; no checkpoints or code URLs found.

  - LoopRPT: 0 (no per-step GSM8K/MATH500 tables; no release)

- Loop-count ablation: MoR/LoopFormer: research-scale; no numbers replicable against an Ollama baseline. LoopFormer supports budget-conditioned depth (elastic loops) by design.

- Eval code: MoR: https://github.com/raymin0223/mixture_of_recursions ; LoopFormer: https://github.com/armenjeddi/loopformer


### Feasible replications on this machine (agent assessment)

- PREREQUISITE (disk): 3.5 GB free is not enough. Ouro-1.4B alone is 2.87 GB (bf16 safetensors); Ouro-2.6B 5.34 GB; huginn-0125 is 15.65 GB (stored fp32 with duplicated embedding); qwen3:1.7b on Ollama 1.4 GB (q4_K_M) / 2.2 GB (q8_0); mradermacher/Qwen3-1.7B-Base-GGUF Q8_0 1.83 GB. Free >= 10 GB for the Ouro-1.4B plan, >= 25 GB if Huginn is included.

- DEPLOYMENT PLAN (confirmed constraints): Ouro cannot run under Ollama/llama.cpp (no llama.cpp issue/PR; ollama/ollama#14252 open since 2026-02-14, no PR) and mlx-lm main has no `ouro` model file (mlx-community/Ouro-1.4B-4bit exists but is likely unloadable; unconfirmed). Run Ouro with `uv venv` + torch (MPS) + transformers>=4.57.1 (HF discussion #14: README's `<4.56` advice is stale; 4.54-4.55 break), `trust_remote_code=True`, `torch_dtype=bfloat16`, batch_size=1 (discussion #9: batched generation corrupts non-longest sequences; sdpa crashed, eager gave whitespace, only flash-attn worked). Set loops with `config = AutoConfig.from_pretrained(..); config.total_ut_steps = T` before `from_pretrained`. Serve it OpenAI-compatible with a ~50-line FastAPI wrapper on a port next to Ollama (11434); evaluate baselines through Ollama's /v1 endpoint with lm-eval `--model local-completions` (generative tasks only; logprob tasks need the HF backend). Qwen3-1.7B-Base is NOT in Ollama's library (qwen3:1.7b is the post-trained hybrid model, enable_thinking defaults True) — import mradermacher/Qwen3-1.7B-Base-GGUF via a Modelfile for a faithful base baseline.

- REPLICATION 1 (best signal, ~1-2 h): Ouro-1.4B base, MMLU 5-shot logprob at T=1,2,3,4 (+5,6 for extrapolation) with lm-eval `--model hf --model_args pretrained=ByteDance/Ouro-1.4B,trust_remote_code=True,dtype=bfloat16 --tasks mmlu --num_fewshot 5 --device mps --batch_size 1 --limit 0.05` (~700 questions, SE ~1.7 pts), one run per T (edit config or pass total_ut_steps=T as a model_arg — the latter should be applied to the config by from_pretrained but is unconfirmed). Do NOT apply a chat template (discussion #8). Expected (paper Table 10): 41.21 / 60.43 / 66.71 / 67.45 / 66.64 / 65.77. Also cheap: HellaSwag 10-shot subset expected 55.24/71.15/74.07/74.29; Winogrande 5-shot 56.99/66.69/71.35/72.30. Time per T scales with T; T=1 is 4x faster than T=4.

- REPLICATION 2 (~2-3 h): Ouro-1.4B base vs Qwen3-1.7B-Base on GSM8K, 200-problem subset, 3-shot CoT strict match (lm-eval task gsm8k_cot with --num_fewshot 3, no chat template, greedy, max_gen_toks 256; --limit 200 gives SE ~2.9 pts). Expected at T=4: Ouro-1.4B 78.92 (paper Table 7; community lm-eval repro landed near the paper once the chat template was removed, 60.80 with it), Qwen3-1.7B-Base 70.28, Qwen2.5-1.5B 60.73, Llama3.2-1B 7.05, Llama3.2-3B 67.20, Gemma3-4B 68.69, Qwen3-4B-Base 72.86. Running the same subset at T=1,2,3 is cheap (T=1 ~4x faster) but has NO published GSM8K reference; the MMLU curve (41->60->67->67) predicts a steep collapse at T=1 and most of the gain by T=2-3. Throughput on M4/MPS for a 1.4B model at 4 loops is unmeasured; budget ~1 h per 200 problems at T=4 (estimate, not a source).

- REPLICATION 3 (~1 h): Ouro-1.4B BBH 3-shot CoT strict match on a 3-4 subtask subset (lm-eval bbh_cot_fewshot_*), T=4, no chat template. Expected full-suite 71.02 (paper) and 71.06 (community repro without chat template) vs Qwen3-1.7B-Base 53.51; the per-subtask paper numbers are not published, so compare Ouro vs baseline on the identical subtasks.

- REPLICATION 4 (~1 h, if disk allows the 5.34 GB download): Ouro-2.6B MMLU 5-shot subset at T=1..4: expected 51.55 / 67.63 / 73.57 / 74.60 (Table 11); GSM8K 200-subset at T=4 expected 81.58 vs Qwen3-4B-Base 72.86 / Qwen3-8B-Base 83.09 (qwen3:4b, qwen3:8b on Ollama are post-trained, so use Base GGUFs). ~2x slower than 1.4B; RAM fine (5.3 GB weights + ~1.5 MiB/token KV).

- REPLICATION 5 (Huginn r-sweep, ~2-3 h, needs ~16 GB free disk for the fp32 download; loads to ~7 GB in bf16): lm-eval `--model hf --model_args pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence=R --tasks arc_challenge,hellaswag --num_fewshot 0 --device mps --batch_size 1 --limit 200` for R in {1,4,8,16,32}. Expected (Tables 1 and 7, full sets): ARC-C 24.06 / 27.99 / 35.15 / 37.71 / 38.23; HellaSwag 29.34 / 43.46 / 58.54 / 64.67 / 65.21; ARC-E 34.89 / 49.07 / 65.11 / 69.49 / 69.91. Compare with olmo2:7b via Ollama on the same subset (paper: ARC-C 57.42, HellaSwag 80.50, but Ollama's olmo2 is the instruct tag, so expect drift). r=32 is ~50B-dense-equivalent compute per token; logprob tasks are prefill-only so this stays tractable.

- REPLICATION 6 (Huginn GSM8K-CoT, borderline, ~2-4 h): 50-100 problem subset, gsm8k_cot 8-shot with chat template as multi-turn and system prompt 'You are a helpful assistant that can assist users with mathematical reasoning.', mean_recurrence=16 and 32, max_gen_toks 256, use cache_kwargs lookup_strategy 'compress-s16' if KV memory is tight. Expected at r=32 (full set): 34.80 flexible / 42.08 strict with sys prompt, 32.60/34.57 without; r=1 gives 0.00; the EMA checkpoint reaches 47.23/38.59 at r=64. Baseline via Ollama olmo2:7b (paper OLMo-2-1124-7B: 61.94/66.19). Generation at r=32 will be slow (a few tok/s at best on M4; unmeasured), so keep the subset small.

- REPLICATION 7 (Ollama-native looped model, ~30-45 min, no loop knob): Nanbeige4.2-3B. Build llama.cpp master (PR #25994 merged 2026-07-27), convert Nanbeige/Nanbeige4.2-3B-Base (8.3 GB bf16 download) to GGUF and quantize Q4_K_M (~2.5 GB, estimate), `ollama create` from a Modelfile (stock Ollama compatibility with the nanbeige arch is unconfirmed; Nanbeige documents its own ollama fork branch nanbeige42; community upload ndavat/Nanbeige4.2-3B has 1.5K pulls). Run a 200-problem GSM8K subset (few-shot setting unpublished; use 8-shot CoT). Expected full-set: 92.7 (Nanbeige4.2-3B-Base) vs Qwen3.5-4B-Base 84.4, Gemma4-E4B-Base 61.8. num_loops=2 is fixed in GGUF metadata; there is no published 1-loop vs 2-loop accuracy to replicate.

- NOT FEASIBLE in a few hours on this machine: Ouro-Thinking AIME24/25/OlympiadBench per-T tables (Table 12/13) — 30 AIME problems x long thinking traces (SFT context 32K, max tokens unstated, temp 1.0/top_p 0.7, LLM-as-judge, unknown sample count) at 4 loops on MPS is >10 h per T (estimate); MATH500/GSM8K are not reported for the Thinking models, so a short-budget MATH500 subset would have no reference number. Ouro KV-cache-sharing (Table 14) is not exposed by modeling_ouro.py. Full 5-shot MMLU (14K x 4 logprob passes at 4 loops) and 25-shot ARC-C are too slow on MPS; use --limit subsets.

- NOT CONFIRMED / CAVEATS: (a) no official Ouro code repo or eval scripts exist (ByteDance/Ouro and Ouro-LLM/Ouro 404; project page 'Code (Coming Soon)'); (b) Ouro on MPS is untested in any source found — the MPS attention path (eager vs sdpa) may hit the discussion-#9 mask bug even at batch 1; (c) baseline numbers differ between paper Tables 7 and 8 for identical models (Qwen3-4B BBH 70.95 vs 71.14, HumanEval 77.40 vs 77.70; Gemma3-4B ARC-C 60.92 vs 60.75, Winogrande 71.07 vs 71.27) and Gemma3-1B MATH500=41.00 looks like a typo; (d) Ouro per-step tables give MMLU 67.45 at T=4 vs 67.35 in Table 7; (e) Huginn r-sweep for GSM8K/HumanEval exists only as Figure 7, no table; (f) Ollama tags other than qwen3:1.7b/4b, olmo2, and the nanbeige community uploads were not individually verified this session; (g) whether lm-eval's `--model local-completions` works against Ollama's /v1/completions was not verified.

## 5. Hands-on findings on this Mac (verified 2026-09-05/06)

- **Ollama 0.33.3** (Homebrew) bundles a llama.cpp runner that already contains the `nanbeige` looped architecture (string check on `libexec/lib/ollama/llama-server`); `ollama pull hf.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M` runs the model natively (`/api/show` reports `nanbeige.num_loops = 2`, capabilities: tools, thinking). ~21 tok/s decode on the M4 at Q4_K_M.
- **`nanbeige.num_loops` override**: copying the GGUF and rewriting the key to 1 (gguf-py `gguf_set_metadata`) doubles decode speed (~40 tok/s) but the output is degenerate ("answer answer answer…"). Nanbeige4.2 was trained with two loops and `loop_loss_weights=[]`, so a single pass is not a usable model — there is no free early exit, unlike Ouro's trained exit gate.
- **Ouro-1.4B on MPS** works with transformers 4.57.6 + torch 2.14, sdpa attention, bf16, batch 1: T=4 ≈ 7 tok/s, T=1/2 ≈ 13 tok/s. transformers 5.x breaks the remote code (separate `.venv-tf4`). Loop count is changed per request by setting both `model.config.total_ut_steps` and `model.model.total_ut_steps` (the inner module caches the value at init).
- **Qwen-based looped models with weights**: only two families exist. (a) `mshapiro123/recurrent-qwen2.5-0.5b-{full-block,natural-keeper,r16-adapter}` — Qwen2.5-0.5B-Instruct retrofit, forced depth via `generate(max_loops=T)`, no KV cache, trained on a synthetic symbolic task; loops=1 reproduces the base exactly (confirmed: 17+26 → 43), loops=3 degrades general answers (confirmed). (b) `Thrillcrazyer/Qwen3_1.7B_LoopUS` (Looped Depth Up-Scaling, Qwen3-1.7B-Base, 4.07 GB) — custom `lds` architecture whose modeling code lives only in the GitHub repo; no math benchmarks published.
- **Traps**: `ollama.com/zoecohn4/Ouro` is an unrelated 8B llama-family Q4_0 with a "snarky sidekick" system prompt, not ByteDance Ouro. `scpalmetto/Ouro-2.6B-Thinking-Fixed` GGUF was re-labelled `LlamaForCausalLM`, so it encodes a single pass, not the loop.
- **Sizes** (all ungated, Apache-2.0): Ouro-1.4B 2.87 GB bf16, Ouro-2.6B 5.34 GB, Nanbeige4.2-3B 8.34 GB bf16 (Q4_K_M 2.7 GB), Huginn-0125 15.65 GB fp32.

## 6. The Qwen question, answered

The user's preference was a looped model "built on top of a trained Qwen model". Exhaustive Hugging Face
and arXiv search (see §2 and the workflow transcripts) finds exactly **two** families of Qwen-derived
looped models with released weights, and neither is a drop-in:

1. **LoopUS** — `Thrillcrazyer/Qwen3_1.7B_LoopUS` (+ `_SFT`, `Ver2.1`, and Qwen3-4B / Qwen3-8B / Phi-4 /
   TinyLlama / EXAONE siblings). Post-training "Looped Depth Up-Scaling" converts Qwen3-1.7B into
   encoder (layers 0-1) → weight-shared looped reasoning block (layers 2-26, up to `N=20` iterations with
   a confidence-head early exit) → decoder (layer 27). 4.07 GB bf16, Apache-2.0, 2.03B params.
   Caveats: the `lds` architecture ships **no** modeling code on the Hub (no `auto_map`, so
   `trust_remote_code` does not help) — the classes live only in `github.com/Thrillcrazyer/LoopUS`,
   which is vendored here under `third_party/LoopUS`. Its loader defaults to **CPU fp32**, with no MPS
   branch, so the shim passes `device_map="mps"` explicitly. No GGUF or MLX port exists, so Ollama
   cannot serve it natively.
2. **Shapiro's retrofit** — `mshapiro123/recurrent-qwen2.5-0.5b-{full-block,natural-keeper,r16-adapter}`.
   Qwen2.5-0.5B-Instruct split into prelude (0-5) / weight-tied looped block (6-17) / coda (18-23), forced
   depth via `generate(max_loops=T)`, `use_cache` disabled. Honest model card: competence is demonstrated
   on a synthetic symbolic family only, general use is meant to run at T=1, and deep forced loops
   degrade general behaviour. Confirmed locally: T=1 answers correctly, T=3 wanders.

**Consequence for evaluation.** LoopUS reports only ARC / HellaSwag / PIQA / WinoGrande / OBQA / MMLU /
LAMBADA and perplexity — **no GSM8K or MATH numbers exist to replicate**. So the Qwen track cannot be a
replication; it has to be an original measurement. The useful experiment is looped-vs-its-own-base on
identical prompts: `Qwen3_1.7B_LoopUS` at several recursion budgets against `Qwen3-1.7B-Base`, which is
already served by Ollama and already measured here at 68.0% on the GSM8K subset. Everything else worth
replicating (Ouro's 78.92, Huginn's r-sweep, Nanbeige's 92.7) comes from models trained looped from
scratch, not from Qwen.

Third-party conversions of *other* bases are better documented than either Qwen option: McLeish et al.
(`smcleish/Recurrent-Llama-3.2-train-recurrence-{16,32}`) convert Llama-3.2-1B with a Huginn-style
4/6/4 split, publish GSM8K 47.6 → 56.2 at recurrence 32, ship a matched non-recurrent baseline, and
expose recurrence as a per-call `num_steps` kwarg. They are stored fp32 (5.54 GB), so they need disk
freed before use.

## 7. What one loop actually costs (observed)

Forcing both looped models down to a single pass fails in two completely different ways, and the
difference tracks how each was trained:

- **Ouro-1.4B at T=1** stays fluent and well-formed but reasons incorrectly. Real GSM8K outputs:
  *"The total cost of the trip is $300. Half of the cost is $300 / 2 = $150. The amount John is missing
  is $300 - $150 = $150."* (gold 100), and *"12 months = 12 x 12 = 144. 144 - 12 = 132."* (gold 30).
  Grammatical, correctly formatted, arithmetically coherent step to step — and wrong. Ouro was trained
  with an entropy-regularized exit gate active at every recurrent step, so every step's hidden state is
  decodable; what the extra loops buy is reasoning depth, not language.
- **Nanbeige4.2-3B at num_loops=1** collapses into degenerate repetition
  (`"** 達 answer answer answer answer …"`) at roughly double the decode speed. Its two-pass structure
  was fixed during training (`loop_loss_weights=[]`, no per-loop supervision), so a single pass is not a
  shallower model, it is an unfinished one.

Test-time depth scaling is therefore a property of the training recipe rather than of looping per se.
A looped architecture only gives you a usable compute/accuracy dial if intermediate depths were
supervised — which is exactly the design difference between Ouro's learned exit gate and Nanbeige's
fixed double pass.

## 8. A scoring artifact that reverses the headline result

While making the replication methodologically exact I hit something worth stating plainly, because it
would silently corrupt any GSM8K comparison between these models.

lm-evaluation-harness's `gsm8k_cot` **strict-match** filter is

```
regex_pattern: The answer is (\-?[0-9\.\,]+).      then take_first
```

The capture class holds only digits, commas and dots, so it cannot match `The answer is $250.` — a
dollar sign stops it dead. The same task's metric block then declares

```
regexes_to_ignore: [",", "\$", "(?s).*#### ", "\.$"]
```

which shows the authors intended a currency prefix to be irrelevant to correctness. But that list is
applied to the string the regex already captured, and on a `$`-prefixed answer the regex captures
nothing, so the ignore rule never runs. The declared intent and the implemented behaviour disagree.

That would be a curiosity if the two models formatted answers alike. They do not:

| | `$`-formatted answers | correct answers discarded by the quirk |
|---|---|---|
| Ouro-1.4B (T=4) | 23.0% | 37 of 200 |
| Qwen3-1.7B-Base | 9.5% | 13 of 200 |

Ouro writes currency answers two and a half times as often, so the verbatim rule fines it roughly three
times as heavily. Scoring the *same saved generations* four ways:

| scorer | Ouro-1.4B T=4 | Qwen3-1.7B-Base | gap |
|---|---|---|---|
| our extractor (strips `$`) | 80.0 | 68.0 | **+12.0** |
| lm-eval strict-match, verbatim | 60.5 | 59.0 | **+1.5** |
| strict-match with `$` allowed | 79.0 | 65.5 | **+13.5** |
| lm-eval flexible-extract | 80.0 | 71.5 | **+8.5** |
| *paper (full test set)* | *78.92* | *70.28* | *+8.64* |

Three of the four scorers agree the looped 1.4B beats the conventional 1.7B by 8.5 to 13.5 points. The
verbatim strict-match alone shrinks that to 1.5 points and would support the opposite write-up — that
looping buys nothing — purely because one model likes dollar signs.

Two conclusions. First, the published 78.92 cannot have come from the verbatim strict-match; both the
`$`-corrected strict figure (79.0) and the flexible-extract gap (+8.5 vs the paper's +8.64) line up with
it closely, so the replication stands. Second, any looped-vs-baseline math comparison should report the
extraction rule and preferably more than one, because here the choice of rule is worth more than the
architecture. `eval/rescore.py` re-scores saved generations all four ways at zero compute cost.
