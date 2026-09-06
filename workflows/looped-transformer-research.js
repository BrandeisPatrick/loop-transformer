export const meta = {
  name: 'looped-transformer-research',
  description: 'Survey open-weight looped/recurrent-depth LLMs, verify availability, check Ollama/llama.cpp/MLX support, extract eval targets, synthesize a local deploy + eval plan for an M4 16GB Mac',
  phases: [
    { title: 'Sweep', detail: 'parallel finders: open-weight models, Qwen-derived variants, runtime support, 2025-2026 literature, eval tables' },
    { title: 'Verify', detail: 'per-model adversarial verification of weights, size, license, inference requirements' },
    { title: 'Deep-dive', detail: 'inference recipe + eval replication targets per viable model' },
    { title: 'Synthesize', detail: 'ranked plan + completeness critic' },
  ],
}

const HW = `TARGET MACHINE: Apple M4 MacBook, 16 GB unified RAM, macOS (Darwin 25), ~3.5 GB free disk right now (user may free more), Python 3.9 system + uv available, no torch/mlx/ollama installed yet. Today is 2026-09-05. The user wants: (1) a looped / recurrent-depth / latent-reasoning transformer LLM deployed LOCALLY, (2) paired with Ollama (either running through Ollama, or exposed via an Ollama-compatible / OpenAI-compatible API alongside Ollama-served baseline models), ideally a looped model built on top of Qwen or another open-source base, (3) then run math etc. evaluations vs its base/comparable model and replicate published results.`

const TOOLS = `You MUST use live web tools: load WebSearch and WebFetch via ToolSearch ("select:WebSearch,WebFetch") and actually fetch Hugging Face pages, arXiv abstracts, GitHub repos/issues. Do not answer from memory alone; cite URLs for every claim. Prefer huggingface.co model pages, config.json, README, and GitHub issues as primary sources. Report what you could NOT confirm explicitly.`

const MODELS_SCHEMA = {
  type: 'object',
  properties: {
    models: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      org: { type: 'string' },
      hf_repo: { type: 'string', description: 'owner/repo on huggingface.co, or empty string if none' },
      params_b: { type: 'number' },
      base_model: { type: 'string', description: 'pretrained base it was derived from, or "from scratch"' },
      loop_mechanism: { type: 'string', description: 'how recurrence works: which blocks loop, loop count, adaptive exit, etc.' },
      release_date: { type: 'string' },
      license: { type: 'string' },
      weights_available: { type: 'boolean' },
      inference_requirements: { type: 'string', description: 'transformers version, trust_remote_code, custom repo, etc.' },
      gguf_available: { type: 'boolean' },
      mlx_available: { type: 'boolean' },
      ollama_status: { type: 'string' },
      paper_url: { type: 'string' },
      code_url: { type: 'string' },
      notes: { type: 'string' },
      confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
      sources: { type: 'array', items: { type: 'string' } },
    }, required: ['name', 'hf_repo', 'weights_available', 'params_b', 'loop_mechanism', 'confidence', 'sources'] } },
    unconfirmed_leads: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['models', 'unconfirmed_leads'],
}

const PAPERS_SCHEMA = {
  type: 'object',
  properties: {
    papers: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, url: { type: 'string' }, date: { type: 'string' }, org: { type: 'string' },
      key_claims: { type: 'string' }, weights_released: { type: 'boolean' }, code_url: { type: 'string' },
      hf_repo: { type: 'string' }, relevance_to_local_deploy: { type: 'string' },
    }, required: ['title', 'url', 'date', 'key_claims', 'weights_released'] } },
    trends_summary: { type: 'string' },
  },
  required: ['papers', 'trends_summary'],
}

const RUNTIME_SCHEMA = {
  type: 'object',
  properties: {
    llama_cpp_support: { type: 'string' },
    ollama_support: { type: 'string' },
    ollama_import_unsupported_arch: { type: 'string', description: 'can Ollama import safetensors/GGUF of an architecture llama.cpp does not know? cite docs' },
    mlx_support: { type: 'string' },
    vllm_sglang_support: { type: 'string' },
    bridge_options: { type: 'array', items: { type: 'object', properties: {
      approach: { type: 'string' }, how: { type: 'string' }, pros: { type: 'string' }, cons: { type: 'string' }, effort: { type: 'string' },
    }, required: ['approach', 'how'] } },
    ollama_api_spec: { type: 'string', description: 'endpoints and request/response shapes a shim must implement to look like Ollama to clients (e.g. /api/tags, /api/chat, /api/generate, /api/show, /v1/chat/completions)' },
    apple_silicon_notes: { type: 'string', description: 'torch MPS vs MLX vs CPU for custom-architecture HF models; bf16 support; memory' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['llama_cpp_support', 'ollama_support', 'bridge_options', 'ollama_api_spec', 'sources'],
}

const EVALS_SCHEMA = {
  type: 'object',
  properties: {
    models: { type: 'array', items: { type: 'object', properties: {
      model: { type: 'string' },
      benchmarks: { type: 'array', items: { type: 'object', properties: {
        benchmark: { type: 'string' }, setting: { type: 'string', description: 'few-shot/CoT/thinking, max tokens, eval harness' },
        results: { type: 'array', items: { type: 'object', properties: {
          variant: { type: 'string', description: 'e.g. Ouro-1.4B loops=4, or baseline Qwen3-1.7B' }, score: { type: 'number' }, note: { type: 'string' },
        }, required: ['variant', 'score'] } },
      }, required: ['benchmark', 'results'] } },
      loop_count_ablation: { type: 'string', description: 'how accuracy changes with recurrence steps, with numbers if reported' },
      eval_code_url: { type: 'string' },
      comparable_baselines: { type: 'array', items: { type: 'string' }, description: 'baseline model names the paper compares to that are available on Ollama' },
      sources: { type: 'array', items: { type: 'string' } },
    }, required: ['model', 'benchmarks', 'sources'] } },
    feasible_replications: { type: 'array', items: { type: 'string' }, description: 'concrete experiments replicable on a 16GB M4 in hours, with expected numbers' },
  },
  required: ['models', 'feasible_replications'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    hf_repo: { type: 'string' },
    exists_and_downloadable: { type: 'boolean' },
    gated: { type: 'boolean' },
    actual_params_b: { type: 'number' },
    weight_size_gb: { type: 'number', description: 'sum of safetensors sizes' },
    dtype: { type: 'string' },
    license: { type: 'string' },
    trust_remote_code: { type: 'boolean' },
    transformers_version_note: { type: 'string' },
    loop_count_configurable_at_inference: { type: 'string', description: 'how (config field / generate kwarg), and reported max' },
    base_model: { type: 'string' },
    apple_silicon_or_cpu_reports: { type: 'string' },
    gguf_or_mlx_conversions: { type: 'string' },
    refuted: { type: 'boolean', description: 'true if the model is NOT actually usable as described (missing weights, gated with no access, broken code)' },
    reason: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['hf_repo', 'exists_and_downloadable', 'refuted', 'reason', 'sources', 'actual_params_b', 'weight_size_gb'],
}

const RECIPE_SCHEMA = {
  type: 'object',
  properties: {
    model: { type: 'string' },
    hf_repo: { type: 'string' },
    pip_deps: { type: 'array', items: { type: 'string' } },
    python_version_note: { type: 'string' },
    load_and_generate_code: { type: 'string', description: 'complete minimal Python snippet for Apple Silicon (MPS, fallback CPU), including how to set loop/recurrence count and chat template' },
    loop_param: { type: 'string' },
    thinking_mode: { type: 'string' },
    expected_ram_gb: { type: 'number' },
    expected_download_gb: { type: 'number' },
    gotchas: { type: 'array', items: { type: 'string' } },
    mlx_or_gguf_path: { type: 'string', description: 'if a community MLX/GGUF port exists, exact repo + how to run; else "none found"' },
    official_eval_recipe: { type: 'string', description: 'how the authors evaluated (harness, prompts, few-shot, sampling) so results can be replicated' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['model', 'hf_repo', 'pip_deps', 'load_and_generate_code', 'loop_param', 'gotchas', 'sources'],
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    recommended_model: { type: 'string' },
    rationale: { type: 'string' },
    alternatives_ranked: { type: 'array', items: { type: 'object', properties: { model: { type: 'string' }, why_not_first: { type: 'string' } }, required: ['model', 'why_not_first'] } },
    qwen_based_option: { type: 'string', description: 'is there a Qwen-derived looped model? if not, what is the closest and could we build one (uptrain) on this hardware?' },
    deployment_approach: { type: 'string' },
    ollama_pairing_approach: { type: 'string' },
    disk_ram_budget: { type: 'string' },
    eval_plan: { type: 'string' },
    replication_targets: { type: 'array', items: { type: 'string' } },
    step_by_step: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    open_questions_for_user: { type: 'array', items: { type: 'string' } },
  },
  required: ['recommended_model', 'rationale', 'alternatives_ranked', 'qwen_based_option', 'deployment_approach', 'ollama_pairing_approach', 'disk_ram_budget', 'eval_plan', 'replication_targets', 'step_by_step', 'risks'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'string' } },
    contradictions: { type: 'array', items: { type: 'string' } },
    unverified_claims: { type: 'array', items: { type: 'string' } },
    suggested_followups: { type: 'array', items: { type: 'string' } },
  },
  required: ['gaps', 'contradictions', 'unverified_claims', 'suggested_followups'],
}

phase('Sweep')
log('Sweeping: open-weight looped LLMs, Qwen-derived variants, runtime support, literature, eval tables')

const FINDERS = [
  { key: 'open-weight', schema: MODELS_SCHEMA, prompt: `${HW}\n\nTASK: Enumerate EVERY looped / recurrent-depth / depth-recurrent / latent-reasoning transformer LANGUAGE MODEL with publicly released weights on Hugging Face as of September 2026. Known starting points you must confirm and expand on: Ouro (ByteDance Seed, "Scaling Latent Reasoning via Looped Language Models", Ouro-1.4B / 2.6B / -Thinking variants), Huginn-0125 (tomg-group-umd, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"), Mixture-of-Recursions (KAIST/Google), Relaxed Recursive Transformers (Google DeepMind), Parallel Loop Transformer, and any 2026 successors (Ouro-2, Huginn-2, looped Qwen/Llama/Gemma variants, etc.). Search HF with queries like "looped", "recurrent depth", "loop transformer", "LoopLM", "latent reasoning", "recursive transformer", "Ouro", "huginn". For each model, fetch its HF page and config.json. Do NOT include tiny task-specific recursive models (HRM/TRM on Sudoku/ARC) as main entries but list them in notes. ${TOOLS}` },
  { key: 'qwen-derived', schema: MODELS_SCHEMA, prompt: `${HW}\n\nTASK: Find looped / recurrent-depth models that were built by CONVERTING or UPTRAINING an existing open pretrained model — especially Qwen (Qwen2.5, Qwen3), but also Llama, Gemma, SmolLM, Pythia, OLMo. Search terms: "looped Qwen", "recursive Qwen", "depth recurrence uptraining", "loop uptraining", "layer looping fine-tune", "Relaxed Recursive Transformers", "recursive transformer LoRA", "converting pretrained transformer to looped", "middle layer recurrence", "block recurrence Qwen3", "Ouro Qwen", "looped language model continual pretraining", 2025-2026 arXiv + GitHub + HF. Also find TRAINING/CONVERSION CODE repos (even if weights not released) and note whether uptraining a ~0.5B-1.7B Qwen into a looped model is feasible on a 16GB M4 (estimate tokens/compute needed from the papers). Be explicit if NO Qwen-based looped model with weights exists — that is a valid and important finding. ${TOOLS}` },
  { key: 'runtime', schema: RUNTIME_SCHEMA, prompt: `${HW}\n\nTASK: Determine the runtime/serving support status for looped / recurrent-depth LLMs (Ouro 1.4B/2.6B by ByteDance, Huginn-0125 by UMD, Mixture-of-Recursions, and any other looped models with weights) across: llama.cpp (GGUF; search github.com/ggml-org/llama.cpp issues and PRs for "Ouro", "looped", "recurrent depth", "huginn"), Ollama (search ollama.com/library and ollama GitHub issues), MLX / mlx-lm (search ml-explore/mlx-lm for Ouro/looped support and community MLX ports on HF), vLLM and SGLang. Then research HOW to make a custom Hugging Face / MLX model appear as an Ollama-compatible endpoint: document Ollama's REST API precisely (GET /api/tags, POST /api/chat, /api/generate, /api/show, /api/version, plus OpenAI-compatible /v1/chat/completions and /v1/models) including streaming NDJSON shapes, from docs at github.com/ollama/ollama/blob/main/docs/api.md. Confirm whether Ollama can import an architecture llama.cpp does not support (docs/import.md). Also research Apple Silicon specifics: torch MPS with trust_remote_code custom models, bf16 on MPS, memory for a 1.4B-3.5B bf16 model with KV cache, and whether MLX would be faster. ${TOOLS}` },
  { key: 'literature', schema: PAPERS_SCHEMA, prompt: `${HW}\n\nTASK: Survey looped-transformer / recurrent-depth / latent-reasoning research from January 2025 through September 2026, focusing on the RECENT wave (2026). Search arXiv, Hugging Face papers, Google Scholar, Twitter/X summaries, for: "looped transformer", "looped language model", "LoopLM", "recurrent depth", "depth recurrence", "latent reasoning loop", "recursive transformer", "mixture of recursions", "Ouro", "Huginn", "parallel loop transformer", "loop residual", "test-time depth scaling", "adaptive computation depth LLM", "universal transformer 2026". For each paper record title, arXiv URL, date, org, key claims (with numbers), and whether weights/code were released with links. Aim for 15-30 papers, prioritize those with released weights or code. End with a trends summary: what is the field converging on (which blocks loop, loop counts, adaptive exit, KV sharing, training recipes) and which papers a practitioner should replicate first. ${TOOLS}` },
  { key: 'evals', schema: EVALS_SCHEMA, prompt: `${HW}\n\nTASK: Extract the published EVALUATION results for open-weight looped LLMs so they can be replicated locally. For Ouro (ByteDance, arXiv 2510.25741 or search "Scaling Latent Reasoning via Looped Language Models"; models Ouro-1.4B, Ouro-2.6B, Ouro-1.4B-Thinking, Ouro-2.6B-Thinking), Huginn-0125 (arXiv 2502.05171), and any other looped model with weights (Mixture-of-Recursions etc.): fetch the arXiv HTML (arxiv.org/html/<id>) or PDF and the HF model card, and extract the benchmark tables — GSM8K, MATH500, MATH, AIME, MMLU, MMLU-Pro, BBH, HumanEval, ARC, HellaSwag, etc. — with the exact numbers for the looped model and for each baseline it is compared against (Qwen3-1.7B, Qwen3-4B, Qwen2.5-1.5B, Llama-3.2-1B/3B, Gemma-3, SmolLM, etc.). Capture the loop-count / recurrence-step ablations (accuracy vs number of loops, e.g. Ouro at 1,2,3,4 loops; Huginn at r=1..64) with numbers. Record eval settings (few-shot count, CoT, thinking mode, max new tokens, temperature, harness like lm-evaluation-harness / OpenCompass / evalscope) and links to released eval code/scripts. Then list concrete replications feasible on a 16GB M4 within a few hours (e.g. GSM8K 200-500 problem subset at loops 1..4 vs Qwen3-1.7B via Ollama) with the expected numbers from the paper. ${TOOLS}` },
]

const sweep = await parallel(FINDERS.map(f => () =>
  agent(f.prompt, { label: `find:${f.key}`, phase: 'Sweep', schema: f.schema })
    .then(r => ({ key: f.key, result: r }))
))
const byKey = Object.fromEntries(sweep.filter(Boolean).map(s => [s.key, s.result]))
const missing = FINDERS.map(f => f.key).filter(k => !byKey[k])
if (missing.length) log(`WARNING: finders returned nothing: ${missing.join(', ')}`)

// dedup model candidates across the two model finders + literature hf_repos (barrier justified: cross-finder dedup)
const REPO_RE = /([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/
function cleanRepo(x) {
  const m = String(x || '').replace(/^https?:\/\/huggingface\.co\//, '').match(REPO_RE)
  if (!m) return ''
  const r = m[1]
  if (/^(collections|datasets|none|n\/a|github\.com|huggingface\.co)$/i.test(r.split('/')[0])) return ''
  return r
}
const candidates = new Map()
for (const k of ['open-weight', 'qwen-derived']) {
  for (const m of (byKey[k]?.models || [])) {
    const repo = cleanRepo(m.hf_repo)
    if (!repo || !m.weights_available) continue
    const id = repo.toLowerCase()
    if (!candidates.has(id)) candidates.set(id, { ...m, hf_repo: repo, found_by: [k] })
    else candidates.get(id).found_by.push(k)
  }
}
for (const p of (byKey['literature']?.papers || [])) {
  const repo = cleanRepo(p.hf_repo)
  if (!repo || !p.weights_released) continue
  const id = repo.toLowerCase()
  if (!candidates.has(id)) candidates.set(id, { name: p.title, hf_repo: repo, weights_available: true, params_b: 0, loop_mechanism: p.key_claims, confidence: 'low', sources: [p.url], found_by: ['literature'] })
}
const MAX_B = 4.5
const allCands = [...candidates.values()]
const candList = allCands.filter(c => !(Number(c.params_b) > MAX_B) && !/IQuest|LoopCoder-V2|40B/i.test(c.hf_repo + ' ' + (c.name || '')))
log(`Dropped ${allCands.length - candList.length} candidates as >${MAX_B}B or known-oversized: ${allCands.filter(c => !candList.includes(c)).map(c => c.hf_repo).join(', ')}`)
log(`Sweep done: ${candList.length} unique weight-bearing candidates to verify`)

phase('Verify')
const verified = await pipeline(candList,
  (c) => agent(`${HW}\n\nADVERSARIAL VERIFICATION. A researcher claims this looped/recurrent-depth model is usable locally:\n${JSON.stringify(c, null, 2)}\n\nTry to REFUTE it. Fetch https://huggingface.co/${c.hf_repo} , https://huggingface.co/${c.hf_repo}/tree/main , https://huggingface.co/${c.hf_repo}/raw/main/config.json , and README/model card. Confirm: repo exists and weights are downloadable (list safetensors files and sizes, total GB); gated or not; license; actual parameter count from config; whether trust_remote_code / a custom package is needed and the minimum transformers version; HOW the loop/recurrence count is set at inference (config field name, generate kwarg, or fixed) and the max reported; the base model it derives from; any GitHub/HF discussion reports of running on Apple Silicon (MPS) or CPU, and any known bugs; any community GGUF/MLX conversions (search HF for "<name> GGUF", "<name> MLX", "<name> mlx-community"). Set refuted=true if weights are missing, gated without access path, or the code is known broken. If uncertain about usability, say so in reason but only set refuted=true when there is evidence. ${TOOLS}`,
    { label: `verify:${c.hf_repo}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'medium' })
    .then(v => ({ candidate: c, verdict: v })),
  (r, c) => {
    if (!r || !r.verdict) return null
    const v = r.verdict
    const viable = !v.refuted && v.exists_and_downloadable && (v.actual_params_b || c.params_b || 0) <= 4.5
    return { ...r, viable }
  },
  (r, c) => {
    if (!r) return null
    const PRIORITY = new Set(['mshapiro123/recurrent-qwen2.5-0.5b-full-block'])
    const sizeB = Number(r.verdict?.actual_params_b) || Number(c.params_b) || 0
    if (!r.viable || !(sizeB >= 1.0 || PRIORITY.has(c.hf_repo.toLowerCase()))) return { ...r, recipe: null }
    return agent(`${HW}\n\nTASK: Write an exact, runnable local inference recipe for this verified looped model on Apple Silicon:\n${JSON.stringify(r.verdict, null, 2)}\nOriginal notes: ${JSON.stringify(c, null, 2)}\n\nFetch the HF model card, the modeling_*.py custom code (https://huggingface.co/${c.hf_repo}/tree/main and raw files), and the official GitHub repo. Produce: pip deps with versions; Python version note (system has 3.9 but uv can install 3.11/3.12); a complete minimal snippet (AutoTokenizer/AutoModelForCausalLM with trust_remote_code, torch_dtype bf16 or fp16, device mps with cpu fallback, chat template, generate) INCLUDING how to set the loop/recurrence count per call; thinking-mode toggling if the model has one; expected RAM and download size; gotchas (MPS unsupported ops, bf16 on MPS, attention impl, cache classes, missing tokenizer files); whether an MLX or GGUF port exists with exact repo and run command; and the OFFICIAL eval recipe the authors used (harness, prompt format, few-shot, sampling, max tokens) with links so results can be replicated. ${TOOLS}`,
      { label: `recipe:${c.hf_repo}`, phase: 'Deep-dive', schema: RECIPE_SCHEMA, effort: 'high' })
      .then(recipe => ({ ...r, recipe }))
  },
)
const verifiedList = verified.filter(Boolean)
const viableList = verifiedList.filter(v => v.viable)
log(`Verified ${verifiedList.length}; ${viableList.length} viable (weights present, <=4.5B); ${verifiedList.length - viableList.length} refuted or too large`)

phase('Synthesize')
const LOCAL_STATE = `LOCAL STATE ALREADY ESTABLISHED (verified hands-on on the target Mac, 2026-09-06): disk now has ~21 GB free (cleanup happened). Installed: Ollama 0.33.3 via Homebrew (its bundled llama-server binary contains the 'nanbeige' architecture strings; the Go binary does not), llama.cpp 0.4.0 via Homebrew (libllama.dylib contains 'nanbeige'). Project dir vibe/looplm with uv venv (Python 3.12, torch 2.14 with MPS, transformers 5.16.1, mlx 0.32.2, mlx-lm, datasets, fastapi). Written and smoke-tested: serve/shim.py = Ollama-API + OpenAI-API compatible FastAPI server over HF transformers on MPS with per-request num_loops (backends: ouro via config.total_ut_steps, recurrent-qwen via generate(max_loops), generic hf); eval/run_eval.py = GSM8K / MATH-500 runner against any OpenAI-compatible endpoint with resumable JSONL + summary. Verified live: mshapiro123/recurrent-qwen2.5-0.5b-r16-adapter (+ Qwen/Qwen2.5-0.5B-Instruct base) loads and generates on MPS through the shim; loops=1 reproduces the base answer, loops=3 degrades off-task as its card warns. Verified: gkraker04/Nanbeige4.2-3B-GGUF Q4_K_M header has general.architecture=nanbeige, nanbeige.num_loops=2, nanbeige.block_count=22. Verified: ollama.com zoecohn4/Ouro is an unrelated 8B llama Q4_0 'snarky sidekick', NOT ByteDance Ouro. Verified HF sizes: Ouro-1.4B 2.87 GB bf16, Ouro-2.6B 5.34 GB, Nanbeige4.2-3B 8.34 GB bf16, huginn-0125 15.65 GB fp32, all ungated Apache-2.0. A second venv with transformers<5 is being created for Ouro.`
const dossier = {
  hardware: HW,
  runtime: byKey['runtime'] || null,
  literature: byKey['literature'] || null,
  evals: byKey['evals'] || null,
  unconfirmed_leads: [...(byKey['open-weight']?.unconfirmed_leads || []), ...(byKey['qwen-derived']?.unconfirmed_leads || [])],
  qwen_finder_notes: byKey['qwen-derived']?.notes || '',
  verified_models: verifiedList.map(v => ({ candidate: v.candidate, verdict: v.verdict, viable: v.viable, recipe: v.recipe })),
}

const plan = await agent(`${HW}\n\nYou are the synthesis lead. Below is a research dossier assembled by parallel agents (all claims carry source URLs). Produce a decisive, ranked plan for: (1) which looped model to deploy locally first and why (consider RAM 16GB, disk ~3.5GB free now, Apple Silicon runtime support, license, quality), (2) whether a Qwen-derived looped model exists and, if not, what the closest option is and whether uptraining one ourselves on this Mac is realistic (with a compute estimate), (3) the deployment approach (transformers on MPS vs MLX vs GGUF) and how to pair with Ollama (native Ollama support if any; otherwise an Ollama-API-compatible shim server so clients see it as a model, plus Ollama serving the baseline Qwen for comparison), (4) an evaluation plan: benchmarks, subset sizes feasible in hours on this machine, loop-count sweep, baselines pulled via Ollama, and the exact published numbers we expect to replicate, (5) a step-by-step build order, disk/RAM budget, risks, and open questions for the user. Be concrete: name repos, params, commands. Flag any contradictions between agents.\n\n${LOCAL_STATE}\n\nDOSSIER:\n${JSON.stringify(dossier, null, 2)}`,
  { label: 'synthesize:plan', phase: 'Synthesize', schema: PLAN_SCHEMA, effort: 'high' })

const critic = await agent(`${HW}\n\nCOMPLETENESS CRITIC. Review this research dossier and the resulting plan for a looped-transformer local deployment + eval project. What is missing? Specifically check: (a) any well-known looped/recurrent-depth model with weights that the sweep missed (do a fresh WebSearch for "looped language model weights 2026", "recurrent depth LLM huggingface", "Ouro 2", "huginn 2026", "loop transformer Qwen3"), (b) claims in the plan not backed by a verified source, (c) contradictions between agents, (d) practical blockers for Apple Silicon 16GB / 3.5GB disk that were glossed over, (e) whether the eval plan actually matches the papers' settings so replication is apples-to-apples. Return gaps, contradictions, unverified claims, and concrete follow-up tasks. ${TOOLS}\n\n${LOCAL_STATE}\n\nPLAN:\n${JSON.stringify(plan, null, 2)}\n\nDOSSIER (abridged):\n${JSON.stringify({ verified: dossier.verified_models.map(v => ({ repo: v.candidate.hf_repo, viable: v.viable, verdict_reason: v.verdict?.reason, size_gb: v.verdict?.weight_size_gb, loop: v.verdict?.loop_count_configurable_at_inference })), runtime: dossier.runtime, evals: dossier.evals, unconfirmed: dossier.unconfirmed_leads }, null, 2)}`,
  { label: 'critic:completeness', phase: 'Synthesize', schema: CRITIC_SCHEMA, effort: 'high' })

return { plan, critic, dossier }