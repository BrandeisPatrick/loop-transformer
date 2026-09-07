#!/usr/bin/env python
"""Model-agnostic math eval runner.

Talks to any OpenAI-compatible chat endpoint (Ollama's /v1, or the looplm shim),
so the looped model and the Ollama baselines are evaluated by the exact same code.

Examples:
  # baseline via Ollama
  python eval/run_eval.py --base-url http://127.0.0.1:11434/v1 --model qwen3:1.7b \
      --benchmark gsm8k --n 200 --out results/gsm8k_qwen3-1.7b.jsonl

  # looped model via shim, passing loop count through extra_body
  python eval/run_eval.py --base-url http://127.0.0.1:11435/v1 --model ouro-1.4b \
      --benchmark gsm8k --n 200 --extra-body '{"num_loops": 4}' \
      --out results/gsm8k_ouro-1.4b_L4.jsonl
"""
import argparse, json, os, re, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# How long Ollama keeps a model resident after a request. "0" unloads immediately, so between
# requests nothing is loaded; each item then pays a reload (~1-3 s for a Q4 model from page cache).
# Chosen deliberately: memory headroom over speed.
OLLAMA_KEEP_ALIVE = os.environ.get("LOOPLM_KEEP_ALIVE", "0")
COT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# The standard 8 chain-of-thought exemplars used by lm-evaluation-harness `gsm8k_cot` (Wei et al. 2022).
GSM8K_COT_EXEMPLARS = [
 ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
  "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
 ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
  "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
 ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
  "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
 ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
  "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
 ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
  "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
 ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
  "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
 ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
  "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
 ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
  "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]

def fewshot_prompt(question, shots):
    parts = [f"Q: {q}\nA: {a}" for q, a in GSM8K_COT_EXEMPLARS[:shots]]
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)

STRICT_RE = re.compile(r"The answer is\s*\$?\s*(-?[\d,]*\.?\d+)")

def extract_strict(s):
    m = STRICT_RE.findall(s)
    return m[0].replace(",", "") if m else None   # lm-eval take_first: the model's first stated answer

def ollama_raw(base_url, model, prompt, max_tokens, temperature, extra_body, timeout, stop):
    """Ollama-native /api/generate with raw=true: no chat template applied (faithful few-shot for base models)."""
    root = base_url.rstrip('/')
    root = root[:-3] if root.endswith('/v1') else root
    opts = {"num_predict": max_tokens, "temperature": temperature, "stop": stop,
            "num_ctx": min(4096, max(2048, 1024 + max_tokens))}   # cap KV cache; see memguard notes
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "options": opts,
            "keep_alive": OLLAMA_KEEP_ALIVE}
    for k, v in (extra_body or {}).items():
        (opts if k in ("num_loops", "num_ctx", "top_p", "top_k", "seed") else body)[k] = v
    t0 = time.time()
    r = requests.post(f"{root}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return {"text": j.get("response") or "", "reasoning": "", "latency_s": time.time() - t0,
            "usage": {"prompt_tokens": j.get("prompt_eval_count"), "completion_tokens": j.get("eval_count")},
            "finish_reason": "length" if j.get("done_reason") == "length" else "stop"}

def ollama_chat(base_url, model, messages, max_tokens, temperature, extra_body, timeout, think):
    """Ollama-native /api/chat (lets us toggle thinking explicitly and read the thinking field)."""
    root = base_url.rstrip('/')
    root = root[:-3] if root.endswith('/v1') else root
    opts = {"num_predict": max_tokens, "temperature": temperature,
            "num_ctx": min(4096, max(2048, 1024 + max_tokens))}   # cap KV cache; see memguard notes
    body = {"model": model, "messages": messages, "stream": False, "options": opts,
            "keep_alive": OLLAMA_KEEP_ALIVE}
    if think is not None: body["think"] = think
    for k, v in (extra_body or {}).items():
        (opts if k in ("num_loops", "num_ctx", "top_p", "top_k", "seed") else body)[k] = v
    t0 = time.time()
    r = requests.post(f"{root}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json(); m = j.get("message", {})
    return {"text": m.get("content") or "", "reasoning": m.get("thinking") or "", "latency_s": time.time() - t0,
            "usage": {"prompt_tokens": j.get("prompt_eval_count"), "completion_tokens": j.get("eval_count")},
            "finish_reason": "length" if j.get("done_reason") == "length" else "stop"}

def completion(base_url, model, prompt, max_tokens, temperature, extra_body, timeout, stop):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, "stop": stop}
    body.update(extra_body or {})
    t0 = time.time()
    r = requests.post(f"{base_url.rstrip('/')}/completions", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return {"text": j["choices"][0].get("text") or "", "reasoning": "", "usage": j.get("usage", {}),
            "latency_s": time.time() - t0, "finish_reason": j["choices"][0].get("finish_reason")}

# ---------------------------------------------------------------- datasets
def load_benchmark(name, n, seed):
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": f"gsm8k-{i}", "question": r["question"],
                  "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                 for i, r in enumerate(ds)]
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"id": f"math500-{i}", "question": r["problem"], "answer": r["answer"],
                  "level": r.get("level"), "subject": r.get("subject")}
                 for i, r in enumerate(ds)]
    else:
        raise SystemExit(f"unknown benchmark {name}")
    if n and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items

# ---------------------------------------------------------------- answer extraction
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{": depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j+1:k]
        k += 1
    return None

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def extract_number(s):
    b = last_boxed(s)
    if b is not None:
        m = NUM_RE.findall(b.replace("$", ""))
        if m:
            return m[-1].replace(",", "")
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", s)
    if m:
        return m[-1].replace(",", "")
    m = NUM_RE.findall(s)
    return m[-1].replace(",", "") if m else None

def norm_math(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.rstrip(".")
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1]
    # a/b -> \frac{a}{b}
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        s = f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    return s

def numeric_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False

def grade(benchmark, response, gold):
    if benchmark == "gsm8k":
        pred = extract_number(response)
        return pred, (pred is not None and numeric_equal(pred, gold))
    pred = last_boxed(response)
    if pred is None:
        return None, False
    p, g = norm_math(pred), norm_math(gold)
    if p == g:
        return pred, True
    pn = NUM_RE.findall(p) if p else []
    gn = NUM_RE.findall(g) if g else []
    if len(pn) == 1 and len(gn) == 1 and p.replace(pn[0], "") == g.replace(gn[0], ""):
        return pred, numeric_equal(pn[0].replace(",", ""), gn[0].replace(",", ""))
    return pred, False

# ---------------------------------------------------------------- client
def chat(base_url, model, messages, max_tokens, temperature, extra_body, timeout):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    body.update(extra_body or {})
    t0 = time.time()
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = j.get("usage", {})
    return {"text": text, "reasoning": reasoning, "usage": usage, "latency_s": time.time() - t0,
            "finish_reason": j["choices"][0].get("finish_reason")}

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", choices=["gsm8k", "math500"], default="gsm8k")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--no-cot-suffix", action="store_true")
    ap.add_argument("--mode", choices=["chat", "completion"], default="chat",
                    help="chat = chat template + boxed CoT (instruct/thinking models); completion = raw few-shot 'Q:/A:' prompt via /v1/completions with strict 'The answer is N' match (base models, lm-eval gsm8k_cot protocol)")
    ap.add_argument("--shots", type=int, default=3, help="few-shot exemplars in completion mode (Ouro paper: 3)")
    ap.add_argument("--api", choices=["openai", "ollama"], default="openai",
                    help="openai = /v1 endpoints (shim or Ollama); ollama = native /api/generate raw=true (completion) or /api/chat (chat)")
    ap.add_argument("--think", choices=["on", "off"], default=None, help="ollama api chat: force thinking on/off")
    ap.add_argument("--flexible", action="store_true", help="completion mode: also accept last number in output (lm-eval flexible-extract)")
    ap.add_argument("--extra-body", default="{}", help="JSON merged into request body (e.g. num_loops)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = json.loads(args.extra_body)
    items = load_benchmark(args.benchmark, args.n, args.seed)
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                j = json.loads(line); done[j["id"]] = j
            except Exception:
                pass
        errored = [k for k, v in done.items() if v.get("error")]
        for k in errored:
            del done[k]          # retry on the next invocation instead of scoring it wrong forever
        if errored:
            print(f"retrying {len(errored)} item(s) that errored previously", file=sys.stderr)
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.benchmark}: {len(items)} items, {len(done)} done, {len(todo)} to run -> {args.out}", file=sys.stderr)

    lock = threading.Lock()
    fout = open(args.out, "a")

    def call_with_retry(fn, *a, **kw):
        """Transient 5xx / connection drops otherwise get recorded as a wrong answer, which silently
        understates a model. Retry a few times with backoff before giving up."""
        last = None
        for attempt in range(3):
            try:
                return fn(*a, **kw)
            except Exception as e:
                last = e
                transient = isinstance(e, (requests.ConnectionError, requests.Timeout)) or \
                    (isinstance(e, requests.HTTPError) and getattr(e.response, "status_code", 0) >= 500)
                if not transient or attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise last

    def work(it):
        try:
            if args.mode == "completion":
                if args.benchmark != "gsm8k":
                    raise SystemExit("completion mode currently supports gsm8k only")
                prompt = fewshot_prompt(it["question"], args.shots)
                fn = ollama_raw if args.api == "ollama" else completion
                res = call_with_retry(fn, args.base_url, args.model, prompt, args.max_tokens,
                                      args.temperature, extra, args.timeout,
                                      stop=["Q:", "</s>", "<|im_end|>"])
                pred = extract_strict(res["text"])
                if pred is None and args.flexible:
                    pred = extract_number(res["text"])
                ok = pred is not None and numeric_equal(pred, it["answer"])
            else:
                q = it["question"] + ("" if args.no_cot_suffix else COT_SUFFIX)
                messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
                           [{"role": "user", "content": q}]
                if args.api == "ollama":
                    res = call_with_retry(ollama_chat, args.base_url, args.model, messages, args.max_tokens,
                                          args.temperature, extra, args.timeout,
                                          None if args.think is None else args.think == "on")
                else:
                    res = call_with_retry(chat, args.base_url, args.model, messages, args.max_tokens,
                                          args.temperature, extra, args.timeout)
                pred, ok = grade(args.benchmark, res["text"], it["answer"])
            rec = {**it, "model": args.model, "extra": extra, "pred": pred, "correct": ok, **res}
        except Exception as e:
            rec = {**it, "model": args.model, "extra": extra, "pred": None, "correct": False, "error": repr(e)}
        with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            done[it["id"]] = rec
            n = len(done); acc = sum(1 for r in done.values() if r.get("correct")) / n
            print(f"[{n}/{len(items)}] acc={acc:.3f} last={'OK' if rec['correct'] else 'X'} "
                  f"pred={rec['pred']!r} gold={it['answer']!r} {rec.get('latency_s', 0):.1f}s", file=sys.stderr)
        return rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(as_completed([ex.submit(work, it) for it in todo]))
    fout.close()

    recs = list(done.values())
    n = len(recs); acc = sum(1 for r in recs if r.get("correct")) / max(n, 1)
    toks = [r.get("usage", {}).get("completion_tokens") for r in recs if r.get("usage")]
    toks = [t for t in toks if isinstance(t, int)]
    summary = {"benchmark": args.benchmark, "model": args.model, "extra": extra, "mode": args.mode, "api": args.api, "think": args.think,
               "shots": args.shots if args.mode == "completion" else None, "n": n, "accuracy": acc,
               "errors": sum(1 for r in recs if r.get("error")),
               "truncated": sum(1 for r in recs if r.get("finish_reason") == "length"),
               "mean_completion_tokens": (sum(toks) / len(toks)) if toks else None,
               "mean_latency_s": sum(r.get("latency_s", 0) for r in recs) / max(n, 1),
               "max_tokens": args.max_tokens, "temperature": args.temperature, "seed": args.seed}
    with open(args.out.replace(".jsonl", "") + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
