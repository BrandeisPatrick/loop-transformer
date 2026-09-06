#!/usr/bin/env python
"""looplm shim: serve a Hugging Face looped/recurrent-depth model behind an Ollama-compatible
AND OpenAI-compatible HTTP API, so Ollama clients / the eval runner can talk to it exactly like
they talk to Ollama. Loop count is settable per request.

Endpoints (Ollama):  GET /api/tags  GET /api/version  GET /api/ps  POST /api/show
                     POST /api/chat  POST /api/generate
Endpoints (OpenAI):  GET /v1/models  POST /v1/chat/completions  POST /v1/completions

Per-request loop control (any of these, all optional):
  - Ollama:  "options": {"num_loops": 3, "exit_threshold": 0.9}
  - OpenAI:  top-level "num_loops": 3   (or "extra_body" merged by clients)
  - default: --loops on the command line / model config

Backends (--backend):
  ouro           ByteDance/Ouro-*      loops via config.total_ut_steps (+ model.model.total_ut_steps), exit_threshold kwarg
  recurrent-qwen mshapiro123/recurrent-qwen2.5-0.5b-*   loops via generate(max_loops=N)
  hf             any HF causal LM      loops via --loop-config-field NAME (set on model.config) if given

Run:
  python serve/shim.py --model ByteDance/Ouro-1.4B --backend ouro --name ouro-1.4b --port 11435
"""
import argparse, json, threading, time, uuid, re, sys, os
from typing import Optional, List
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TextIteratorStreamer

# ----------------------------------------------------------------------------- backend
class HFBackend:
    def __init__(self, model_id, backend, dtype, device, default_loops, loop_config_field=None, revision=None, tokenizer_id=None, code_path=None):
        self.model_id, self.backend = model_id, backend
        self.default_loops = default_loops
        self.loop_config_field = loop_config_field
        self.device = device
        self.lock = threading.Lock()
        t0 = time.time()
        if backend == "lds":
            # LoopUS (Looped Depth Up-Scaling): custom LDSForCausalLM from the LoopUS repo, not in HF auto classes.
            sys.path.insert(0, code_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "LoopUS"))
            from models.modeling_lds import LDSForCausalLM
            # LoopUS's loader builds the backbone at the default dtype on CPU, loads a 4 GB state dict,
            # THEN casts and moves. At fp32 that peaks near 12 GB on a 16 GB machine, so construct
            # directly in the target dtype instead (~4 GB peak). Restore the global default afterwards.
            _prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                self.model = LDSForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=str(device))
            finally:
                torch.set_default_dtype(_prev_dtype)
            self.tok = getattr(self.model, "tokenizer", None) or AutoTokenizer.from_pretrained(tokenizer_id or model_id)
            if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
            self.model.eval()
            if default_loops: self.model.N = int(default_loops)
            self.dtype = dtype; self.eos_ids = [self.tok.eos_token_id]
            self.load_s = time.time() - t0
            self.param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[shim] loaded LoopUS {model_id} ({self.param_count/1e9:.2f}B params, N={self.model.N}, q_threshold={self.model.q_threshold}, {dtype}, {device}) in {self.load_s:.1f}s", file=sys.stderr)
            return
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        tok_src = tokenizer_id or model_id
        try:
            self.tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, revision=revision if tok_src == model_id else None)
        except Exception as e:
            base = getattr(cfg, "base_model_name_or_path", None)
            if not base or tok_src == base:
                raise
            print(f"[shim] tokenizer not in {tok_src} ({type(e).__name__}); using base model tokenizer {base}", file=sys.stderr)
            self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if backend == "ouro" and default_loops:
            cfg.total_ut_steps = int(default_loops)
        if backend == "hf" and loop_config_field and default_loops:
            setattr(cfg, loop_config_field, int(default_loops))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, trust_remote_code=True, torch_dtype=dtype, revision=revision,
            low_cpu_mem_usage=True)
        self.model.to(device).eval()
        self.dtype = dtype
        # stop on every plausible end-of-turn token (custom wrappers often ship a generation_config without <|im_end|>)
        eos = set()
        for cand in [self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)]:
            if isinstance(cand, int): eos.add(cand)
            elif isinstance(cand, (list, tuple)): eos.update(int(c) for c in cand)
        for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "</s>"]:
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != getattr(self.tok, "unk_token_id", -1): eos.add(i)
        self.eos_ids = sorted(eos)
        self.load_s = time.time() - t0
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[shim] loaded {model_id} ({self.param_count/1e9:.2f}B params, {dtype}, {device}) in {self.load_s:.1f}s", file=sys.stderr)

    # --- loop control -------------------------------------------------------
    def _apply_loops(self, loops: Optional[int], gen_kwargs: dict):
        if loops is None:
            loops = self.default_loops
        if loops is None:
            return None
        loops = int(loops)
        if self.backend == "ouro":
            self.model.config.total_ut_steps = loops
            inner = getattr(self.model, "model", None)
            if inner is not None and hasattr(inner, "total_ut_steps"):
                inner.total_ut_steps = loops
        elif self.backend == "recurrent-qwen":
            gen_kwargs["max_loops"] = loops
        elif self.backend == "lds":
            self.model.N = loops  # max recursions; q-head halting (q_threshold) may stop earlier
        elif self.backend == "hf" and self.loop_config_field:
            setattr(self.model.config, self.loop_config_field, loops)
        return loops

    def build_prompt(self, messages: List[dict], think: Optional[bool]):
        kw = {}
        if think is not None:
            kw["enable_thinking"] = bool(think)
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # no chat template: naive fallback
            return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"

    def stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
                        loops: Optional[int], exit_threshold: Optional[float], stop: Optional[List[str]] = None):
        """Yields text chunks; final yield is a dict with usage."""
        with self.lock:
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
            n_prompt = inputs["input_ids"].shape[1]
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer,
                              do_sample=temperature > 0, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                              eos_token_id=self.eos_ids)
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            used_loops = self._apply_loops(loops, gen_kwargs)
            if exit_threshold is not None and self.backend == "ouro":
                gen_kwargs["exit_threshold"] = float(exit_threshold)
            if exit_threshold is not None and self.backend == "lds":
                self.model.q_threshold = float(exit_threshold)  # 1.0 = never halt early (fixed depth N)
            err = {}
            def run():
                try:
                    with torch.inference_mode():
                        if self.backend == "lds":
                            gk = {k: v for k, v in gen_kwargs.items() if k not in ("streamer",)}
                            gk["max_context"] = max(4096, int(inputs["input_ids"].shape[1]) + max_tokens)
                            out = self.model.generate(**gk)
                            new = out[0, inputs["input_ids"].shape[1]:]
                            streamer.put(new.unsqueeze(0).cpu()); streamer.end(); return
                        self.model.generate(**gen_kwargs)
                except Exception as e:  # surface errors to the stream
                    err["e"] = repr(e)
                    streamer.end()
            t = threading.Thread(target=run, daemon=True); t.start()
            t0 = time.time(); text = ""
            for chunk in streamer:
                if not chunk:
                    continue
                text += chunk
                yield chunk
                if stop and any(s in text for s in stop):
                    break
            t.join()
            n_out = len(self.tok(text, add_special_tokens=False)["input_ids"])
            yield {"prompt_tokens": n_prompt, "completion_tokens": n_out, "elapsed_s": time.time() - t0,
                   "loops": used_loops, "error": err.get("e")}

THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.S)
def split_think(text: str):
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():]
    if "<think>" in text and "</think>" not in text:  # unfinished thinking
        return text.split("<think>", 1)[1], ""
    return "", text

# ----------------------------------------------------------------------------- app
def make_app(be: HFBackend, name: str):
    app = FastAPI(title="looplm shim")
    created = int(time.time())
    details = {"parent_model": "", "format": "safetensors", "family": be.backend, "families": [be.backend],
               "parameter_size": f"{be.param_count/1e9:.1f}B", "quantization_level": str(be.dtype).replace("torch.", "")}

    def now(): return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    @app.get("/")
    def root(): return JSONResponse("Ollama is running", media_type="text/plain")
    @app.get("/api/version")
    def version(): return {"version": "0.33.3-looplm-shim"}
    @app.get("/api/tags")
    def tags():
        return {"models": [{"name": f"{name}:latest", "model": f"{name}:latest", "modified_at": now(),
                            "size": int(be.param_count * 2), "digest": "looplm", "details": details}]}
    @app.get("/api/ps")
    def ps(): return {"models": [{"name": f"{name}:latest", "model": f"{name}:latest", "size": int(be.param_count*2),
                                  "details": details, "expires_at": now(), "size_vram": 0}]}
    @app.post("/api/show")
    async def show(req: Request):
        return {"modelfile": f"# looplm shim for {be.model_id}", "parameters": f"num_loops {be.default_loops}",
                "template": "{{ .Prompt }}", "details": details, "capabilities": ["completion", "thinking"],
                "model_info": {"general.architecture": be.backend, "looplm.hf_repo": be.model_id,
                               "looplm.default_loops": be.default_loops}}
    @app.get("/v1/models")
    def v1_models(): return {"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "looplm"}]}

    def _opts(body):
        o = body.get("options") or {}
        loops = body.get("num_loops", o.get("num_loops"))
        exit_t = body.get("exit_threshold", o.get("exit_threshold"))
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or o.get("num_predict") or 1024
        temperature = body.get("temperature", o.get("temperature", 0.0)) or 0.0
        top_p = body.get("top_p", o.get("top_p", 0.95))
        stop = body.get("stop") or o.get("stop")
        if isinstance(stop, str): stop = [stop]
        think = body.get("think")
        if isinstance(think, str): think = think != "false"
        return loops, exit_t, int(max_tokens), float(temperature), float(top_p), stop, think

    # ---------------- Ollama /api/chat and /api/generate
    async def ollama_endpoint(req: Request, kind: str):
        body = await req.json()
        loops, exit_t, max_tokens, temperature, top_p, stop, think = _opts(body)
        if kind == "chat":
            prompt = be.build_prompt(body.get("messages", []), think)
        else:
            prompt = body.get("prompt", "")
            if not body.get("raw"):
                prompt = be.build_prompt(([{"role": "system", "content": body["system"]}] if body.get("system") else []) +
                                         [{"role": "user", "content": prompt}], think)
        stream = body.get("stream", True)
        model = body.get("model", name)

        def gen():
            full = ""; usage = None
            for item in be.stream_generate(prompt, max_tokens, temperature, top_p, loops, exit_t, stop):
                if isinstance(item, dict):
                    usage = item; break
                full += item
                if stream:
                    payload = {"model": model, "created_at": now(), "done": False}
                    if kind == "chat": payload["message"] = {"role": "assistant", "content": item}
                    else: payload["response"] = item
                    yield json.dumps(payload) + "\n"
            thinking, content = split_think(full)
            final = {"model": model, "created_at": now(), "done": True, "done_reason": "stop",
                     "total_duration": int(usage["elapsed_s"] * 1e9), "prompt_eval_count": usage["prompt_tokens"],
                     "eval_count": usage["completion_tokens"], "eval_duration": int(usage["elapsed_s"] * 1e9),
                     "looplm": {"loops": usage["loops"], "error": usage["error"]}}
            if kind == "chat":
                final["message"] = {"role": "assistant", "content": "" if stream else content}
                if thinking and not stream: final["message"]["thinking"] = thinking
            else:
                final["response"] = "" if stream else content
            yield json.dumps(final) + "\n"
        if stream:
            return StreamingResponse(gen(), media_type="application/x-ndjson")
        last = None
        for line in gen(): last = line
        return JSONResponse(json.loads(last))

    @app.post("/api/chat")
    async def api_chat(req: Request): return await ollama_endpoint(req, "chat")
    @app.post("/api/generate")
    async def api_generate(req: Request): return await ollama_endpoint(req, "generate")

    # ---------------- OpenAI
    async def openai_endpoint(req: Request, kind: str):
        body = await req.json()
        loops, exit_t, max_tokens, temperature, top_p, stop, think = _opts(body)
        if kind == "chat":
            prompt = be.build_prompt(body.get("messages", []), think)
        else:
            prompt = body.get("prompt", "")
        stream = body.get("stream", False)
        model = body.get("model", name)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def gen():
            full = ""; usage = None
            for item in be.stream_generate(prompt, max_tokens, temperature, top_p, loops, exit_t, stop):
                if isinstance(item, dict):
                    usage = item; break
                full += item
                if stream:
                    chunk = {"id": rid, "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None} if kind == "chat"
                                         else {"index": 0, "text": item, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
            thinking, content = split_think(full)
            fin = "length" if usage["completion_tokens"] >= max_tokens else "stop"
            u = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                 "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                 "looplm": {"loops": usage["loops"], "elapsed_s": round(usage["elapsed_s"], 3), "error": usage["error"]}}
            if stream:
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": fin}], "usage": u}
                yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            else:
                if kind == "chat":
                    msg = {"role": "assistant", "content": content}
                    if thinking: msg["reasoning"] = thinking
                    yield json.dumps({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model,
                                      "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": u})
                else:
                    yield json.dumps({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                                      "choices": [{"index": 0, "text": full, "finish_reason": fin}], "usage": u})
        if stream:
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(json.loads(next(gen())))

    @app.post("/v1/chat/completions")
    async def v1_chat(req: Request): return await openai_endpoint(req, "chat")
    @app.post("/v1/completions")
    async def v1_completions(req: Request): return await openai_endpoint(req, "completions")
    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--backend", choices=["ouro", "recurrent-qwen", "lds", "hf"], default="hf")
    ap.add_argument("--code-path", default=None, help="for --backend lds: directory containing LoopUS models/ and utils/ (default third_party/LoopUS)")
    ap.add_argument("--name", default=None, help="model name exposed to clients")
    ap.add_argument("--loops", type=int, default=None, help="default loop / recurrence count")
    ap.add_argument("--loop-config-field", default=None, help="for --backend hf: config attribute holding loop count")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokenizer", default=None, help="tokenizer repo/path if different from --model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()
    name = a.name or a.model.split("/")[-1].lower()
    be = HFBackend(a.model, a.backend, getattr(torch, a.dtype), a.device, a.loops, a.loop_config_field, a.revision, a.tokenizer, a.code_path)
    uvicorn.run(make_app(be, name), host=a.host, port=a.port, log_level="warning")

if __name__ == "__main__":
    main()
