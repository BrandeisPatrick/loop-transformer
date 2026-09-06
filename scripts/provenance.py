#!/usr/bin/env python
"""Record exactly what produced a result: model revision sha, library versions, device, dtype.

The critic's point is fair -- trust_remote_code caches whatever `main` pointed at when the custom
modeling code was first fetched, so an unpinned run is not reproducible. This writes the resolved
commit sha for each model plus the local toolchain into results/provenance.json.
"""
import json, os, platform, subprocess, sys
import requests

MODELS = ["ByteDance/Ouro-1.4B", "Nanbeige/Nanbeige4.2-3B", "Thrillcrazyer/Qwen3_1.7B_LoopUS",
          "mshapiro123/recurrent-qwen2.5-0.5b-r16-adapter", "Qwen/Qwen2.5-0.5B-Instruct"]

def hf(repo):
    try:
        j = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30).json()
        return {"sha": j.get("sha"), "lastModified": j.get("lastModified"), "gated": j.get("gated")}
    except Exception as e:
        return {"error": repr(e)}

def venv(py):
    out = {}
    for mod in ("torch", "transformers", "mlx", "lm_eval"):
        try:
            out[mod] = subprocess.run([py, "-c", f"import {mod};print({mod}.__version__)"],
                                      capture_output=True, text=True, timeout=120).stdout.strip() or None
        except Exception:
            out[mod] = None
    return out

def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    prov = {"host": {"platform": platform.platform(), "machine": platform.machine()},
            "models": {r: hf(r) for r in MODELS},
            "venvs": {"default": venv(os.path.join(root, ".venv/bin/python")),
                      "tf4": venv(os.path.join(root, ".venv-tf4/bin/python"))}}
    try:
        prov["ollama"] = requests.get("http://127.0.0.1:11434/api/version", timeout=5).json()
        prov["ollama_models"] = [m["name"] for m in requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json()["models"]]
    except Exception as e:
        prov["ollama"] = {"error": repr(e)}
    p = os.path.join(root, "results", "provenance.json")
    json.dump(prov, open(p, "w"), indent=2)
    print(json.dumps(prov, indent=2))

if __name__ == "__main__":
    main()
