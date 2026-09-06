"""Stream-convert tomg-group-umd/huginn-0125 fp32 -> local bf16, one shard at a time.

Why: the repo is 15.65 GB of float32 and shard 4 (1.384 GB) is nothing but a duplicate of
the TIED embedding (index maps only 'lm_head.weight' there; config.tie_word_embeddings=True
and the model ties lm_head.weight <- transformer.wte.weight at load).  Downloading all four
shards and then also holding the bf16 copy needs ~23 GB free.  This script needs ~10 GB peak
and leaves 7.2 GB on disk.

    ./.venv-tf4/bin/python scripts/huginn_fetch_bf16.py /Volumes/ext/huginn-0125-bf16
"""
import json, os, shutil, sys
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO = "tomg-group-umd/huginn-0125"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/huginn-0125-bf16")
SMALL = ["config.json", "generation_config.json", "raven_config_minimal.py",
         "raven_modeling_minimal.py", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json"]
SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in (1, 2, 3)]  # 4 = duplicate tied embedding

os.makedirs(OUT, exist_ok=True)
for f in SMALL:
    shutil.copyfile(hf_hub_download(REPO, f), os.path.join(OUT, f))
cfg = json.load(open(os.path.join(OUT, "config.json")))
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

index = {"metadata": {"total_size": 0}, "weight_map": {}}
for shard in SHARDS:
    src = hf_hub_download(REPO, shard)                       # ~4.8 GB
    sd = {k: v.to(torch.bfloat16) for k, v in load_file(src).items()}
    save_file(sd, os.path.join(OUT, shard), metadata={"format": "pt"})
    for k, v in sd.items():
        index["weight_map"][k] = shard
        index["metadata"]["total_size"] += v.numel() * 2
    del sd
    os.remove(src)                                           # drop the fp32 shard immediately
json.dump(index, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
print("wrote", OUT, index["metadata"]["total_size"] / 1e9, "GB bf16")
print("lm_head.weight intentionally absent - it is tied to transformer.wte.weight")
