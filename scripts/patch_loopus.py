#!/usr/bin/env python
"""Make the vendored LoopUS modeling code work with the installed transformers.

LoopUS pins transformers>=5.0.0 and its config.json was written by 5.3.0. Between then and 5.16
`create_causal_mask` dropped its `cache_position` parameter (positions are now derived from
`past_key_values` and `position_ids`), so the shipped code raises

    TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'

on the first forward pass. Rather than pin an old transformers (a second ~1 GB env on a disk with
3 GB free), filter the kwargs to whatever the installed function actually accepts. That keeps the
file working on both old and new versions.

Idempotent: safe to run repeatedly. third_party/ is gitignored, so this script is the record of
the change.
"""
import inspect, os, sys

MARK = "# --- looplm compat patch"
PATCH = f'''{MARK}: drop kwargs the installed transformers no longer accepts ---
    import inspect as _inspect
    from transformers.masking_utils import create_causal_mask as _ccm
    _accepted = set(_inspect.signature(_ccm).parameters)
    mask_kwargs = {{k: v for k, v in mask_kwargs.items() if k in _accepted}}
    # --- end patch ---
'''

def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = os.path.join(root, "third_party", "LoopUS", "models", "modeling_lds.py")
    if not os.path.exists(path):
        sys.exit(f"not found: {path} (clone github.com/Thrillcrazyer/LoopUS into third_party/)")
    src = open(path).read()
    if MARK in src:
        print("already patched")
        return
    anchor = "    full_attention_mask = create_causal_mask(**mask_kwargs)"
    if anchor not in src:
        sys.exit("anchor line not found; LoopUS upstream changed, re-inspect modeling_lds.py")
    src = src.replace(anchor, PATCH + anchor, 1)
    open(path, "w").write(src)
    print(f"patched {path}")

if __name__ == "__main__":
    main()
