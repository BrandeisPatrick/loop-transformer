#!/usr/bin/env python
"""Read GGUF metadata KV pairs from a local file or an HTTP URL (range request), without downloading tensors."""
import struct, sys, io, requests

GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}

class R:
    def __init__(self, b): self.b=b; self.p=0
    def read(self, n):
        if self.p+n > len(self.b): raise EOFError("need more bytes")
        v=self.b[self.p:self.p+n]; self.p+=n; return v
    def val(self, t):
        if t==8:
            n=struct.unpack('<Q', self.read(8))[0]; return self.read(n).decode('utf-8','replace')
        if t==9:
            et=struct.unpack('<I', self.read(4))[0]; n=struct.unpack('<Q', self.read(8))[0]
            return [self.val(et) for _ in range(n)]
        f=FMT[t]; return struct.unpack(f, self.read(struct.calcsize(f)))[0]

def parse(b, max_arr=8):
    r=R(b); assert r.read(4)==b'GGUF'
    ver=struct.unpack('<I', r.read(4))[0]; nt=struct.unpack('<Q', r.read(8))[0]; nkv=struct.unpack('<Q', r.read(8))[0]
    out={'gguf_version':ver,'n_tensors':nt,'n_kv':nkv}
    for _ in range(nkv):
        k=r.val(8); t=struct.unpack('<I', r.read(4))[0]; v=r.val(t)
        if isinstance(v, list): v = f"<array len={len(v)} first={v[:max_arr]}>"
        out[k]=v
    return out

def load(src, nbytes):
    if src.startswith('http'):
        h={'Range': f'bytes=0-{nbytes-1}'}
        return requests.get(src, headers=h, timeout=120, allow_redirects=True).content
    return open(src,'rb').read(nbytes)

if __name__=='__main__':
    src=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 16_000_000
    b=load(src, n)
    try: md=parse(b)
    except EOFError:
        b=load(src, n*4); md=parse(b)
    for k,v in md.items():
        if k.startswith('tokenizer.ggml.') and k not in ('tokenizer.ggml.model','tokenizer.ggml.pre'): continue
        print(f"{k} = {str(v)[:160]}")
