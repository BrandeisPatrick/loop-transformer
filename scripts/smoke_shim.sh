#!/bin/bash
# Usage: scripts/smoke_shim.sh <port>   — assumes shim already running on that port
P=${1:-11435}
echo "== /api/version"; curl -s http://127.0.0.1:$P/api/version; echo
echo "== /api/tags"; curl -s http://127.0.0.1:$P/api/tags | python3 -c "import sys,json; print([m['name'] for m in json.load(sys.stdin)['models']])"
echo "== /v1/models"; curl -s http://127.0.0.1:$P/v1/models | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])"
for L in 1 3; do
  echo "== /v1/chat/completions num_loops=$L"
  curl -s http://127.0.0.1:$P/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 + 26? Answer with just the number.\"}],\"max_tokens\":24,\"temperature\":0,\"num_loops\":$L}" | python3 -c "import sys,json; j=json.load(sys.stdin); print(repr(j['choices'][0]['message']['content']), j['usage'])"
done
echo "== /api/chat (stream, options.num_loops=2)"
curl -s http://127.0.0.1:$P/api/chat -H 'content-type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Name three primary colors."}],"options":{"num_loops":2,"num_predict":30}}' | python3 -c "
import sys,json
chunks=[json.loads(l) for l in sys.stdin if l.strip()]
print('chunks:', len(chunks), '| text:', repr(''.join(c.get('message',{}).get('content','') for c in chunks))[:120]); print('final:', {k:v for k,v in chunks[-1].items() if k in ('done','eval_count','prompt_eval_count','looplm')})"
echo "== /api/generate (non-stream)"
curl -s http://127.0.0.1:$P/api/generate -H 'content-type: application/json' -d '{"model":"x","prompt":"Say hello in French.","stream":false,"options":{"num_predict":16}}' | python3 -c "import sys,json; j=json.load(sys.stdin); print(repr(j['response']), j['looplm'])"
