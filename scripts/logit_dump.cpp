// Dump full-vocabulary logits at EVERY position for one or more token files, through libllama.
// Used by scripts/logit_parity.py to compare the GGUF against the transformers reference on the
// same token ids (so tokenizer differences cannot leak into the comparison).
//
//   logit_dump MODEL.gguf OUT_DIR NUM_LOOPS TOKENS1.txt [TOKENS2.txt ...]
//
// Each tokens file holds one token id per line. For each file F the tool writes OUT_DIR/F.bin:
// int32 n_pos, int32 n_vocab, then n_pos * n_vocab float32 logits (row-major by position).
// NUM_LOOPS 0 means "as stored in the file"; otherwise it is applied as the ouro.num_loops override.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<llama_token> read_tokens(const char * path) {
    std::vector<llama_token> toks;
    std::ifstream f(path);
    long long t;
    while (f >> t) toks.push_back((llama_token) t);
    return toks;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL.gguf OUT_DIR NUM_LOOPS TOKENS.txt [more.txt ...]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string out_dir = argv[2];
    const int n_loops = atoi(argv[3]);

    std::vector<std::vector<llama_token>> prompts;
    size_t max_len = 0;
    for (int i = 4; i < argc; ++i) {
        prompts.push_back(read_tokens(argv[i]));
        if (prompts.back().size() > max_len) max_len = prompts.back().size();
    }

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    std::vector<llama_model_kv_override> ov;
    if (n_loops > 0) {
        llama_model_kv_override o{};
        o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        strncpy(o.key, "ouro.num_loops", sizeof(o.key) - 1);
        o.val_i64 = n_loops;
        ov.push_back(o);
        llama_model_kv_override end{};
        end.key[0] = 0;
        ov.push_back(end);
        mp.kv_overrides = ov.data();
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 2; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) max_len + 8;
    cp.n_batch  = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.no_perf  = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 3; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init((int32_t) max_len, 0, 1);

    for (size_t p = 0; p < prompts.size(); ++p) {
        const auto & toks = prompts[p];
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = (llama_pos) i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;   // keep logits at every position
        }
        batch.n_tokens = (int32_t) toks.size();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed on prompt %zu\n", p); return 4; }

        std::string name = argv[4 + p];
        size_t slash = name.find_last_of('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);
        const std::string out_path = out_dir + "/" + name + ".bin";
        FILE * out = fopen(out_path.c_str(), "wb");
        if (!out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 5; }
        const int32_t hdr[2] = { (int32_t) toks.size(), (int32_t) n_vocab };
        fwrite(hdr, sizeof(int32_t), 2, out);
        for (size_t i = 0; i < toks.size(); ++i) {
            fwrite(llama_get_logits_ith(ctx, (int32_t) i), sizeof(float), n_vocab, out);
        }
        fclose(out);
        fprintf(stderr, "prompt %zu: %zu positions x %d vocab -> %s\n", p, toks.size(), n_vocab, out_path.c_str());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
