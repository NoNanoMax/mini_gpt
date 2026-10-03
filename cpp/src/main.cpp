// CLI: генерация текста из модели.
#include "model.hpp"

int main(int argc, char** argv) {
    std::string dir, prompt = "";
    int n = 128, top_k = 40;
    float temp = 0.8f;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "%s требует аргумент\n", name); exit(1); }
            return argv[++i];
        };
        if (a == "--model") dir = next("--model");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--n") n = atoi(next("--n"));
        else if (a == "--temp") temp = (float)atof(next("--temp"));
        else if (a == "--top-k") top_k = atoi(next("--top-k"));
        else if (a == "--seed") seed = strtoull(next("--seed"), nullptr, 10);
        else if (a == "--help") {
            printf("MicroGPT C++ инференс\n"
                   "  --model DIR       папка с weights.safetensors, config.json, tokenizer.json\n"
                   "  --prompt TEXT     промпт (пусто — генерация с <bos>)\n"
                   "  --n N             число токенов (128)\n"
                   "  --temp T          температура (0.8)\n"
                   "  --top-k K         top-k (40)\n"
                   "  --seed S          сид (42)\n");
            return 0;
        }
    }
    if (dir.empty()) { fprintf(stderr, "нужен --model\n"); return 1; }

    fprintf(stderr, "Загрузка модели...\n");
    auto t0 = std::chrono::steady_clock::now();
    Model model = Model::load(dir);
    CppTokenizer tok = CppTokenizer::load(dir + "/tokenizer.json");
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "  %d слоёв, %d эмбеддингов, загрузка %.2fs\n",
            model.cfg.n_layers, model.cfg.n_embd,
            std::chrono::duration<double>(t1 - t0).count());

    std::vector<int> ids = prompt.empty() ? std::vector<int>{tok.vocab.at("<bos>")} : tok.encode(prompt);
    if (ids.size() > (size_t)model.cfg.max_seq_len) {
        fprintf(stderr, "промпт длиннее контекста (%d)\n", model.cfg.max_seq_len);
        return 1;
    }

    std::mt19937 rng(seed);
    std::vector<KVCache> caches(model.cfg.n_layers,
                                KVCache(model.cfg.n_kv_heads, model.cfg.head_dim()));
    std::vector<float> logits;
    int pos = 0;

    auto t2 = std::chrono::steady_clock::now();
    model.forward(ids, pos, caches, logits);
    pos += ids.size();
    for (int i = 0; i < n; i++) {
        std::vector<float> row(logits.begin() + (logits.size() / model.cfg.vocab_size - 1) *
                                model.cfg.vocab_size, logits.end());
        int id = sample_token(row, top_k, temp, rng);
        if (id == tok.vocab.at("<eos>")) break;
        fputs(tok.id_to_piece[id].c_str(), stdout);
        fflush(stdout);
        std::vector<int> one = {id};
        model.forward(one, pos, caches, logits);
        pos++;
    }
    auto t3 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t3 - t2).count();
    fprintf(stderr, "\n%.2fs, %.0f токенов/с\n", dt, n / dt);
    printf("\n");
    return 0;
}
