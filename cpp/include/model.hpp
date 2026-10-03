#pragma once
#include "llm.hpp"
#include <random>

// ---------------------------------------------------------------- конфиг
struct ModelConfig {
    int vocab_size, n_layers, n_heads, n_kv_heads, n_embd, max_seq_len;
    float norm_eps = 1e-6f;
    double rope_base = 10000.0;
    bool tie_word_emb = true;
    int head_dim() const { return n_embd / n_heads; }

    static ModelConfig from_json(const std::string& path) {
        JsonValue c = json_parse(read_file(path));
        ModelConfig m;
        m.vocab_size = (int)c.find("vocab_size")->as_int();
        m.n_layers = (int)c.find("n_layers")->as_int();
        m.n_heads = (int)c.find("n_heads")->as_int();
        m.n_kv_heads = (int)c.find("n_kv_heads")->as_int();
        m.n_embd = (int)c.find("n_embd")->as_int();
        m.max_seq_len = (int)c.find("max_seq_len")->as_int();
        m.norm_eps = (float)c.find("norm_eps")->num;
        m.rope_base = c.find("rope_base")->num;
        m.tie_word_emb = c.find("tie_word_emb")->b;
        return m;
    }
};

// ---------------------------------------------------------------- токенизатор
struct CppTokenizer {
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> merge_rank;

    static CppTokenizer load(const std::string& path) {
        JsonValue j = json_parse(read_file(path));
        CppTokenizer t;
        for (auto& [piece, id] : j.find("vocab")->obj) {
            t.vocab[piece] = (int)id.as_int();
            if ((int)t.id_to_piece.size() <= (int)id.as_int()) t.id_to_piece.resize(id.as_int() + 1);
            t.id_to_piece[id.as_int()] = piece;
        }
        for (size_t i = 0; i < j.find("merges")->arr.size(); i++) {
            auto& m = j.find("merges")->arr[i];
            t.merge_rank[m.arr[0].str + m.arr[1].str] = (int)i;
        }
        return t;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        size_t start = 0;
        bool first = true;
        while (start <= text.size()) {
            size_t sp = text.find(' ', start);
            bool last = sp == std::string::npos;
            size_t end = last ? text.size() : sp;
            if (end > start) {
                std::string word = text.substr(start, end - start) + (last ? "" : " ");
                std::vector<std::string> symbols;
                for (int32_t cp : utf8_decode(word)) symbols.push_back(utf8_encode(cp));
                for (size_t i = 0; i + 1 < symbols.size();) {
                    auto it = merge_rank.find(symbols[i] + symbols[i + 1]);
                    if (it != merge_rank.end()) {
                        symbols[i] = symbols[i] + symbols[i + 1];
                        symbols.erase(symbols.begin() + i + 1);
                    } else i++;
                }
                for (auto& s : symbols) ids.push_back(vocab.at(s));
            }
            if (last) break;
            start = sp + 1;
            first = false;
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) out += id_to_piece[id];
        return out;
    }
};

// ---------------------------------------------------------------- модель
struct Layer {
    Tensor qkv, o, gate, up, down, norm1, norm2;
};

struct KVCache {
    int n_kv, hd;
    int T = 0;
    std::vector<float> k, v;
    KVCache(int n, int h) : n_kv(n), hd(h) {}
    void push(const float* kk, const float* vv, int Tnew) {
        int Told = T;
        k.resize(n_kv * (T + Tnew) * hd, 0.f);
        v.resize(n_kv * (T + Tnew) * hd, 0.f);
        for (int h = 0; h < n_kv; h++)
            for (int t = 0; t < Tnew; t++) {
                memcpy(k.data() + (h * (Told + Tnew) + Told + t) * hd, kk + (t * n_kv + h) * hd, hd * sizeof(float));
                memcpy(v.data() + (h * (Told + Tnew) + Told + t) * hd, vv + (t * n_kv + h) * hd, hd * sizeof(float));
            }
        T = T + Tnew;
    }
};

struct Model {
    ModelConfig cfg;
    std::vector<Layer> layers;
    Tensor emb, norm, out;

    static Model load(const std::string& dir) {
        Model m;
        m.cfg = ModelConfig::from_json(dir + "/config.json");
        auto w = load_safetensors(dir + "/weights.safetensors");
        m.emb = w.at("tok_embeddings.weight");
        m.norm = w.at("norm.weight");
        m.out = m.cfg.tie_word_emb ? m.emb : w.at("output.weight");
        m.layers.resize(m.cfg.n_layers);
        for (int i = 0; i < m.cfg.n_layers; i++) {
            std::string p = "blocks." + std::to_string(i) + ".";
            m.layers[i].qkv = w.at(p + "attn.qkv.weight");
            m.layers[i].o = w.at(p + "attn.o.weight");
            m.layers[i].gate = w.at(p + "ffn.gate.weight");
            m.layers[i].up = w.at(p + "ffn.up.weight");
            m.layers[i].down = w.at(p + "ffn.down.weight");
            m.layers[i].norm1 = w.at(p + "norm1.weight");
            m.layers[i].norm2 = w.at(p + "norm2.weight");
        }
        w.clear();
        return m;
    }

    void rmsnorm(const float* x, const Tensor& w, float* y, int D) const {
        float ss = 0.f;
        for (int d = 0; d < D; d++) ss += x[d] * x[d];
        float r = 1.0f / std::sqrt(ss / D + cfg.norm_eps);
        for (int d = 0; d < D; d++) y[d] = x[d] * r * w.data[d];
    }

    // y[M, N] = A[M, K] @ W[N, K]^T
    void matmul(const float* A, const Tensor& W, float* y, int M, int N, int K) const {
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; i++) {
            const float* a = A + i * K;
            float* yo = y + i * N;
            for (int n = 0; n < N; n++) {
                float s = 0.f;
                const float* w = W.data.data() + n * K;
                for (int k = 0; k < K; k++) s += a[k] * w[k];
                yo[n] = s;
            }
        }
    }

    // x: [T, D] -> logits: [T, V]
    void forward(const std::vector<int>& ids, int pos_start,
                 std::vector<KVCache>& caches, std::vector<float>& logits) const {
        int T = (int)ids.size();
        int D = cfg.n_embd, V = cfg.vocab_size, hd = cfg.head_dim();
        int H = cfg.n_heads, HK = cfg.n_kv_heads, G = H / HK;

        std::vector<float> x(T * D);
        for (int t = 0; t < T; t++) memcpy(x.data() + t * D, emb.data.data() + ids[t] * D, D * sizeof(float));

        std::vector<float> cos(T * hd), sin(T * hd);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < hd / 2; i++) {
                double f = std::pow(cfg.rope_base, -2.0 * i / hd);
                float c = std::cos((pos_start + t) * f);
                float s = std::sin((pos_start + t) * f);
                cos[t * hd + i] = cos[t * hd + i + hd / 2] = c;
                sin[t * hd + i] = sin[t * hd + i + hd / 2] = s;
            }

        std::vector<float> buf(std::max(T * 3 * D, T * 4 * D));
        std::vector<float> q(T * H * hd), k(T * HK * hd), v(T * HK * hd), o(T * D);

        for (int li = 0; li < (int)layers.size(); li++) {
            auto& L = layers[li];
            std::vector<float> n1(T * D);
            for (int t = 0; t < T; t++) rmsnorm(x.data() + t * D, L.norm1, n1.data() + t * D, D);
            matmul(n1.data(), L.qkv, buf.data(), T, 3 * D, D);
            for (int t = 0; t < T; t++)
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < hd; i++)
                        q[t * H * hd + h * hd + i] = buf[(t * 3 + 0) * D + h * hd + i];
            for (int t = 0; t < T; t++)
                for (int h = 0; h < HK; h++)
                    for (int i = 0; i < hd; i++) {
                        k[t * HK * hd + h * hd + i] = buf[(t * 3 + 1) * D + h * hd + i];
                        v[t * HK * hd + h * hd + i] = buf[(t * 3 + 2) * D + h * hd + i];
                    }
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < H; h++) {
                    float* qp = q.data() + t * H * hd + h * hd;
                    for (int i = 0; i < hd / 2; i++) {
                        float a = qp[i];
                        float b = qp[i + hd / 2];
                        float c = cos[t * hd + i];
                        float s = sin[t * hd + i];
                        qp[i] = a * c - b * s;
                        qp[i + hd / 2] = b * c + a * s;
                    }
                }
                for (int h = 0; h < HK; h++) {
                    float* kp = k.data() + t * HK * hd + h * hd;
                    for (int i = 0; i < hd / 2; i++) {
                        float a = kp[i];
                        float b = kp[i + hd / 2];
                        float c = cos[t * hd + i];
                        float s = sin[t * hd + i];
                        kp[i] = a * c - b * s;
                        kp[i + hd / 2] = b * c + a * s;
                    }
                }
            }
            if (getenv("MG_DEBUG") && li == 0) {
                int tl = T - 1;
                printf("DBGQ ");
                for (int i = 0; i < hd; i++) printf("%f ", q[tl * H * hd + i]);
                printf("\nDBGK ");
                for (int i = 0; i < hd; i++) printf("%f ", k[0 * HK * hd + i]);
                printf("\n");
            }
            caches[li].push(k.data(), v.data(), T);

            int Tc = caches[li].T;
            std::vector<float> att(H * T * Tc);
            float scale = 1.0f / std::sqrt((float)hd);
            for (int h = 0; h < H; h++) {
                int hk = h / G;
                const float* kp = caches[li].k.data() + hk * Tc * hd;
                const float* vp = caches[li].v.data() + hk * Tc * hd;
                float* am = att.data() + h * T * Tc;
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (int t = 0; t < T; t++) {
                    const float* qp = q.data() + t * H * hd + h * hd;
                    for (int tc = 0; tc < Tc; tc++) {
                        float s = 0.f;
                        const float* krow = kp + tc * hd;
                        for (int i = 0; i < hd; i++) s += qp[i] * krow[i];
                        am[t * Tc + tc] = s * scale;
                    }
                }
                if (getenv("MG_DEBUG") && li == 0 && h == 0) {
                    printf("DBGATT ");
                    for (int t = 0; t < T; t++)
                        for (int tc = 0; tc < Tc; tc++) printf("%f ", att[t * Tc + tc]);
                    printf("\n");
                }
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (int t = 0; t < T; t++) {
                    float* row = am + t * Tc;
                    int lim = pos_start + t + 1;
                    for (int tc = Tc - (T - t); tc < Tc; tc++)
                        if (tc >= lim) row[tc] = -1e30f;
                    float mx = row[0];
                    for (int tc = 1; tc < Tc; tc++) mx = std::max(mx, row[tc]);
                    float sum = 0.f;
                    for (int tc = 0; tc < Tc; tc++) { row[tc] = std::exp(row[tc] - mx); sum += row[tc]; }
                    for (int tc = 0; tc < Tc; tc++) row[tc] /= sum;
                    float* od = o.data() + t * D + h * hd;
                    for (int i = 0; i < hd; i++) {
                        float s = 0.f;
                        const float* vrow = vp;
                        for (int tc = 0; tc < Tc; tc++, vrow += hd) s += row[tc] * vrow[i];
                        od[i] += s;
                    }
                }
            }
            if (getenv("MG_DEBUG") && li == 0) {
                FILE* fbin = fopen("/tmp/o.bin", "wb");
                fwrite(o.data(), sizeof(float), (size_t)T * D, fbin);
                fclose(fbin);
                printf("DBGO ");
                for (int i = 0; i < D; i++) printf("%f ", o[(T-1)*D+i]);
                printf("\n");
            }
            matmul(o.data(), L.o, buf.data(), T, D, D);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) x[t * D + d] += buf[t * D + d];
            if (getenv("MG_DEBUG") && li == 0) {
                printf("DBGX ");
                for (int i = 0; i < D; i++) printf("%f ", x[(T-1)*D+i]);
                printf("\n");
            }

            std::vector<float> n2(T * D);
            for (int t = 0; t < T; t++) rmsnorm(x.data() + t * D, L.norm2, n2.data() + t * D, D);
            if (getenv("MG_DEBUG") && li == 0) {
                FILE* fbin = fopen("/tmp/n2.bin", "wb");
                fwrite(n2.data() + (T-1)*D, sizeof(float), D, fbin);
                fclose(fbin);
            }
            matmul(n2.data(), L.gate, buf.data(), T, 4 * D, D);
            std::vector<float> up(T * 4 * D);
            matmul(n2.data(), L.up, up.data(), T, 4 * D, D);
            for (int i = 0; i < T * 4 * D; i++) {
                float g = buf[i];
                buf[i] = g * std::exp(g) / (1.f + std::exp(g)) * up[i];
            }
            std::vector<float> ffn_out(T * D);
            matmul(buf.data(), L.down, ffn_out.data(), T, D, 4 * D);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++) x[t * D + d] += ffn_out[t * D + d];
        }

        std::vector<float> xn(T * D);
        for (int t = 0; t < T; t++) rmsnorm(x.data() + t * D, norm, xn.data() + t * D, D);
        logits.assign(T * V, 0.f);
        matmul(xn.data(), out, logits.data(), T, V, D);
    }
};

// ---------------------------------------------------------------- sampler
inline int sample_token(const std::vector<float>& logits, int top_k, float temp, std::mt19937& rng) {
    std::vector<float> l = logits;
    if (temp > 1e-5f)
        for (auto& x : l) x /= temp;
    if (top_k > 0 && top_k < (int)l.size()) {
        std::vector<float> v(l.begin(), l.end());
        std::partial_sort(v.begin(), v.begin() + top_k, v.end(), std::greater<float>());
        float thr = v[top_k - 1];
        for (auto& x : l) if (x < thr) x = -1e30f;
    }
    float mx = *std::max_element(l.begin(), l.end());
    float sum = 0.f;
    std::vector<float> p(l.size());
    for (size_t i = 0; i < l.size(); i++) { p[i] = std::exp(l[i] - mx); sum += p[i]; }
    for (auto& x : p) x /= sum;
    std::discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}
