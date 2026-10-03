// MicroGPT C++ инференс-движок.
//
// Бинарь:
//   microgpt --model experiments/<name> --prompt "..." --n 128 --temp 0.8 --top-k 40
//
// Читает weights.safetensors + config.json + tokenizer.json из папки модели.
//
// Сборка:
//   cmake -B build -S . && cmake --build build

#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------- JSON
// Минипарсер: объект, массив, строка (с \" \\ \uXXXX), число, true/false/null.
struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const {
        for (auto& [k, v] : obj) if (k == key) return &v;
        return nullptr;
    }
    long long as_int() const { return (long long)num; }
};

class JsonParser {
public:
    explicit JsonParser(const char* s) : s_(s), p_(0), len_(s ? std::strlen(s) : 0) {}

    JsonValue parse() {
        JsonValue v = parse_value();
        skip();
        if (p_ < len_) throw std::runtime_error("json: trailing data");
        return v;
    }

private:
    const char* s_;
    size_t p_;
    size_t len_;

    void skip() {
        while (p_ < len_ && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) p_++;
    }
    char peek() {
        if (p_ >= len_) throw std::runtime_error("json: unexpected EOF");
        return s_[p_];
    }
    void expect(char c) {
        if (p_ >= len_ || s_[p_] != c) throw std::runtime_error(std::string("json: expected '") + c + "'");
        p_++;
    }

    JsonValue parse_value() {
        skip();
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': { JsonValue v; v.type = JsonValue::String; v.str = parse_string(); return v; }
            case 't': p_ += 4; { JsonValue v; v.type = JsonValue::Bool; v.b = true; return v; }
            case 'f': p_ += 5; { JsonValue v; v.type = JsonValue::Bool; v.b = false; return v; }
            case 'n': p_ += 4; return JsonValue{};
            default: return parse_number();
        }
    }
    JsonValue parse_object() {
        JsonValue v; v.type = JsonValue::Object;
        expect('{');
        skip();
        if (peek() == '}') { p_++; return v; }
        while (true) {
            skip();
            std::string key = parse_string();
            skip();
            expect(':');
            v.obj.emplace_back(std::move(key), parse_value());
            skip();
            if (peek() == ',') { p_++; continue; }
            expect('}');
            break;
        }
        return v;
    }
    JsonValue parse_array() {
        JsonValue v; v.type = JsonValue::Array;
        expect('[');
        skip();
        if (peek() == ']') { p_++; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip();
            if (peek() == ',') { p_++; continue; }
            expect(']');
            break;
        }
        return v;
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (p_ < len_ && s_[p_] != '"') {
            char c = s_[p_++];
            if (c == '\\') {
                c = s_[p_++];
                switch (c) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        unsigned cp = 0;
                        for (int i = 0; i < 4; i++) {
                            char h = s_[p_++];
                            cp <<= 4;
                            cp += (h >= 'a') ? (h - 'a' + 10) : (h >= 'A') ? (h - 'A' + 10) : (h - '0');
                        }
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 63)); }
                        else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63)); }
                        break;
                    }
                    default: throw std::runtime_error("json: bad escape");
                }
            } else {
                out += c;
            }
        }
        expect('"');
        return out;
    }
    JsonValue parse_number() {
        size_t start = p_;
        if (peek() == '-') p_++;
        while (p_ < len_ && (isdigit((unsigned char)s_[p_]) || s_[p_] == '.' || s_[p_] == 'e' ||
               s_[p_] == 'E' || s_[p_] == '+' || s_[p_] == '-')) p_++;
        JsonValue v; v.type = JsonValue::Number;
        v.num = std::stod(std::string(s_ + start, p_ - start));
        return v;
    }
};

inline JsonValue json_parse(const std::string& s) { return JsonParser(s.c_str()).parse(); }

// ---------------------------------------------------------------- utf-8
inline std::vector<int32_t> utf8_decode(const std::string& s) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        int n = c < 0x80 ? 0 : c < 0xE0 ? 1 : c < 0xF0 ? 2 : 3;
        int32_t cp = c & (0xFF >> (n + 1));
        for (int j = 1; j <= n; j++) cp = (cp << 6) | (s[i + j] & 63);
        out.push_back(cp);
        i += n + 1;
    }
    return out;
}

inline std::string utf8_encode(int32_t cp) {
    std::string out;
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 63)); }
    else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 63)); out += (char)(0x80 | (cp & 63)); }
    return out;
}

// ---------------------------------------------------------------- файлы
inline std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("не найден: " + path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(n, '\0');
    if (n > 0 && fread(buf.data(), 1, n, f) != (size_t)n) throw std::runtime_error("ошибка чтения: " + path);
    fclose(f);
    return buf;
}

// ---------------------------------------------------------------- safetensors
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

inline Tensor to_float(const std::string& raw, const char* dtype, const int64_t* shape, int64_t ndim,
                       int64_t off) {
    Tensor t;
    for (int i = 0; i < ndim; i++) t.shape.push_back(shape[i]);
    int64_t n = t.numel();
    t.data.resize(n);
    const char* p = raw.data() + off;
    if (strcmp(dtype, "F32") == 0) {
        memcpy(t.data.data(), p, n * 4);
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t* h = (const uint16_t*)p;
        for (int64_t i = 0; i < n; i++) {
            uint16_t x = h[i];
            uint32_t sign = (uint32_t)(x >> 15) << 31;
            uint32_t exp = (x >> 10) & 0x1F;
            uint32_t mant = x & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                f = sign;
                if (mant) {
                    int e = 126 - 15;
                    while (!(mant & 0x400)) { mant <<= 1; e--; }
                    mant &= 0x3FF;
                    f = sign | ((uint32_t)e << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f = sign | 0x7F800000 | (mant << 13);
            } else {
                f = sign | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(&t.data[i], &f, 4);
        }
    } else {
        throw std::runtime_error(std::string("safetensors: неподдерживаемый dtype ") + dtype);
    }
    return t;
}

inline std::unordered_map<std::string, Tensor> load_safetensors(const std::string& path) {
    std::string raw = read_file(path);
    uint64_t hlen;
    memcpy(&hlen, raw.data(), 8);
    JsonValue h = json_parse(raw.substr(8, hlen));
    std::unordered_map<std::string, Tensor> out;
    for (auto& [name, meta] : h.obj) {
        int64_t shape[8];
        int ndim = (int)meta.find("shape")->arr.size();
        for (int i = 0; i < ndim; i++) shape[i] = meta.find("shape")->arr[i].as_int();
        const auto* off = meta.find("data_offsets")->arr.data();
        out[name] = to_float(raw, meta.find("dtype")->str.c_str(), shape, ndim, (int64_t)(8 + hlen) + off[0].as_int());
    }
    return out;
}
