/**
 * @file llm_engine.cpp
 * @brief Forward pass for llama2.c-format checkpoints, ported for ESP32-S3.
 *
 *  ---- Made by @Doominator1 on GitHub, Jul 2026
 *  ---- Much thanks to @karpathy for the base of this project; llama2.c!
 *
 * File-backed (no mmap) port of karpathy/llama2.c's run.c. Reads the "v2"
 * export format written by export.py's `version()` path:
 *   uint32 magic ("ak42" as little-endian int, i.e. 0x616b3432)
 *   int32  version (0 = fp32, 2 = int8/Q8_0 quantized)
 *   Config (7 x int32)
 *   uint8  shared_classifier
 *   int32  group_size (quantized only)
 *   ... weights ...
 * Legacy (no-header) checkpoints are not supported - re-export with export.py.
 */
#include "llm_engine.h"

#include <algorithm>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace bruce_llm {

namespace {

constexpr uint32_t kMagic = 0x616b3432; // "ak42"

void *psram_alloc(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p;
}

struct QuantizedTensor {
    int8_t *q = nullptr; // quantized values
    float *s = nullptr;  // per-group scales
};

float *readF32Vec(File &f, size_t n) {
    float *buf = (float *)psram_alloc(n * sizeof(float));
    if (!buf) return nullptr;
    f.read((uint8_t *)buf, n * sizeof(float));
    return buf;
}

bool readQuantized(File &f, QuantizedTensor &t, size_t n, int groupSize) {
    t.q = (int8_t *)psram_alloc(n);
    size_t nGroups = n / groupSize;
    t.s = (float *)psram_alloc(nGroups * sizeof(float));
    if (!t.q || !t.s) return false;
    f.read((uint8_t *)t.q, n);
    f.read((uint8_t *)t.s, nGroups * sizeof(float));
    return true;
}

void dequantize(const QuantizedTensor &t, float *out, size_t n, int groupSize) {
    for (size_t i = 0; i < n; i++) out[i] = t.q[i] * t.s[i / groupSize];
}

// View into one layer's slice of a multi-layer quantized tensor (no copy).
QuantizedTensor layerView(const QuantizedTensor &t, size_t elemOffset, int groupSize) {
    QuantizedTensor v;
    v.q = t.q + elemOffset;
    v.s = t.s + elemOffset / groupSize;
    return v;
}

void quantize(QuantizedTensor &t, const float *x, size_t n, int groupSize) {
    size_t nGroups = n / groupSize;
    for (size_t g = 0; g < nGroups; g++) {
        float wmax = 0.0f;
        for (int i = 0; i < groupSize; i++) {
            float v = fabsf(x[g * groupSize + i]);
            if (v > wmax) wmax = v;
        }
        float scale = wmax / 127.0f;
        t.s[g] = scale;
        for (int i = 0; i < groupSize; i++) {
            float v = x[g * groupSize + i] / (scale == 0 ? 1.0f : scale);
            t.q[g * groupSize + i] = (int8_t)roundf(v);
        }
    }
}

// out(d) = W(d,n) @ x(n), quantized weights, fp32 activations (quantizes x on the fly).
void matmulQ(float *out, const float *x, const QuantizedTensor &w, int n, int d, int groupSize) {
    QuantizedTensor xq;
    xq.q = (int8_t *)alloca(n);
    xq.s = (float *)alloca((n / groupSize) * sizeof(float));
    quantize(xq, x, n, groupSize);

    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - groupSize; j += groupSize) {
            for (int k = 0; k < groupSize; k++) ival += (int32_t)xq.q[j + k] * (int32_t)w.q[in + j + k];
            val += ((float)ival) * w.s[(in + j) / groupSize] * xq.s[j / groupSize];
            ival = 0;
        }
        out[i] = val;
    }
}

void matmulF(float *out, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float *row = w + i * n;
        for (int j = 0; j < n; j++) val += row[j] * x[j];
        out[i] = val;
    }
}

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int i = 0; i < size; i++) out[i] = weight[i] * (x[i] * ss);
}

void softmax(float *x, int size) {
    float maxv = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

} // namespace

struct LLMEngine::Impl {
    bool quantized = false;
    bool legacy = false;
    int groupSize = 0;
    bool sharedClassifier = true;

    // fp32 weights (used when !quantized)
    float *tokEmbF = nullptr;
    float *rmsAttW = nullptr, *rmsFfnW = nullptr, *rmsFinalW = nullptr;
    float *wqF = nullptr, *wkF = nullptr, *wvF = nullptr, *woF = nullptr;
    float *w1F = nullptr, *w2F = nullptr, *w3F = nullptr;
    float *wclsF = nullptr;

    // quantized weights (used when quantized)
    QuantizedTensor tokEmbQ;
    QuantizedTensor wqQ, wkQ, wvQ, woQ;
    QuantizedTensor w1Q, w2Q, w3Q;
    QuantizedTensor wclsQ;
    float *tokEmbDequant = nullptr; // dequantized embedding table (small, kept fp32 for lookup)

    // run state
    float *x = nullptr, *xb = nullptr, *xb2 = nullptr;
    float *hb = nullptr, *hb2 = nullptr;
    float *q = nullptr, *att = nullptr, *logits = nullptr;
    float *keyCache = nullptr, *valCache = nullptr;

    // tokenizer
    std::unique_ptr<char *[]> vocab;
    std::unique_ptr<float[]> vocabScores;
    int vocabSize = 0;

    ~Impl() { freeAll(); }

    void freeAll() {
        auto f = [](void *p) {
            if (p) heap_caps_free(p);
        };
        f(tokEmbF);
        f(rmsAttW);
        f(rmsFfnW);
        f(rmsFinalW);
        f(wqF);
        f(wkF);
        f(wvF);
        f(woF);
        f(w1F);
        f(w2F);
        f(w3F);
        // wclsF aliases tokEmbF when the checkpoint has a shared classifier
        // (see load()) - freeing both would double-free the same allocation.
        if (wclsF != tokEmbF) f(wclsF);
        f(tokEmbQ.q);
        f(tokEmbQ.s);
        f(wqQ.q);
        f(wqQ.s);
        f(wkQ.q);
        f(wkQ.s);
        f(wvQ.q);
        f(wvQ.s);
        f(woQ.q);
        f(woQ.s);
        f(w1Q.q);
        f(w1Q.s);
        f(w2Q.q);
        f(w2Q.s);
        f(w3Q.q);
        f(w3Q.s);
        f(wclsQ.q);
        f(wclsQ.s);
        f(tokEmbDequant);
        f(x);
        f(xb);
        f(xb2);
        f(hb);
        f(hb2);
        f(q);
        f(att);
        f(logits);
        f(keyCache);
        f(valCache);
        if (vocab) {
            for (int i = 0; i < vocabSize; i++)
                if (vocab[i]) free(vocab[i]);
        }
    }
};

LLMEngine::LLMEngine() : impl(std::make_unique<Impl>()) {}
LLMEngine::~LLMEngine() = default;

LLMLoadError LLMEngine::load(
    FS &fs, const String &checkpointPath, const String &tokenizerPath, bool overrideSafetyChecks
) {
    unload();
    impl = std::make_unique<Impl>();

    if (!fs.exists(checkpointPath)) return LLMLoadError::CheckpointNotFound;
    if (!fs.exists(tokenizerPath)) return LLMLoadError::TokenizerNotFound;

    File cf = fs.open(checkpointPath, FILE_READ);
    if (!cf) return LLMLoadError::CheckpointNotFound;

    // export.py's legacy_export() (--version 0, the original karpathy/tinyllamas
    // .bin files) writes NO magic/header at all - the file starts directly
    // with the 7-int Config struct. version1_export()/version2_export() both
    // start with a 4-byte "ak42" magic. We tell them apart by trying the
    // magic first and falling back to legacy parsing if it doesn't match,
    // rather than rejecting anything header-less outright.
    uint32_t magic = 0;
    cf.read((uint8_t *)&magic, 4);
    bool legacy = (magic != kMagic);

    int32_t version = 0;
    uint8_t sharedClassifierByte = 1;
    int32_t groupSize = 0;

    if (legacy) {
        // Legacy format signals shared_classifier via the sign of vocab_size
        // instead of an explicit byte, and has no version field - the Config
        // struct starts at byte 0, so back up over the 4 bytes we already
        // spec-ulatively read as "magic".
        cf.seek(0, SeekSet);
        LLMConfig c{};
        cf.read((uint8_t *)&c, sizeof(LLMConfig));
        sharedClassifierByte = (c.vocab_size > 0) ? 1 : 0;
        c.vocab_size = c.vocab_size > 0 ? c.vocab_size : -c.vocab_size;
        cfg = c;
    } else {
        // export.py's model_export(): version 1 is the header-having fp32
        // format, version 2 is header-having int8/Q8_0. (There is no
        // magic-having version 0 - that version number is the legacy format
        // above, which never writes a magic at all.)
        cf.read((uint8_t *)&version, 4);
        if (version != 1 && version != 2) {
            cf.close();
            return LLMLoadError::BadMagicOrVersion;
        }

        // Unlike the legacy format, v1/v2 write vocab_size as a plain
        // positive int and signal shared_classifier via an explicit byte
        // below - no sign trick to undo here.
        LLMConfig c{};
        cf.read((uint8_t *)&c, sizeof(LLMConfig));
        cfg = c;

        // Both v1 and v2 write a shared_classifier byte here; only v2
        // additionally writes a group_size int before the header is
        // zero-padded out to 256 bytes.
        cf.read((uint8_t *)&sharedClassifierByte, 1);
        if (version == 2) cf.read((uint8_t *)&groupSize, 4);
        cf.seek(256, SeekSet);
    }

    impl->legacy = legacy;
    impl->quantized = (version == 2);
    impl->groupSize = groupSize;
    impl->sharedClassifier = sharedClassifierByte != 0;

    int dim = cfg.dim, hidden = cfg.hidden_dim, layers = cfg.n_layers;
    int heads = cfg.n_heads, kvHeads = cfg.n_kv_heads, vocab = cfg.vocab_size, seqLen = cfg.seq_len;
    int headSize = dim / heads;
    int kvDim = (dim * kvHeads) / heads;

    // quantize_q80 flattens each whole weight tensor into consecutive
    // group_size-sized chunks with no regard for row boundaries. Our matmul
    // (like reference run.c's matmul()) only sums whole groups within each
    // row's own local range, so it silently drops/misaligns data whenever a
    // row length isn't itself a multiple of group_size. dim, kvDim, and
    // hidden are all used as row lengths (n) somewhere in the forward pass,
    // so all three must divide evenly or generation will be garbage.
    if (!overrideSafetyChecks && impl->quantized &&
        (dim % groupSize != 0 || kvDim % groupSize != 0 || hidden % groupSize != 0)) {
        cf.close();
        return LLMLoadError::IncompatibleGroupSize;
    }

    // Rough PSRAM budget guard - refuse loads that clearly won't fit, unless
    // the caller has already warned the user and wants to try anyway.
    size_t psramFree = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t approxBytes = (size_t)vocab * dim * (impl->quantized ? 1 : 4) +
                         (size_t)layers * dim * dim * 4 * (impl->quantized ? 1 : 4) +
                         (size_t)layers * dim * hidden * 3 * (impl->quantized ? 1 : 4);
    if (!overrideSafetyChecks && approxBytes > psramFree * 0.85) {
        cf.close();
        return LLMLoadError::ConfigTooLarge;
    }

    if (!impl->quantized && impl->legacy) {
        // Matches legacy_export()'s exact write order: embedding table
        // first, then every layer's attention weights/norms, then FFN
        // weights/norm, then the final norm, then two freq_cis tables
        // (precomputed RoPE cos/sin - unused here since we compute RoPE on
        // the fly) that must be skipped over, then an optional classifier.
        impl->tokEmbF = readF32Vec(cf, (size_t)vocab * dim);
        impl->rmsAttW = readF32Vec(cf, (size_t)layers * dim);
        impl->wqF = readF32Vec(cf, (size_t)layers * dim * (heads * headSize));
        impl->wkF = readF32Vec(cf, (size_t)layers * dim * (kvHeads * headSize));
        impl->wvF = readF32Vec(cf, (size_t)layers * dim * (kvHeads * headSize));
        impl->woF = readF32Vec(cf, (size_t)layers * (heads * headSize) * dim);
        impl->rmsFfnW = readF32Vec(cf, (size_t)layers * dim);
        impl->w1F = readF32Vec(cf, (size_t)layers * dim * hidden);
        impl->w2F = readF32Vec(cf, (size_t)layers * hidden * dim);
        impl->w3F = readF32Vec(cf, (size_t)layers * dim * hidden);
        impl->rmsFinalW = readF32Vec(cf, dim);
        cf.seek((size_t)seqLen * (headSize / 2) * sizeof(float) * 2, SeekCur); // freqs_cos + freqs_sin
        impl->wclsF = impl->sharedClassifier ? impl->tokEmbF : readF32Vec(cf, (size_t)vocab * dim);
    } else if (!impl->quantized) {
        // Matches version1_export()'s exact write order: norms first, then
        // the embedding table, then every layer's attention/FFN weights, then
        // optionally an unshared classifier. There is no freq_cis table in
        // this header-having format - RoPE is computed on the fly instead.
        impl->rmsAttW = readF32Vec(cf, (size_t)layers * dim);
        impl->rmsFfnW = readF32Vec(cf, (size_t)layers * dim);
        impl->rmsFinalW = readF32Vec(cf, dim);
        impl->tokEmbF = readF32Vec(cf, (size_t)vocab * dim);
        impl->wqF = readF32Vec(cf, (size_t)layers * dim * (heads * headSize));
        impl->wkF = readF32Vec(cf, (size_t)layers * dim * (kvHeads * headSize));
        impl->wvF = readF32Vec(cf, (size_t)layers * dim * (kvHeads * headSize));
        impl->woF = readF32Vec(cf, (size_t)layers * (heads * headSize) * dim);
        impl->w1F = readF32Vec(cf, (size_t)layers * dim * hidden);
        impl->w2F = readF32Vec(cf, (size_t)layers * hidden * dim);
        impl->w3F = readF32Vec(cf, (size_t)layers * dim * hidden);
        impl->wclsF = impl->sharedClassifier ? impl->tokEmbF : readF32Vec(cf, (size_t)vocab * dim);
    } else {
        int gs = groupSize;
        impl->rmsAttW = readF32Vec(cf, (size_t)layers * dim);
        impl->rmsFfnW = readF32Vec(cf, (size_t)layers * dim);
        impl->rmsFinalW = readF32Vec(cf, dim);

        readQuantized(cf, impl->tokEmbQ, (size_t)vocab * dim, gs);
        readQuantized(cf, impl->wqQ, (size_t)layers * dim * (heads * headSize), gs);
        readQuantized(cf, impl->wkQ, (size_t)layers * dim * (kvHeads * headSize), gs);
        readQuantized(cf, impl->wvQ, (size_t)layers * dim * (kvHeads * headSize), gs);
        readQuantized(cf, impl->woQ, (size_t)layers * (heads * headSize) * dim, gs);
        readQuantized(cf, impl->w1Q, (size_t)layers * dim * hidden, gs);
        readQuantized(cf, impl->w2Q, (size_t)layers * hidden * dim, gs);
        readQuantized(cf, impl->w3Q, (size_t)layers * dim * hidden, gs);
        if (!impl->sharedClassifier) readQuantized(cf, impl->wclsQ, (size_t)vocab * dim, gs);

        impl->tokEmbDequant = (float *)psram_alloc((size_t)vocab * dim * sizeof(float));
        if (impl->tokEmbDequant) dequantize(impl->tokEmbQ, impl->tokEmbDequant, (size_t)vocab * dim, gs);
    }
    cf.close();

    // Tokenizer: llama2.c tokenizer.bin format = max_token_length(int32) then
    // per-token: score(float) len(int32) bytes[len]
    File tf = fs.open(tokenizerPath, FILE_READ);
    if (!tf) return LLMLoadError::TokenizerNotFound;
    int32_t maxTokLen = 0;
    tf.read((uint8_t *)&maxTokLen, 4);
    impl->vocabSize = vocab;
    impl->vocab = std::make_unique<char *[]>(vocab);
    impl->vocabScores = std::make_unique<float[]>(vocab);
    for (int i = 0; i < vocab; i++) {
        tf.read((uint8_t *)&impl->vocabScores[i], 4);
        int32_t len = 0;
        tf.read((uint8_t *)&len, 4);
        char *s = (char *)malloc(len + 1);
        tf.read((uint8_t *)s, len);
        s[len] = '\0';
        impl->vocab[i] = s;
    }
    tf.close();

    // run-state buffers (small, fp32, PSRAM)
    impl->x = (float *)psram_alloc(dim * sizeof(float));
    impl->xb = (float *)psram_alloc(dim * sizeof(float));
    impl->xb2 = (float *)psram_alloc(dim * sizeof(float));
    impl->hb = (float *)psram_alloc(hidden * sizeof(float));
    impl->hb2 = (float *)psram_alloc(hidden * sizeof(float));
    impl->q = (float *)psram_alloc(dim * sizeof(float));
    impl->att = (float *)psram_alloc(heads * seqLen * sizeof(float));
    impl->logits = (float *)psram_alloc(vocab * sizeof(float));
    impl->keyCache = (float *)psram_alloc((size_t)layers * seqLen * kvDim * sizeof(float));
    impl->valCache = (float *)psram_alloc((size_t)layers * seqLen * kvDim * sizeof(float));

    if (!impl->x || !impl->xb || !impl->xb2 || !impl->hb || !impl->hb2 || !impl->q || !impl->att ||
        !impl->logits || !impl->keyCache || !impl->valCache) {
        return LLMLoadError::OutOfMemory;
    }

    loaded = true;
    return LLMLoadError::None;
}

void LLMEngine::unload() {
    impl = std::make_unique<Impl>();
    loaded = false;
}

namespace {

int sampleArgmax(const float *probabilities, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probabilities[i] > probabilities[best]) best = i;
    return best;
}

int sampleMult(const float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1; // rounding fallback
}

// Faithful port of run.c's sample_topp(): nucleus sampling over the smallest
// set of tokens whose cumulative probability exceeds topP, so the tail of
// very-low-probability tokens can never be picked.
int sampleTopp(const float *probabilities, int n, float topP, float coin) {
    std::vector<std::pair<float, int>> probIndex;
    probIndex.reserve(n);
    float cutoff = (1.0f - topP) / (n - 1);
    for (int i = 0; i < n; i++)
        if (probabilities[i] >= cutoff) probIndex.push_back({probabilities[i], i});
    std::sort(probIndex.begin(), probIndex.end(), [](auto &a, auto &b) { return a.first > b.first; });

    float cumulative = 0.0f;
    int lastIdx = (int)probIndex.size() - 1;
    for (size_t i = 0; i < probIndex.size(); i++) {
        cumulative += probIndex[i].first;
        if (cumulative > topP) {
            lastIdx = (int)i;
            break;
        }
    }

    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= lastIdx; i++) {
        cdf += probIndex[i].first;
        if (r < cdf) return probIndex[i].second;
    }
    return probIndex[lastIdx].second;
}

// xorshift RNG, identical to run.c's random_u32/random_f32 - lets a fixed
// seed reproduce the same output for the same prompt, which esp_random()
// (a hardware TRNG) can't do.
uint32_t randomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 0x2545F4914F6CDD1Dull) >> 32);
}
float randomF32(uint64_t &state) { return (randomU32(state) >> 8) / 16777216.0f; }

int sampleToken(float *logits, int n, float temperature, float topP, uint64_t &rngState) {
    if (temperature <= 0.0f) return sampleArgmax(logits, n);
    for (int i = 0; i < n; i++) logits[i] /= temperature;
    softmax(logits, n);
    float coin = randomF32(rngState);
    if (topP <= 0.0f || topP >= 1.0f) return sampleMult(logits, n, coin);
    return sampleTopp(logits, n, topP, coin);
}

// Not part of upstream llama2.c. Downweights logits of recently-used tokens
// so tiny models are less likely to loop on the same phrase - the standard
// HF-transformers-style repetition penalty (divide positive logits, multiply
// negative ones, by the penalty).
void applyRepetitionPenalty(float *logits, const std::vector<int> &history, float penalty) {
    if (penalty == 1.0f) return;
    for (int id : history) {
        if (logits[id] > 0) logits[id] /= penalty;
        else logits[id] *= penalty;
    }
}

// llama2.c's vocab stores raw bytes that have no printable token as
// "<0xXX>" fallback pieces (see run.c's decode()). Those must be turned back
// into the literal byte, or streamed output looks like hex garbage.
String decodePiece(const char *piece) {
    unsigned int byteVal;
    if (piece[0] == '<' && sscanf(piece, "<0x%02X>", &byteVal) == 1) {
        char c = (char)byteVal;
        return String(c);
    }
    return String(piece);
}

String encodeBpeGreedy(const String &text, char **vocab, int vocabSize, int *outIds, int &outCount) {
    // Minimal byte-level fallback tokenizer: match the longest vocab entry at
    // each position, else emit the raw byte token if present in vocab.
    outCount = 0;
    size_t i = 0;
    while (i < (size_t)text.length()) {
        int bestId = -1, bestLen = 0;
        for (int v = 0; v < vocabSize; v++) {
            size_t l = strlen(vocab[v]);
            if (l > 0 && l <= text.length() - i && text.substring(i, i + l) == vocab[v]) {
                if ((int)l > bestLen) {
                    bestLen = l;
                    bestId = v;
                }
            }
        }
        if (bestId < 0) {
            bestLen = 1;
            bestId = 0; // unknown -> id 0 by convention
        }
        outIds[outCount++] = bestId;
        i += bestLen;
    }
    return "";
}
} // namespace

void LLMEngine::generate(
    const String &prompt, int maxTokens, const GenerationParams &params, const TokenCallback &onToken
) {
    if (!loaded) return;
    Impl *m = impl.get();
    uint64_t rngState =
        params.seed != 0 ? (uint64_t)params.seed : (((uint64_t)esp_random() << 32) | esp_random());
    std::vector<int> history;
    int dim = cfg.dim, hidden = cfg.hidden_dim, layers = cfg.n_layers;
    int heads = cfg.n_heads, kvHeads = cfg.n_kv_heads, vocab = cfg.vocab_size, seqLen = cfg.seq_len;
    int headSize = dim / heads;
    int kvDim = (dim * kvHeads) / heads;
    int kvMul = heads / kvHeads;
    int gs = m->groupSize;

    // Matches run.c's encode(): BOS token (id 1) first, then sentencepiece's
    // "dummy prefix" space token, then the actual encoded text. Skipping
    // these (as an earlier version of this engine did) feeds the model a
    // prompt shaped differently from anything it saw in training.
    int *bodyIds = (int *)alloca(sizeof(int) * (prompt.length() + 1));
    int nBody = 0;
    encodeBpeGreedy(prompt, m->vocab.get(), m->vocabSize, bodyIds, nBody);

    int spaceId = 0;
    for (int v = 0; v < m->vocabSize; v++) {
        if (strcmp(m->vocab[v], " ") == 0) {
            spaceId = v;
            break;
        }
    }

    int *promptIds = (int *)alloca(sizeof(int) * (nBody + 2));
    int nPrompt = 0;
    promptIds[nPrompt++] = 1; // BOS
    if (prompt.length() > 0) promptIds[nPrompt++] = spaceId;
    for (int i = 0; i < nBody; i++) promptIds[nPrompt++] = bodyIds[i];

    int steps = maxTokens < seqLen ? maxTokens : seqLen;
    int token = promptIds[0];

    for (int pos = 0; pos < steps; pos++) {
        // --- forward pass for `token` at position `pos` ---
        memcpy(
            m->x, (m->quantized ? m->tokEmbDequant : m->tokEmbF) + (size_t)token * dim, dim * sizeof(float)
        );

        for (int l = 0; l < layers; l++) {
            rmsnorm(m->xb, m->x, m->rmsAttW + l * dim, dim);

            float *kRow = m->keyCache + ((size_t)l * seqLen + pos) * kvDim;
            float *vRow = m->valCache + ((size_t)l * seqLen + pos) * kvDim;

            if (!m->quantized) {
                matmulF(m->q, m->xb, m->wqF + (size_t)l * dim * dim, dim, dim);
                matmulF(kRow, m->xb, m->wkF + (size_t)l * dim * kvDim, dim, kvDim);
                matmulF(vRow, m->xb, m->wvF + (size_t)l * dim * kvDim, dim, kvDim);
            } else {
                matmulQ(m->q, m->xb, layerView(m->wqQ, (size_t)l * dim * dim, gs), dim, dim, gs);
                matmulQ(kRow, m->xb, layerView(m->wkQ, (size_t)l * dim * kvDim, gs), dim, kvDim, gs);
                matmulQ(vRow, m->xb, layerView(m->wvQ, (size_t)l * dim * kvDim, gs), dim, kvDim, gs);
            }

            // RoPE rotation
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < headSize; i += 2) {
                    float freq = 1.0f / powf(10000.0f, (float)i / headSize);
                    float val = pos * freq;
                    float fcr = cosf(val), fci = sinf(val);
                    int base = h * headSize;
                    if (h < kvHeads) {
                        float v0 = kRow[base + i], v1 = kRow[base + i + 1];
                        kRow[base + i] = v0 * fcr - v1 * fci;
                        kRow[base + i + 1] = v0 * fci + v1 * fcr;
                    }
                    float v0 = m->q[base + i], v1 = m->q[base + i + 1];
                    m->q[base + i] = v0 * fcr - v1 * fci;
                    m->q[base + i + 1] = v0 * fci + v1 * fcr;
                }
            }

            for (int h = 0; h < heads; h++) {
                float *qh = m->q + h * headSize;
                float *attRow = m->att + h * seqLen;
                for (int t = 0; t <= pos; t++) {
                    float *kt = m->keyCache + ((size_t)l * seqLen + t) * kvDim + (h / kvMul) * headSize;
                    float score = 0.0f;
                    for (int i = 0; i < headSize; i++) score += qh[i] * kt[i];
                    attRow[t] = score / sqrtf((float)headSize);
                }
                softmax(attRow, pos + 1);
                float *out = m->xb2 + h * headSize;
                memset(out, 0, headSize * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float *vt = m->valCache + ((size_t)l * seqLen + t) * kvDim + (h / kvMul) * headSize;
                    float a = attRow[t];
                    for (int i = 0; i < headSize; i++) out[i] += a * vt[i];
                }
            }

            if (!m->quantized) matmulF(m->xb, m->xb2, m->woF + (size_t)l * dim * dim, dim, dim);
            else matmulQ(m->xb, m->xb2, layerView(m->woQ, (size_t)l * dim * dim, gs), dim, dim, gs);

            for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];

            rmsnorm(m->xb, m->x, m->rmsFfnW + l * dim, dim);
            if (!m->quantized) {
                matmulF(m->hb, m->xb, m->w1F + (size_t)l * dim * hidden, dim, hidden);
                matmulF(m->hb2, m->xb, m->w3F + (size_t)l * dim * hidden, dim, hidden);
            } else {
                matmulQ(m->hb, m->xb, layerView(m->w1Q, (size_t)l * dim * hidden, gs), dim, hidden, gs);
                matmulQ(m->hb2, m->xb, layerView(m->w3Q, (size_t)l * dim * hidden, gs), dim, hidden, gs);
            }
            for (int i = 0; i < hidden; i++) {
                float v = m->hb[i];
                v *= 1.0f / (1.0f + expf(-v)); // SiLU
                m->hb[i] = v * m->hb2[i];
            }
            if (!m->quantized) matmulF(m->xb, m->hb, m->w2F + (size_t)l * hidden * dim, hidden, dim);
            else matmulQ(m->xb, m->hb, layerView(m->w2Q, (size_t)l * hidden * dim, gs), hidden, dim, gs);

            for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];
        }

        rmsnorm(m->x, m->x, m->rmsFinalW, dim);
        if (!m->quantized) matmulF(m->logits, m->x, m->wclsF, dim, vocab);
        else matmulQ(m->logits, m->x, m->sharedClassifier ? m->tokEmbQ : m->wclsQ, dim, vocab, gs);

        int nextToken;
        if (pos + 1 < nPrompt) {
            nextToken = promptIds[pos + 1];
        } else {
            applyRepetitionPenalty(m->logits, history, params.repetitionPenalty);
            nextToken = sampleToken(m->logits, vocab, params.temperature, params.topP, rngState);
        }

        if (pos >= nPrompt - 1) {
            String piece = decodePiece(m->vocab[token]);
            if (!onToken(piece)) return; // cancelled
        }
        // EOS (id 2, by llama2.c/sentencepiece convention) marks a natural
        // end of generation - stop instead of rambling on past it. Only
        // meaningful once we're actually generating, not echoing the prompt.
        if (pos >= nPrompt - 1 && nextToken == 2) return;
        history.push_back(token);
        if (history.size() > 64) history.erase(history.begin());
        token = nextToken;
    }
}

} // namespace bruce_llm
