/**
 * MedGemma JNI Bridge — MedLens App
 *
 * JNI wrapper around llama.cpp for running MedGemma GGUF models on Android.
 * Provides: init, load, generate (streaming), benchmark, systemInfo, unload.
 *
 * Streaming: nativeGenerate writes tokens into a shared buffer.
 *   Kotlin polls nativeGetPartialResult() every ~200ms.
 *   nativeStopGeneration() sets a flag to abort early.
 *
 * JNI class: com.medlens.app.inference.MedGemmaInference
 */

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <chrono>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "ggml.h"

#define TAG "MedGemma"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGw(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Global state (single model at a time)
// ---------------------------------------------------------------------------
static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;

// ── Streaming state ─────────────────────────────────────────────────────────
static std::mutex       g_result_mutex;
static std::string      g_partial_result;
static std::atomic<bool> g_stop_flag{false};
static std::atomic<int>  g_tokens_generated{0};
static std::atomic<bool> g_is_generating{false};
static std::atomic<float> g_tok_per_sec{0.0f};

static void log_callback(ggml_log_level level, const char * text, void *) {
    int prio = ANDROID_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_INFO:  prio = ANDROID_LOG_INFO;  break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        default: break;
    }
    __android_log_print(prio, TAG, "%s", text);
}

// ---------------------------------------------------------------------------
// JNI exports — class: com.medlens.app.inference.MedGemmaInference
// ---------------------------------------------------------------------------
extern "C" {

JNIEXPORT void JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeInit(
        JNIEnv * env, jobject, jstring jNativeLibDir) {

    llama_log_set(log_callback, nullptr);

    const char * libdir = env->GetStringUTFChars(jNativeLibDir, nullptr);
    LOGi("Init: native lib dir = %s", libdir);
    ggml_backend_load_all_from_path(libdir);
    env->ReleaseStringUTFChars(jNativeLibDir, libdir);

    llama_backend_init();
    LOGi("Backend initialized");
}

JNIEXPORT jint JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeLoadModel(
        JNIEnv * env, jobject, jstring jModelPath) {

    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model)   { llama_model_free(g_model); g_model = nullptr; }

    const char * path = env->GetStringUTFChars(jModelPath, nullptr);
    LOGi("Loading model: %s", path);
    LOGi("System info: %s", llama_print_system_info());

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap  = false;   // force sequential RAM load (avoids page-fault thrashing)
    model_params.use_mlock = false;
    LOGi("Loading with mmap=false (full RAM load)...");

    auto t_load_start = std::chrono::high_resolution_clock::now();
    g_model = llama_model_load_from_file(path, model_params);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t_load_end - t_load_start).count();

    env->ReleaseStringUTFChars(jModelPath, path);

    if (!g_model) {
        LOGe("Failed to load model");
        return 1;
    }
    LOGi("Model loaded into RAM in %.1f seconds", load_s);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = 512;
    ctx_params.n_batch           = 512;
    ctx_params.n_threads         = 4;
    ctx_params.n_threads_batch   = 4;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        LOGe("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return 2;
    }

    LOGi("Model and context created successfully");
    return 0;
}

JNIEXPORT jstring JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeGenerate(
        JNIEnv * env, jobject, jstring jPrompt, jint maxTokens) {

    if (!g_model || !g_context) {
        return env->NewStringUTF("[Error: model not loaded]");
    }

    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        g_partial_result.clear();
    }
    g_stop_flag.store(false);
    g_tokens_generated.store(0);
    g_is_generating.store(true);
    g_tok_per_sec.store(0.0f);

    const char * prompt_cstr = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(prompt_cstr);
    env->ReleaseStringUTFChars(jPrompt, prompt_cstr);

    std::vector<llama_token> tokens = common_tokenize(g_context, prompt, true);
    LOGi("Prompt tokens: %zu", tokens.size());

    if (tokens.empty()) {
        g_is_generating.store(false);
        return env->NewStringUTF("[Error: empty prompt after tokenization]");
    }

    llama_memory_clear(llama_get_memory(g_context), false);

    const int n_prompt = (int) tokens.size();
    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = llama_batch_init(std::max(n_prompt, 512), 0, 1);

    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(g_context, batch) != 0) {
        llama_batch_free(batch);
        g_is_generating.store(false);
        return env->NewStringUTF("[Error: failed to evaluate prompt]");
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
    double pp_speed = (prompt_ms > 0) ? (n_prompt / (prompt_ms / 1000.0)) : 0;
    LOGi("Prompt eval: %.0f ms (%.1f tok/s)", prompt_ms, pp_speed);

    common_params_sampling sparams;
    sparams.temp = 0.3f;
    common_sampler * sampler = common_sampler_init(g_model, sparams);

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    int pos = n_prompt;
    int n_gen = 0;

    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < maxTokens; i++) {
        if (g_stop_flag.load()) {
            LOGi("Generation stopped by user after %d tokens", n_gen);
            break;
        }

        llama_token new_token = common_sampler_sample(sampler, g_context, -1);
        common_sampler_accept(sampler, new_token, true);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGi("EOS reached after %d tokens", n_gen);
            break;
        }

        std::string piece = common_token_to_piece(g_context, new_token);
        n_gen++;

        {
            std::lock_guard<std::mutex> lock(g_result_mutex);
            g_partial_result += piece;
        }
        g_tokens_generated.store(n_gen);

        if (n_gen % 4 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_gen_start).count();
            if (elapsed > 0) {
                g_tok_per_sec.store((float)(n_gen / elapsed));
            }
        }

        if (n_gen % 16 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_gen_start).count();
            double speed = (elapsed > 0) ? (n_gen / elapsed) : 0;
            LOGi("Generated %d/%d tokens (%.1f tok/s)", n_gen, (int)maxTokens, speed);
        }

        common_batch_clear(batch);
        common_batch_add(batch, new_token, pos++, {0}, true);
        if (llama_decode(g_context, batch) != 0) {
            LOGe("Failed to decode generated token %d", n_gen);
            break;
        }
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double tg_speed = (gen_ms > 0) ? (n_gen / (gen_ms / 1000.0)) : 0;

    LOGi("Generation done: %d tokens in %.0f ms (%.2f tok/s)", n_gen, gen_ms, tg_speed);

    common_sampler_free(sampler);
    llama_batch_free(batch);
    g_is_generating.store(false);

    std::string final_result;
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        final_result = g_partial_result;
    }

    char stats[512];
    snprintf(stats, sizeof(stats),
        "[pp: %d tok in %.0fms = %.1f tok/s | gen: %d tok in %.0fms = %.1f tok/s]",
        n_prompt, prompt_ms, pp_speed, n_gen, gen_ms, tg_speed);

    std::string output = std::string(stats) + "\n\n" + final_result;
    return env->NewStringUTF(output.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeGetPartialResult(
        JNIEnv * env, jobject) {
    std::string text;
    {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        text = g_partial_result;
    }
    int tokens = g_tokens_generated.load();
    float speed = g_tok_per_sec.load();
    bool generating = g_is_generating.load();

    char header[128];
    snprintf(header, sizeof(header), "%d|%.1f|%d|", tokens, speed, generating ? 1 : 0);
    std::string result = std::string(header) + text;
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeStopGeneration(
        JNIEnv *, jobject) {
    LOGi("Stop requested");
    g_stop_flag.store(true);
}

JNIEXPORT jstring JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeBench(
        JNIEnv * env, jobject, jint pp, jint tg, jint reps) {

    if (!g_model) {
        return env->NewStringUTF("Error: model not loaded");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = pp + tg + 64;
    ctx_params.n_batch           = std::max((int)pp, 512);
    ctx_params.n_threads         = 4;
    ctx_params.n_threads_batch   = 4;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * bench_ctx = llama_init_from_model(g_model, ctx_params);
    if (!bench_ctx) {
        return env->NewStringUTF("Error: failed to create bench context");
    }

    llama_batch batch = llama_batch_init(std::max((int)pp, 512), 0, 1);
    double pp_sum = 0.0, tg_sum = 0.0;

    for (int r = 0; r < reps; r++) {
        common_batch_clear(batch);
        for (int i = 0; i < pp; i++) {
            common_batch_add(batch, 0, i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(bench_ctx), false);

        const int64_t t_pp_start = ggml_time_us();
        if (llama_decode(bench_ctx, batch) != 0) break;
        const int64_t t_pp_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(bench_ctx), false);
        const int64_t t_tg_start = ggml_time_us();
        for (int i = 0; i < tg; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, 0, i, {0}, true);
            if (llama_decode(bench_ctx, batch) != 0) break;
        }
        const int64_t t_tg_end = ggml_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;
        pp_sum += (t_pp > 0) ? pp / t_pp : 0;
        tg_sum += (t_tg > 0) ? tg / t_tg : 0;
    }

    llama_batch_free(batch);
    llama_free(bench_ctx);

    pp_sum /= reps;
    tg_sum /= reps;

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "=== MedLens Benchmark ===\n"
        "Prompt Processing: %.2f tok/s (%d tokens)\n"
        "Token Generation:  %.2f tok/s (%d tokens)\n"
        "Repetitions: %d\n\nSystem: %s",
        pp_sum, (int)pp, tg_sum, (int)tg, (int)reps,
        llama_print_system_info());

    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeSystemInfo(
        JNIEnv * env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

JNIEXPORT void JNICALL
Java_com_medlens_app_inference_MedGemmaInference_nativeUnload(
        JNIEnv *, jobject) {
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model)   { llama_model_free(g_model); g_model = nullptr; }
    LOGi("Model unloaded");
}

} // extern "C"
