/**
 * ExecuTorch JNI bridge for EdgeMind.
 *
 * Build requirements:
 *   - ExecuTorch Android AAR placed in app/libs/
 *   - ExecuTorch headers in cpp/include/executorch/
 *   - Uncomment CMakeLists.txt ExecuTorch link lines
 *
 * When the AAR is absent the native library still compiles;
 * ExecuTorchRunner.kt detects the missing symbols and falls
 * back to its Kotlin mock implementation automatically.
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <memory>

#define LOG_TAG "ExecuTorchJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// ── Conditional ExecuTorch include ───────────────────────────────────────
#ifdef EXECUTORCH_AVAILABLE
#include <executorch/runtime/executor/module.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
using namespace executorch::runtime;
using namespace executorch::aten;
#endif

// ── Model handle registry ─────────────────────────────────────────────────
// Maps opaque jlong handle → Module pointer.
// In production use an AtomicLong registry; kept simple here for clarity.

#ifdef EXECUTORCH_AVAILABLE
static std::unordered_map<jlong, std::unique_ptr<Module>> g_modules;
static jlong g_next_handle = 1;
#endif

extern "C" {

/**
 * Load a .pte model from the given absolute path.
 * Returns a non-zero handle on success, 0 on failure.
 */
JNIEXPORT jlong JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeLoad(
        JNIEnv* env, jobject /* this */, jstring path_jstr) {
#ifdef EXECUTORCH_AVAILABLE
    const char* path = env->GetStringUTFChars(path_jstr, nullptr);
    LOGI("Loading model: %s", path);

    auto module = std::make_unique<Module>(path, Module::MmapType::MmapUseMlock);
    auto err = module->load();
    env->ReleaseStringUTFChars(path_jstr, path);

    if (err != Error::Ok) {
        LOGE("Module::load failed: %d", (int)err);
        return 0L;
    }

    jlong handle = g_next_handle++;
    g_modules[handle] = std::move(module);
    LOGI("Model loaded, handle=%lld", (long long)handle);
    return handle;
#else
    LOGW("ExecuTorch not compiled in — nativeLoad returning 0");
    return 0L;
#endif
}

/**
 * Forward pass with LongArray input (token IDs).
 * Returns FloatArray of logits.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeForward(
        JNIEnv* env, jobject /* this */, jlong handle, jlongArray input_ids) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it == g_modules.end()) {
        LOGE("Invalid handle: %lld", (long long)handle);
        return env->NewFloatArray(0);
    }

    jsize len = env->GetArrayLength(input_ids);
    jlong* ids = env->GetLongArrayElements(input_ids, nullptr);

    // Build EValue input tensor [1, seq_len]
    std::vector<int64_t> input_data(ids, ids + len);
    env->ReleaseLongArrayElements(input_ids, ids, JNI_ABORT);

    int64_t shape[2] = {1, (int64_t)len};
    auto tensor = from_blob(input_data.data(), {1, (int64_t)len}, ScalarType::Long);

    auto result = it->second->forward({EValue(tensor)});
    if (!result.ok()) {
        LOGE("forward() failed");
        return env->NewFloatArray(0);
    }

    auto out_tensor = result.get()[0].toTensor();
    auto* data = out_tensor.data_ptr<float>();
    jsize out_size = (jsize)out_tensor.numel();

    jfloatArray jout = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(jout, 0, out_size, data);
    return jout;
#else
    return env->NewFloatArray(0);
#endif
}

/**
 * Forward pass with FloatArray input (embeddings / audio features).
 */
JNIEXPORT jfloatArray JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeForwardFloat(
        JNIEnv* env, jobject /* this */,
        jlong handle, jfloatArray input, jlongArray shape_arr) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it == g_modules.end()) {
        return env->NewFloatArray(0);
    }

    jsize in_len = env->GetArrayLength(input);
    jfloat* in_data = env->GetFloatArrayElements(input, nullptr);

    jsize shape_len = env->GetArrayLength(shape_arr);
    jlong* shape_data = env->GetLongArrayElements(shape_arr, nullptr);
    std::vector<int64_t> shape(shape_data, shape_data + shape_len);
    env->ReleaseLongArrayElements(shape_arr, shape_data, JNI_ABORT);

    auto tensor = from_blob(in_data, shape, ScalarType::Float);
    auto result = it->second->forward({EValue(tensor)});
    env->ReleaseFloatArrayElements(input, in_data, JNI_ABORT);

    if (!result.ok()) {
        return env->NewFloatArray(0);
    }

    auto out_tensor = result.get()[0].toTensor();
    auto* out_data = out_tensor.data_ptr<float>();
    jsize out_size = (jsize)out_tensor.numel();

    jfloatArray jout = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(jout, 0, out_size, out_data);
    return jout;
#else
    return env->NewFloatArray(0);
#endif
}

/**
 * Release the Module and free memory.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeClose(
        JNIEnv* /* env */, jobject /* this */, jlong handle) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it != g_modules.end()) {
        g_modules.erase(it);
        LOGI("Model unloaded, handle=%lld", (long long)handle);
    }
#endif
}

// ── Tokenizer JNI ─────────────────────────────────────────────────────────

JNIEXPORT jlongArray JNICALL
Java_com_edgemind_inference_Tokenizer_nativeEncode(
        JNIEnv* env, jobject /* this */, jstring text, jint max_length) {
    // Placeholder — real impl calls sentencepiece::SentencePieceProcessor
    LOGW("nativeEncode: SentencePiece not compiled in — returning empty");
    return env->NewLongArray(0);
}

JNIEXPORT jstring JNICALL
Java_com_edgemind_inference_Tokenizer_nativeDecode(
        JNIEnv* env, jobject /* this */, jlongArray token_ids) {
    return env->NewStringUTF("");
}

// ── LoRA training JNI ─────────────────────────────────────────────────────

JNIEXPORT jfloat JNICALL
Java_com_edgemind_training_LoraTrainer_nativeTrainLora(
        JNIEnv* env, jobject /* this */,
        jstring model_asset, jstring adapter_output_path,
        jobjectArray samples, jint batch_size,
        jfloat learning_rate, jint max_steps,
        jint lora_rank, jint lora_alpha, jint fragment_size) {
    // Placeholder — real impl calls ExecuTorch training API
    LOGW("nativeTrainLora: ExecuTorch training not compiled in");
    return 0.0f;
}

} // extern "C"
