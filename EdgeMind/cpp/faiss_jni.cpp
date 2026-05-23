/**
 * FAISS JNI bridge for EdgeMind on-device vector search.
 *
 * Build requirements:
 *   - FAISS prebuilt .so for arm64-v8a in app/jniLibs/arm64-v8a/libfaiss.so
 *   - FAISS headers in cpp/include/faiss/
 *   - Uncomment CMakeLists.txt FAISS link lines
 *
 * When FAISS is absent the library still compiles;
 * FaissIndex.kt detects the missing symbols and falls back
 * to its Kotlin cosine-similarity implementation.
 */

#include <jni.h>
#include <android/log.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>

#define LOG_TAG "FaissJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

#ifdef FAISS_AVAILABLE
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

static std::unordered_map<jlong, std::unique_ptr<faiss::IndexFlat>> g_indices;
static jlong g_next_idx_handle = 1;

enum MetricType { METRIC_L2 = 0, METRIC_IP = 1 };
#endif

extern "C" {

/**
 * Create a new FAISS flat index.
 * metric: 0 = L2, 1 = Inner Product (cosine with normalised vectors).
 */
JNIEXPORT jlong JNICALL
Java_com_edgemind_rag_FaissIndex_nativeCreate(
        JNIEnv* /* env */, jobject /* this */, jint dimension, jint metric) {
#ifdef FAISS_AVAILABLE
    faiss::MetricType faiss_metric = (metric == 1) ? faiss::METRIC_INNER_PRODUCT
                                                    : faiss::METRIC_L2;
    auto index = std::make_unique<faiss::IndexFlat>(dimension, faiss_metric);
    jlong handle = g_next_idx_handle++;
    g_indices[handle] = std::move(index);
    LOGI("FAISS index created: dim=%d metric=%d handle=%lld", dimension, metric, (long long)handle);
    return handle;
#else
    LOGW("FAISS not compiled in — nativeCreate returning 0");
    return 0L;
#endif
}

/**
 * Add a single vector with associated ID to the index.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_rag_FaissIndex_nativeAdd(
        JNIEnv* env, jobject /* this */, jlong handle, jfloatArray vector, jlong id) {
#ifdef FAISS_AVAILABLE
    auto it = g_indices.find(handle);
    if (it == g_indices.end()) { LOGE("Invalid index handle"); return; }

    jsize dim = env->GetArrayLength(vector);
    jfloat* data = env->GetFloatArrayElements(vector, nullptr);
    it->second->add_with_ids(1, data, &id);
    env->ReleaseFloatArrayElements(vector, data, JNI_ABORT);
#endif
}

/**
 * Retrieve top-k nearest neighbour IDs for the query vector.
 */
JNIEXPORT jlongArray JNICALL
Java_com_edgemind_rag_FaissIndex_nativeSearch(
        JNIEnv* env, jobject /* this */, jlong handle, jfloatArray query, jint k) {
#ifdef FAISS_AVAILABLE
    auto it = g_indices.find(handle);
    if (it == g_indices.end()) return env->NewLongArray(0);

    jsize dim = env->GetArrayLength(query);
    jfloat* q = env->GetFloatArrayElements(query, nullptr);

    std::vector<faiss::idx_t> labels(k, -1);
    std::vector<float> distances(k, 0.0f);
    it->second->search(1, q, k, distances.data(), labels.data());
    env->ReleaseFloatArrayElements(query, q, JNI_ABORT);

    jlongArray result = env->NewLongArray(k);
    std::vector<jlong> jlabels(labels.begin(), labels.end());
    env->SetLongArrayRegion(result, 0, k, jlabels.data());
    return result;
#else
    return env->NewLongArray(0);
#endif
}

/**
 * Retrieve top-k nearest neighbour distances for the query vector.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_edgemind_rag_FaissIndex_nativeSearchDistances(
        JNIEnv* env, jobject /* this */, jlong handle, jfloatArray query, jint k) {
#ifdef FAISS_AVAILABLE
    auto it = g_indices.find(handle);
    if (it == g_indices.end()) return env->NewFloatArray(0);

    jsize dim = env->GetArrayLength(query);
    jfloat* q = env->GetFloatArrayElements(query, nullptr);

    std::vector<faiss::idx_t> labels(k, -1);
    std::vector<float> distances(k, 0.0f);
    it->second->search(1, q, k, distances.data(), labels.data());
    env->ReleaseFloatArrayElements(query, q, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(k);
    env->SetFloatArrayRegion(result, 0, k, distances.data());
    return result;
#else
    return env->NewFloatArray(0);
#endif
}

/**
 * Persist the index to disk.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_rag_FaissIndex_nativeSave(
        JNIEnv* env, jobject /* this */, jlong handle, jstring path_jstr) {
#ifdef FAISS_AVAILABLE
    auto it = g_indices.find(handle);
    if (it == g_indices.end()) return;
    const char* path = env->GetStringUTFChars(path_jstr, nullptr);
    faiss::write_index(it->second.get(), path);
    env->ReleaseStringUTFChars(path_jstr, path);
    LOGI("Index saved to %s", path);
#endif
}

/**
 * Load a previously saved index.
 */
JNIEXPORT jlong JNICALL
Java_com_edgemind_rag_FaissIndex_nativeLoad(
        JNIEnv* env, jobject /* this */, jstring path_jstr) {
#ifdef FAISS_AVAILABLE
    const char* path = env->GetStringUTFChars(path_jstr, nullptr);
    auto* raw = faiss::read_index(path);
    env->ReleaseStringUTFChars(path_jstr, path);
    if (!raw) { LOGE("Failed to load index"); return 0L; }

    jlong handle = g_next_idx_handle++;
    g_indices[handle] = std::unique_ptr<faiss::IndexFlat>(
        dynamic_cast<faiss::IndexFlat*>(raw)
    );
    LOGI("Index loaded, handle=%lld", (long long)handle);
    return handle;
#else
    return 0L;
#endif
}

/**
 * Free index memory.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_rag_FaissIndex_nativeFree(
        JNIEnv* /* env */, jobject /* this */, jlong handle) {
#ifdef FAISS_AVAILABLE
    g_indices.erase(handle);
    LOGI("Index freed, handle=%lld", (long long)handle);
#endif
}

} // extern "C"
