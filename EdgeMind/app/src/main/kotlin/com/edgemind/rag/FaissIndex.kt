package com.edgemind.rag

import android.util.Log
import com.edgemind.data.RagChunk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * FAISS flat index — JNI bridge with a pure-Kotlin cosine-similarity fallback.
 * The native path uses a FAISS IndexFlatL2 stored as a binary file.
 * The fallback stores vectors in memory and does a linear scan.
 */
class FaissIndex(private val dimension: Int) {

    private var nativeHandle: Long = 0L
    private val nativeAvailable: Boolean

    // Fallback in-memory store
    private val fallbackVectors = mutableListOf<FloatArray>()
    private val fallbackChunks = mutableListOf<RagChunk>()

    init {
        var available = false
        try {
            System.loadLibrary("faiss_jni")
            available = true
            Log.i(TAG, "FAISS native library loaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "FAISS native unavailable — using Kotlin fallback")
        }
        nativeAvailable = available

        if (nativeAvailable) {
            nativeHandle = nativeCreate(dimension, METRIC_L2)
        }
    }

    fun add(embedding: FloatArray, chunk: RagChunk) {
        if (nativeAvailable && nativeHandle != 0L) {
            nativeAdd(nativeHandle, embedding, chunk.id)
        } else {
            fallbackVectors.add(embedding)
            fallbackChunks.add(chunk)
        }
    }

    suspend fun search(queryEmbedding: FloatArray, k: Int = 5): List<Pair<Long, Float>> =
        withContext(Dispatchers.Default) {
            if (nativeAvailable && nativeHandle != 0L) {
                val ids = nativeSearch(nativeHandle, queryEmbedding, k)
                val distances = nativeSearchDistances(nativeHandle, queryEmbedding, k)
                ids.zip(distances.toTypedArray())
            } else {
                fallbackSearch(queryEmbedding, k)
            }
        }

    private fun fallbackSearch(query: FloatArray, k: Int): List<Pair<Long, Float>> {
        return fallbackVectors
            .mapIndexed { idx, vec ->
                val score = cosineSimilarity(query, vec)
                fallbackChunks[idx].id to score
            }
            .sortedByDescending { it.second }
            .take(k)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices.take(b.size)) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = Math.sqrt((normA * normB).toDouble()).toFloat()
        return if (denom < 1e-8f) 0f else dot / denom
    }

    fun save(path: String) {
        if (nativeAvailable && nativeHandle != 0L) nativeSave(nativeHandle, path)
    }

    fun loadFrom(path: String) {
        if (nativeAvailable) {
            if (nativeHandle != 0L) nativeFree(nativeHandle)
            nativeHandle = nativeLoad(path)
        }
    }

    fun free() {
        if (nativeAvailable && nativeHandle != 0L) {
            nativeFree(nativeHandle)
            nativeHandle = 0L
        }
    }

    private external fun nativeCreate(dimension: Int, metric: Int): Long
    private external fun nativeAdd(handle: Long, vector: FloatArray, id: Long)
    private external fun nativeSearch(handle: Long, query: FloatArray, k: Int): LongArray
    private external fun nativeSearchDistances(handle: Long, query: FloatArray, k: Int): FloatArray
    private external fun nativeSave(handle: Long, path: String)
    private external fun nativeLoad(path: String): Long
    private external fun nativeFree(handle: Long)

    companion object {
        private const val TAG = "FaissIndex"
        private const val METRIC_L2 = 0
        private const val METRIC_IP = 1
    }
}
