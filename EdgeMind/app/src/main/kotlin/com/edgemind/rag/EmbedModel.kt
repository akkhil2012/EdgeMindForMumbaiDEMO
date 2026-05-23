package com.edgemind.rag

import android.content.Context
import com.edgemind.inference.ExecuTorchRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/**
 * all-MiniLM-L6-v2 distilled embedding model (22 MB .pte).
 * Produces 384-dimensional sentence embeddings.
 */
class EmbedModel(context: Context) {

    private val runner = ExecuTorchRunner(context, MODEL_ASSET)
    val dimension = 384

    suspend fun load() = runner.load()

    suspend fun embed(text: String): FloatArray = withContext(Dispatchers.Default) {
        if (runner.nativeAvailable) {
            val tokens = tokenize(text)
            val inputShape = longArrayOf(1, tokens.size.toLong())
            val raw = runner.forwardFloat(tokens, inputShape)
            meanPool(raw, tokens.size, dimension).let { normalize(it) }
        } else {
            mockEmbed(text)
        }
    }

    private fun tokenize(text: String): FloatArray {
        val words = text.lowercase().split(Regex("\\s+")).take(128)
        return FloatArray(words.size) { i -> words[i].hashCode().toFloat() / Int.MAX_VALUE }
    }

    private fun meanPool(raw: FloatArray, seqLen: Int, dim: Int): FloatArray {
        val result = FloatArray(dim)
        if (raw.size < seqLen * dim) return result
        for (d in 0 until dim) {
            var sum = 0f
            for (t in 0 until seqLen) result[d] += raw[t * dim + d]
            result[d] = sum / seqLen
        }
        return result
    }

    private fun normalize(v: FloatArray): FloatArray {
        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
        return FloatArray(v.size) { v[it] / norm }
    }

    // Deterministic mock embedding: hash-based, normalized
    private fun mockEmbed(text: String): FloatArray {
        val seed = text.hashCode()
        val random = java.util.Random(seed.toLong())
        val vec = FloatArray(dimension) { random.nextGaussian().toFloat() }
        return normalize(vec)
    }

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot // vectors are already normalized
    }

    fun close() = runner.close()

    companion object {
        private const val MODEL_ASSET = "minilm_embed.pte"
    }
}
