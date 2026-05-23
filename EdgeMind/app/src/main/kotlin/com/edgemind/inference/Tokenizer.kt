package com.edgemind.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * SentencePiece tokenizer wrapper.
 * Uses a native JNI bridge when available; falls back to whitespace tokenization for mock mode.
 */
class Tokenizer(private val context: Context) {

    private val nativeAvailable: Boolean

    init {
        var available = false
        try {
            System.loadLibrary("executorch_jni")
            available = true
        } catch (_: UnsatisfiedLinkError) {}
        nativeAvailable = available
    }

    suspend fun encode(text: String, maxLength: Int = 2048): LongArray =
        withContext(Dispatchers.Default) {
            if (nativeAvailable) {
                nativeEncode(text, maxLength)
            } else {
                mockEncode(text, maxLength)
            }
        }

    suspend fun decode(tokenIds: LongArray): String =
        withContext(Dispatchers.Default) {
            if (nativeAvailable) {
                nativeDecode(tokenIds)
            } else {
                mockDecode(tokenIds)
            }
        }

    private fun mockEncode(text: String, maxLength: Int): LongArray {
        val tokens = text.lowercase().split(Regex("\\s+"))
            .map { it.hashCode().toLong().and(0x7FFF) }
            .take(maxLength)
        return LongArray(tokens.size) { tokens[it] }
    }

    private fun mockDecode(ids: LongArray): String = ids.joinToString(" ") { "<$it>" }

    private external fun nativeEncode(text: String, maxLength: Int): LongArray
    private external fun nativeDecode(tokenIds: LongArray): String

    companion object {
        private const val TAG = "Tokenizer"
    }
}
