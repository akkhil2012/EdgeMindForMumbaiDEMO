package com.edgemind.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * JNI bridge to the ExecuTorch runtime.
 * Falls back to a mock implementation when native libraries are absent
 * (development / CI without .pte model files).
 */
class ExecuTorchRunner(private val context: Context, private val modelAsset: String) {

    private var nativeHandle: Long = 0L
    val nativeAvailable: Boolean

    /** True only when the native library AND a model are loaded and ready. */
    val isReady: Boolean get() = nativeAvailable && nativeHandle != 0L

    init {
        var available = false
        try {
            System.loadLibrary("executorch_jni")
            available = true
            Log.i(TAG, "ExecuTorch native library loaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Native library unavailable — running in mock mode")
        }
        nativeAvailable = available
    }

    suspend fun load() = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            try {
                val assetPath = copyAssetToCache(modelAsset)
                nativeHandle = nativeLoad(assetPath)
                if (nativeHandle == 0L) {
                    Log.w(TAG, "Native load returned null handle for $modelAsset — mock mode active")
                }
            } catch (e: java.io.FileNotFoundException) {
                Log.w(TAG, "Model asset not found: $modelAsset — mock mode active")
            }
        }
    }

    suspend fun forward(inputIds: LongArray): FloatArray = withContext(Dispatchers.Default) {
        if (nativeAvailable && nativeHandle != 0L) {
            nativeForward(nativeHandle, inputIds)
        } else {
            mockForward(inputIds)
        }
    }

    suspend fun forwardFloat(input: FloatArray, shape: LongArray): FloatArray =
        withContext(Dispatchers.Default) {
            if (nativeAvailable && nativeHandle != 0L) {
                nativeForwardFloat(nativeHandle, input, shape)
            } else {
                mockForwardFloat(input, shape)
            }
        }

    fun close() {
        if (nativeAvailable && nativeHandle != 0L) {
            nativeClose(nativeHandle)
            nativeHandle = 0L
        }
    }

    private fun copyAssetToCache(asset: String): String {
        val outFile = context.cacheDir.resolve(asset)
        if (!outFile.exists()) {
            context.assets.open("models/$asset").use { input ->
                outFile.outputStream().use { input.copyTo(it) }
            }
        }
        return outFile.absolutePath
    }

    // ── Mock implementations ──────────────────────────────────────────────

    private suspend fun mockForward(inputIds: LongArray): FloatArray {
        delay(20)
        return FloatArray(inputIds.size * 32000) { Math.random().toFloat() }
    }

    private suspend fun mockForwardFloat(input: FloatArray, shape: LongArray): FloatArray {
        delay(15)
        val outSize = shape.fold(1L) { acc, d -> acc * d }.toInt()
        return FloatArray(outSize) { Math.random().toFloat() }
    }

    // ── JNI declarations ─────────────────────────────────────────────────

    private external fun nativeLoad(path: String): Long
    private external fun nativeForward(handle: Long, inputIds: LongArray): FloatArray
    private external fun nativeForwardFloat(handle: Long, input: FloatArray, shape: LongArray): FloatArray
    private external fun nativeClose(handle: Long)

    companion object {
        private const val TAG = "ExecuTorchRunner"
    }
}
