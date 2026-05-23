package com.edgemind.inference

import android.content.Context
import com.edgemind.data.InferencePath
import com.edgemind.data.InferenceResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp

/**
 * SLM path — Phi-3 Mini (int4 .pte) via ExecuTorch.
 * Targets < 300 ms on Snapdragon 8 Gen 3 NPU.
 * In mock mode, returns realistic demo responses.
 */
class SlmInferenceEngine(context: Context) {

    private val runner = ExecuTorchRunner(context, MODEL_ASSET)
    private val tokenizer = Tokenizer(context)

    suspend fun load() = runner.load()

    suspend fun runInference(
        prompt: String,
        taskType: TaskType = TaskType.AUTO,
        maxNewTokens: Int = 256
    ): InferenceResult = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()

        return@withContext if (runner.isReady) {
            runNativeInference(prompt, taskType, maxNewTokens, start)
        } else {
            runMockInference(prompt, taskType, start)
        }
    }

    private suspend fun runNativeInference(
        prompt: String,
        taskType: TaskType,
        maxNewTokens: Int,
        startMs: Long
    ): InferenceResult {
        val inputIds = tokenizer.encode(buildPrompt(prompt, taskType))
        val logits = runner.forward(inputIds)
        val generatedIds = greedyDecode(logits, inputIds.size, maxNewTokens)
        val text = tokenizer.decode(generatedIds)
        val confidence = extractConfidence(logits, taskType)
        return InferenceResult(
            text = text,
            confidence = confidence,
            latencyMs = System.currentTimeMillis() - startMs,
            path = InferencePath.SLM,
            tokens = generatedIds.size
        )
    }

    private suspend fun runMockInference(
        prompt: String,
        taskType: TaskType,
        startMs: Long
    ): InferenceResult {
        val lowerPrompt = prompt.lowercase()
        return when {
            taskType == TaskType.NER || lowerPrompt.contains("chest pain") || lowerPrompt.contains("shortness of breath") -> {
                simulateLatency(100, 160)
                InferenceResult(
                    text = """{"entities": [{"type": "symptom", "value": "chest pain"}, {"type": "symptom", "value": "shortness of breath"}]}""",
                    confidence = 0.93f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 42
                )
            }
            taskType == TaskType.TRIAGE || lowerPrompt.contains("triage") || lowerPrompt.contains("severity") -> {
                simulateLatency(140, 220)
                // Low confidence triggers RAG in the demo
                InferenceResult(
                    text = "Cardiac or respiratory emergency — confidence insufficient for final triage.",
                    confidence = 0.62f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 18
                )
            }
            taskType == TaskType.TRIAGE_WITH_RAG -> {
                simulateLatency(180, 250)
                InferenceResult(
                    text = "ESI Level 2 — High acuity. Symptoms consistent with ACS or PE. Immediate evaluation required. Do not leave patient unattended.",
                    confidence = 0.91f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.RAG,
                    tokens = 38
                )
            }
            taskType == TaskType.INTENT -> {
                simulateLatency(80, 120)
                InferenceResult(
                    text = """{"intent": "symptom_report", "urgency": "high"}""",
                    confidence = 0.97f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 12
                )
            }
            taskType == TaskType.FORM_EXTRACT -> {
                simulateLatency(120, 180)
                InferenceResult(
                    text = """{"fields": {"chief_complaint": "chest pain and shortness of breath", "onset": "unknown", "severity": "unknown"}}""",
                    confidence = 0.88f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 30
                )
            }
            lowerPrompt.contains("medication") || lowerPrompt.contains("drug") -> {
                simulateLatency(100, 140)
                InferenceResult(
                    text = "Aspirin 325 mg — standard normalised name: Acetylsalicylic acid. No critical interactions found in local DB.",
                    confidence = 0.89f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 24
                )
            }
            else -> {
                simulateLatency(180, 280)
                InferenceResult(
                    text = "I understand your query. Based on the available information, I'll provide a response.",
                    confidence = 0.78f,
                    latencyMs = System.currentTimeMillis() - startMs,
                    path = InferencePath.SLM,
                    tokens = 20
                )
            }
        }
    }

    private suspend fun simulateLatency(minMs: Int, maxMs: Int) {
        val delay = (minMs + Math.random() * (maxMs - minMs)).toLong()
        kotlinx.coroutines.delay(delay)
    }

    private fun buildPrompt(userText: String, taskType: TaskType): String {
        val systemPrompt = when (taskType) {
            TaskType.NER -> "Extract medical entities (symptoms, conditions, medications) from the following text. Return JSON."
            TaskType.TRIAGE -> "Classify the urgency level of the following patient presentation. Return a confidence score and ESI level."
            TaskType.TRIAGE_WITH_RAG -> "Using the clinical context provided, triage the patient. Return ESI level and disposition."
            TaskType.INTENT -> "Classify the intent of the following healthcare query. Return JSON."
            TaskType.FORM_EXTRACT -> "Extract structured fields from the following clinical text. Return JSON."
            TaskType.AUTO -> "You are a helpful clinical assistant."
        }
        return "<|system|>$systemPrompt<|end|><|user|>$userText<|end|><|assistant|>"
    }

    private fun greedyDecode(logits: FloatArray, inputLen: Int, maxNew: Int): LongArray {
        val vocabSize = 32000
        val result = LongArray(maxNew.coerceAtMost(logits.size / vocabSize))
        for (i in result.indices) {
            val offset = (inputLen + i) * vocabSize
            if (offset + vocabSize > logits.size) break
            result[i] = (offset until (offset + vocabSize)).maxByOrNull { logits[it] }?.minus(offset)?.toLong() ?: 0L
        }
        return result
    }

    private fun extractConfidence(logits: FloatArray, taskType: TaskType): Float {
        if (logits.isEmpty()) return 0.5f
        val slice = logits.take(100)
        val maxL = slice.max()
        val sumExp = slice.sumOf { exp((it - maxL).toDouble()) }
        return (exp((slice.max() - maxL).toDouble()) / sumExp).toFloat().coerceIn(0.01f, 0.99f)
    }

    fun close() = runner.close()

    companion object {
        private const val MODEL_ASSET = "phi3_mini.pte"
        const val CONFIDENCE_THRESHOLD = 0.75f
    }
}

enum class TaskType {
    AUTO, NER, TRIAGE, TRIAGE_WITH_RAG, INTENT, FORM_EXTRACT
}
