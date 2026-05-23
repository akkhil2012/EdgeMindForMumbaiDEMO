package com.edgemind.mcp.tools

import android.util.Log
import com.edgemind.data.AuditEntry
import com.edgemind.data.CapabilityToken
import com.edgemind.data.InferencePath
import com.edgemind.data.InferenceResult
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.inference.SlmInferenceEngine
import com.edgemind.inference.LlmInferenceEngine
import com.edgemind.inference.TaskType
import com.edgemind.mcp.ToolContext
import com.edgemind.mcp.ToolHandler
import com.edgemind.rag.RagPipeline
import com.edgemind.security.AuditJournal
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * run_inference tool handler.
 * Implements the three-path routing decision logic:
 * SLM → (if confidence < 0.75) RAG fallback → (if still < 0.75) LLM escalation.
 * Every decision is audited.
 */
class InferenceTool(
    private val slm: SlmInferenceEngine,
    private val llm: LlmInferenceEngine,
    private val rag: RagPipeline,
    private val auditJournal: AuditJournal,
    private val tokenProvider: () -> CapabilityToken,
    private val onMetricsUpdate: (InferenceResult) -> Unit = {}
) : ToolHandler {

    override suspend fun handle(call: ToolCall, ctx: ToolContext): ToolResult {
        val prompt = call.params["prompt"] ?: return ToolResult.Error("Missing 'prompt' parameter")
        val preferLlm = call.params["prefer_llm"]?.toBooleanStrictOrNull() ?: false
        val taskType = call.params["task_type"]?.let { runCatching { TaskType.valueOf(it) }.getOrNull() } ?: TaskType.AUTO

        val token = tokenProvider()
        val result = route(prompt, taskType, preferLlm, ctx.agentId, token)

        onMetricsUpdate(result)
        return ToolResult.Success(
            data = buildResponse(result),
            latencyMs = result.latencyMs
        )
    }

    suspend fun route(
        prompt: String,
        taskType: TaskType = TaskType.AUTO,
        forceLlm: Boolean = false,
        agentId: String = "assistant",
        token: CapabilityToken
    ): InferenceResult {
        // Direct LLM for complex reasoning
        if (forceLlm) {
            Log.d(TAG, "Forced LLM escalation")
            val result = llm.runInference(prompt)
            auditJournal.logInference(agentId, token, InferencePath.LLM, result.latencyMs, result.confidence, "forced-llm")
            return result
        }

        // ── Step 4: SLM inference ─────────────────────────────────────────
        val slmResult = slm.runInference(prompt, taskType)
        Log.d(TAG, "SLM: conf=${slmResult.confidence} latency=${slmResult.latencyMs}ms")

        if (slmResult.confidence >= SlmInferenceEngine.CONFIDENCE_THRESHOLD) {
            auditJournal.logInference(agentId, token, InferencePath.SLM, slmResult.latencyMs, slmResult.confidence)
            return slmResult
        }

        // ── Step 5: RAG fallback ──────────────────────────────────────────
        Log.d(TAG, "SLM confidence ${slmResult.confidence} < threshold — triggering RAG")
        val chunks = rag.retrieve(prompt)
        val augmentedPrompt = rag.buildAugmentedPrompt(prompt, chunks)

        val ragTaskType = if (taskType == TaskType.TRIAGE) TaskType.TRIAGE_WITH_RAG else taskType
        val ragResult = slm.runInference(augmentedPrompt, ragTaskType).copy(
            path = InferencePath.RAG,
            ragChunks = chunks
        )
        Log.d(TAG, "RAG: conf=${ragResult.confidence} latency=${ragResult.latencyMs}ms")

        if (ragResult.confidence >= SlmInferenceEngine.CONFIDENCE_THRESHOLD) {
            auditJournal.logInference(agentId, token, InferencePath.RAG, ragResult.latencyMs, ragResult.confidence,
                "rag-chunks=${chunks.size}")
            return ragResult
        }

        // ── Step 6: LLM escalation ────────────────────────────────────────
        Log.d(TAG, "RAG confidence ${ragResult.confidence} still below threshold — LLM escalation")
        val llmResult = llm.runInference(prompt, augmentedContext = chunks.joinToString("\n") { it.text })
        auditJournal.logInference(agentId, token, InferencePath.LLM, llmResult.latencyMs, llmResult.confidence,
            "escalated-from-rag")
        return llmResult
    }

    private fun buildResponse(result: InferenceResult): String {
        val chunks = if (result.ragChunks.isNotEmpty()) {
            "\nRAG chunks retrieved: ${result.ragChunks.size}"
        } else ""
        return """{"text":${Json.encodeToString(result.text)},"confidence":${result.confidence},"path":"${result.path.name}","latency_ms":${result.latencyMs}$chunks}"""
    }

    companion object {
        private const val TAG = "InferenceTool"
    }
}
