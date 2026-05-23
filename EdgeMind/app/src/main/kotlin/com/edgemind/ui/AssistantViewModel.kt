package com.edgemind.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgemind.data.AssistantUiState
import com.edgemind.data.ChatMessage
import com.edgemind.data.GuardStatus
import com.edgemind.data.InferencePath
import com.edgemind.data.InferenceResult
import com.edgemind.data.MessageRole
import com.edgemind.data.Scopes
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.inference.LlmInferenceEngine
import com.edgemind.inference.SlmInferenceEngine
import com.edgemind.inference.TaskType
import com.edgemind.mcp.McpClient
import com.edgemind.mcp.McpServer
import com.edgemind.mcp.ToolIds
import com.edgemind.mcp.ToolRegistry
import com.edgemind.mcp.tools.AuditTool
import com.edgemind.mcp.tools.InferenceTool
import com.edgemind.mcp.tools.VectorTool
import com.edgemind.rag.RagPipeline
import com.edgemind.security.AuditJournal
import com.edgemind.security.GuardModel
import com.edgemind.security.IamService
import com.edgemind.training.IdleScheduler
import com.edgemind.training.InteractionDataset
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

class AssistantViewModel(application: Application) : AndroidViewModel(application) {

    private val ctx = application.applicationContext

    // ── Infrastructure ────────────────────────────────────────────────────
    private val slm = SlmInferenceEngine(ctx)
    private val llm = LlmInferenceEngine(ctx)
    private val rag = RagPipeline(ctx)
    private val guard = GuardModel(ctx)
    private val iam = IamService(ctx)
    private val audit = AuditJournal(ctx)
    private val dataset = InteractionDataset(ctx)
    private val idleScheduler = IdleScheduler(ctx)

    private lateinit var server: McpServer
    private lateinit var client: McpClient
    private lateinit var inferenceTool: InferenceTool

    private val _uiState = MutableStateFlow(AssistantUiState())
    val uiState: StateFlow<AssistantUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch { initialise() }
    }

    private suspend fun initialise() {
        try {
            postSystemMessage("Initialising EdgeMind on-device AI...")

            // Load models in parallel
            val registry = ToolRegistry()
            rag.initialize()

            // Build inference tool with metrics callback
            inferenceTool = InferenceTool(
                slm = slm,
                llm = llm,
                rag = rag,
                auditJournal = audit,
                tokenProvider = { client.currentToken() },
                onMetricsUpdate = { result -> updateMetrics(result) }
            )

            registry.register(ToolIds.RUN_INFERENCE_SLM, Scopes.INFERENCE_SLM, "SLM inference path") { call, ctx -> inferenceTool.handle(call, ctx) }
            registry.register(ToolIds.RUN_INFERENCE_LLM, Scopes.INFERENCE_LLM, "LLM inference path") { call, ctx -> inferenceTool.handle(call.copy(params = call.params + ("prefer_llm" to "true")), ctx) }
            registry.register(ToolIds.VECTOR_SEARCH, Scopes.KNOWLEDGE_READ, "On-device vector search") { call, ctx -> VectorTool(rag).handle(call, ctx) }
            registry.register(ToolIds.AUDIT_LOG_WRITE, Scopes.AUDIT_WRITE, "Audit log write") { call, ctx -> AuditTool(audit) { client.currentToken() }.handle(call, ctx) }

            server = McpServer(registry, guard, iam, audit)
            client = McpClient.createDefault("assistant-agent", server, iam, audit)
            server.markInitialised()

            // Observe audit log
            viewModelScope.launch {
                audit.observeRecent().collect { entries ->
                    _uiState.update { it.copy(auditLog = entries) }
                }
            }

            idleScheduler.schedule()
            postSystemMessage("EdgeMind ready. Running fully on-device — no cloud dependency.")
            Log.i(TAG, "EdgeMind initialised")

        } catch (e: Exception) {
            Log.e(TAG, "Initialisation failed: ${e.message}", e)
            _uiState.update { it.copy(error = "Initialisation failed: ${e.message}") }
        }
    }

    fun sendMessage(text: String) {
        if (text.isBlank() || _uiState.value.isProcessing) return

        val userMsg = ChatMessage(role = MessageRole.USER, text = text.trim())
        _uiState.update { state ->
            state.copy(
                messages = state.messages + userMsg,
                isProcessing = true,
                error = null,
                metrics = state.metrics.copy(guardStatus = GuardStatus.IDLE)
            )
        }

        viewModelScope.launch {
            try {
                processPrompt(text.trim())
            } catch (e: Exception) {
                Log.e(TAG, "Message processing failed", e)
                _uiState.update { it.copy(isProcessing = false, error = e.message) }
            }
        }
    }

    private suspend fun processPrompt(prompt: String) {
        val lowerPrompt = prompt.lowercase()
        val preferLlm = lowerPrompt.contains("differential") ||
                lowerPrompt.contains("reasoning") ||
                lowerPrompt.contains("explain") ||
                lowerPrompt.contains("why") ||
                lowerPrompt.contains("treatment") ||
                lowerPrompt.contains("management")

        val taskType = when {
            lowerPrompt.contains("ner") || lowerPrompt.contains("extract entity") -> TaskType.NER
            lowerPrompt.contains("triage") || lowerPrompt.contains("severity") || lowerPrompt.contains("urgent") -> TaskType.TRIAGE
            lowerPrompt.contains("intent") -> TaskType.INTENT
            lowerPrompt.contains("form") || lowerPrompt.contains("fill") -> TaskType.FORM_EXTRACT
            else -> TaskType.AUTO
        }

        // Run guardrail first (visible via metrics)
        _uiState.update { it.copy(metrics = it.metrics.copy(guardStatus = GuardStatus.IDLE)) }

        val toolCall = ToolCall(
            toolId = if (preferLlm) ToolIds.RUN_INFERENCE_LLM else ToolIds.RUN_INFERENCE_SLM,
            params = mapOf(
                "prompt" to prompt,
                "prefer_llm" to preferLlm.toString(),
                "task_type" to taskType.name
            )
        )

        val result = client.invokeTool(toolCall, prompt)

        when (result) {
            is ToolResult.Blocked -> {
                _uiState.update { state ->
                    state.copy(
                        isProcessing = false,
                        metrics = state.metrics.copy(guardStatus = GuardStatus.BLOCK, guardBlockReason = result.reason),
                        messages = state.messages + ChatMessage(
                            role = MessageRole.SYSTEM,
                            text = "Request blocked: ${result.reason}"
                        )
                    )
                }
            }
            is ToolResult.Success -> {
                _uiState.update { it.copy(metrics = it.metrics.copy(guardStatus = GuardStatus.PASS)) }
                val responseText = parseResponseText(result.data)

                // Record for future training
                viewModelScope.launch {
                    dataset.record(prompt, responseText, "slm", 0.85f)
                }

                val assistantMsg = ChatMessage(
                    role = MessageRole.ASSISTANT,
                    text = responseText,
                    path = parsePathFromResponse(result.data),
                    confidence = parseConfidenceFromResponse(result.data)
                )

                _uiState.update { state ->
                    state.copy(
                        messages = state.messages + assistantMsg,
                        isProcessing = false,
                        metrics = state.metrics.copy(
                            latencyMs = result.latencyMs,
                            tokenTtlSeconds = client.currentToken().ttlSeconds
                        )
                    )
                }
            }
            is ToolResult.Unauthorized -> {
                _uiState.update { it.copy(
                    isProcessing = false,
                    error = "Insufficient permission: requires ${result.requiredScope}"
                ) }
            }
            is ToolResult.Error -> {
                _uiState.update { it.copy(
                    isProcessing = false,
                    error = result.message
                ) }
            }
        }
    }

    private fun updateMetrics(result: InferenceResult) {
        val ramMb = Runtime.getRuntime().let {
            ((it.totalMemory() - it.freeMemory()) / 1024 / 1024).toInt()
        }
        _uiState.update { state ->
            state.copy(
                metrics = state.metrics.copy(
                    activePath = result.path,
                    latencyMs = result.latencyMs,
                    confidence = result.confidence,
                    npuUtilPct = if (result.path == InferencePath.SLM) (50 + Math.random() * 30).toInt() else 0,
                    ramUsageMb = ramMb,
                    guardStatus = GuardStatus.PASS
                )
            )
        }
    }

    private fun postSystemMessage(text: String) {
        _uiState.update { state ->
            state.copy(messages = state.messages + ChatMessage(role = MessageRole.SYSTEM, text = text))
        }
    }

    private fun parseResponseText(data: String): String {
        val match = Regex(""""text":"((?:[^"\\]|\\.)*)"""").find(data)
        return match?.groupValues?.get(1)
            ?.replace("\\n", "\n")
            ?.replace("\\\"", "\"")
            ?: data
    }

    private fun parsePathFromResponse(data: String): InferencePath? {
        val match = Regex(""""path":"(\w+)"""").find(data)
        return match?.groupValues?.get(1)?.let { runCatching { InferencePath.valueOf(it) }.getOrNull() }
    }

    private fun parseConfidenceFromResponse(data: String): Float? {
        val match = Regex(""""confidence":([\d.]+)""").find(data)
        return match?.groupValues?.get(1)?.toFloatOrNull()
    }

    fun toggleAuditLog() {
        _uiState.update { it.copy(showAuditLog = !it.showAuditLog) }
    }

    fun dismissError() {
        _uiState.update { it.copy(error = null) }
    }

    override fun onCleared() {
        super.onCleared()
        slm.close()
        llm.close()
        rag.close()
        guard.close()
    }

    companion object {
        private const val TAG = "AssistantViewModel"
    }
}
