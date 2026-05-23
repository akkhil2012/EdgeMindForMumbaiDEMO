package com.edgemind.mcp

import com.edgemind.data.CapabilityManifest
import com.edgemind.data.Scopes
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult

/**
 * Typed tool registry. Each tool carries a CapabilityManifest declaring required permission scope.
 * Tools are registered at MCP server startup and immutable thereafter.
 */
class ToolRegistry {

    private val tools = mutableMapOf<String, RegisteredTool>()

    fun register(
        toolId: String,
        requiredScope: String,
        description: String,
        handler: ToolHandler
    ) {
        tools[toolId] = RegisteredTool(
            manifest = CapabilityManifest(toolId, requiredScope, description),
            handler = handler
        )
    }

    fun getManifest(toolId: String): CapabilityManifest? = tools[toolId]?.manifest

    fun getHandler(toolId: String): ToolHandler? = tools[toolId]?.handler

    fun listTools(): List<CapabilityManifest> = tools.values.map { it.manifest }

    fun requiredScope(toolId: String): String? = tools[toolId]?.manifest?.requiredScope

    data class RegisteredTool(
        val manifest: CapabilityManifest,
        val handler: ToolHandler
    )
}

fun interface ToolHandler {
    suspend fun handle(call: ToolCall, ctx: ToolContext): ToolResult
}

data class ToolContext(
    val agentId: String,
    val sessionId: String,
    val metadata: Map<String, String> = emptyMap()
)

// ── Well-known tool IDs ───────────────────────────────────────────────────

object ToolIds {
    const val RUN_INFERENCE_SLM = "run_inference:slm"
    const val RUN_INFERENCE_LLM = "run_inference:llm"
    const val VECTOR_SEARCH = "vector_search"
    const val DEVICE_SENSOR_MIC = "device_sensor:mic"
    const val AUDIT_LOG_WRITE = "audit_log:write"
}
