package com.edgemind.mcp

import android.util.Log
import com.edgemind.data.CapabilityToken
import com.edgemind.data.GuardBlockReason
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.security.AuditJournal
import com.edgemind.security.GuardModel
import com.edgemind.security.IamService
import com.edgemind.security.InsufficientScopeException
import com.edgemind.security.TokenExpiredException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * On-device MCP server — tool registry, routing engine, IAM validation, guardrails.
 * Initialised once at app startup. Thread-safe via coroutine dispatch.
 */
class McpServer(
    private val registry: ToolRegistry,
    private val guardModel: GuardModel,
    private val iamService: IamService,
    private val auditJournal: AuditJournal
) {
    private var initialised = false

    fun markInitialised() { initialised = true }

    suspend fun handleToolCall(
        call: ToolCall,
        token: CapabilityToken,
        promptForGuard: String? = null
    ): ToolResult = withContext(Dispatchers.Default) {
        if (!initialised) return@withContext ToolResult.Error("MCP server not initialised")

        val agentId = token.agentId
        Log.d(TAG, "ToolCall[${call.toolId}] agent=$agentId")

        // ── Step 1: Guardrail check ───────────────────────────────────────
        val textToGuard = promptForGuard ?: call.params["prompt"] ?: ""
        if (textToGuard.isNotBlank()) {
            val guardResult = guardModel.classify(textToGuard)
            if (guardResult.isBlocked) {
                val reason = guardResult.reason ?: GuardBlockReason.JAILBREAK
                auditJournal.logBlocked(agentId, token, reason, guardResult.promptHash)
                Log.w(TAG, "BLOCKED ToolCall[${call.toolId}] reason=${reason.name}")
                return@withContext ToolResult.Blocked(
                    "Request blocked by guardrail: ${reason.name.lowercase().replace('_', ' ')}"
                )
            }
        }

        // ── Step 2: IAM token validation ─────────────────────────────────
        val requiredScope = registry.requiredScope(call.toolId)
            ?: return@withContext ToolResult.Error("Unknown tool: ${call.toolId}")

        try {
            iamService.validateScope(token, requiredScope)
        } catch (e: TokenExpiredException) {
            Log.w(TAG, "Token expired for $agentId")
            return@withContext ToolResult.Unauthorized(requiredScope)
        } catch (e: InsufficientScopeException) {
            Log.w(TAG, "Insufficient scope for $agentId: ${e.message}")
            return@withContext ToolResult.Unauthorized(requiredScope)
        }

        // ── Step 3: Route to tool handler ─────────────────────────────────
        val handler = registry.getHandler(call.toolId)
            ?: return@withContext ToolResult.Error("No handler for tool: ${call.toolId}")

        val ctx = ToolContext(agentId = agentId, sessionId = token.jwt.take(16))

        return@withContext try {
            handler.handle(call, ctx)
        } catch (e: Exception) {
            Log.e(TAG, "Tool handler error: ${e.message}", e)
            ToolResult.Error("Tool execution failed: ${e.message}")
        }
    }

    fun listTools() = registry.listTools()

    companion object {
        private const val TAG = "McpServer"
    }
}
