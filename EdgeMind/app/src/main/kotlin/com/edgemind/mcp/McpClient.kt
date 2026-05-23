package com.edgemind.mcp

import com.edgemind.data.AgentRole
import com.edgemind.data.CapabilityToken
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.security.AuditJournal
import com.edgemind.security.IamService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Client-facing MCP tool invocation API.
 * Manages the agent session token and delegates to McpServer.
 * Each McpClient instance represents one agent session.
 */
class McpClient(
    private val agentId: String,
    private val role: AgentRole,
    private val server: McpServer,
    private val iamService: IamService,
    private val auditJournal: AuditJournal
) {
    private var token: CapabilityToken = iamService.issueToken(agentId, role)

    init {
        @Suppress("OPT_IN_USAGE")
        GlobalScope.launch { auditJournal.logTokenIssued(agentId, token) }
    }

    private fun ensureFreshToken(): CapabilityToken {
        if (token.isExpired) {
            token = iamService.issueToken(agentId, role)
        }
        return token
    }

    suspend fun invokeTool(call: ToolCall, prompt: String? = null): ToolResult =
        withContext(Dispatchers.Default) {
            server.handleToolCall(call, ensureFreshToken(), prompt)
        }

    suspend fun runInference(
        prompt: String,
        preferLlm: Boolean = false
    ): ToolResult = invokeTool(
        call = ToolCall(
            toolId = if (preferLlm) ToolIds.RUN_INFERENCE_LLM else ToolIds.RUN_INFERENCE_SLM,
            params = mapOf("prompt" to prompt, "prefer_llm" to preferLlm.toString())
        ),
        prompt = prompt
    )

    suspend fun vectorSearch(query: String, k: Int = 5): ToolResult = invokeTool(
        call = ToolCall(
            toolId = ToolIds.VECTOR_SEARCH,
            params = mapOf("query" to query, "k" to k.toString())
        ),
        prompt = query
    )

    fun currentToken(): CapabilityToken = token

    companion object {
        fun createDefault(agentId: String, server: McpServer, iam: IamService, audit: AuditJournal) =
            McpClient(agentId, AgentRole.ELEVATED, server, iam, audit)
    }
}
