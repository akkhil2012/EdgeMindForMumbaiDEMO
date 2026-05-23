package com.edgemind.mcp.tools

import com.edgemind.data.AuditEventType
import com.edgemind.data.CapabilityToken
import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.mcp.ToolContext
import com.edgemind.mcp.ToolHandler
import com.edgemind.security.AuditJournal

/**
 * audit_log:write tool handler.
 * Restricted to AUDIT_WRITE scope — system agent only.
 */
class AuditTool(
    private val auditJournal: AuditJournal,
    private val tokenProvider: () -> CapabilityToken
) : ToolHandler {

    override suspend fun handle(call: ToolCall, ctx: ToolContext): ToolResult {
        val detail = call.params["detail"] ?: ""
        val start = System.currentTimeMillis()

        val token = tokenProvider()
        val entries = auditJournal.getRecent(50)

        val json = buildString {
            append("""{"entries":[""")
            entries.forEachIndexed { i, entry ->
                if (i > 0) append(",")
                append("""{"id":"${entry.id}","agent":"${entry.agentId}","ts":${entry.timestamp},"type":"${entry.eventType}","path":"${entry.inferencePathName}","latency":${entry.latencyMs},"confidence":${entry.confidence},"sig":"${entry.signature.take(12)}..."}""")
            }
            append("""],"count":${entries.size}}""")
        }

        return ToolResult.Success(data = json, latencyMs = System.currentTimeMillis() - start)
    }
}
