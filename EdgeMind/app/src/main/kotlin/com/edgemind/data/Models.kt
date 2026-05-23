package com.edgemind.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlinx.serialization.Serializable

// ── Inference ─────────────────────────────────────────────────────────────

enum class InferencePath { SLM, RAG, LLM }

data class InferenceResult(
    val text: String,
    val confidence: Float,
    val latencyMs: Long,
    val path: InferencePath,
    val tokens: Int = 0,
    val ragChunks: List<RagChunk> = emptyList()
)

data class RagChunk(
    val id: Long,
    val text: String,
    val score: Float
)

// ── Guard ─────────────────────────────────────────────────────────────────

data class GuardResult(
    val isBlocked: Boolean,
    val reason: GuardBlockReason? = null,
    val promptHash: String = ""
)

enum class GuardBlockReason {
    JAILBREAK, PII, MEDICAL_SAFETY, OFF_TOPIC, INSTRUCTION_OVERRIDE
}

// ── IAM ───────────────────────────────────────────────────────────────────

enum class AgentRole {
    DEFAULT,    // SLM + knowledge read
    ELEVATED,   // + LLM access
    SYSTEM      // + audit write + sensor
}

@Serializable
data class CapabilityToken(
    val jwt: String,
    val agentId: String,
    val scopes: Set<String>,
    val expiresAt: Long
) {
    val isExpired get() = System.currentTimeMillis() > expiresAt
    val ttlSeconds get() = ((expiresAt - System.currentTimeMillis()) / 1000).coerceAtLeast(0)
}

// ── MCP Tools ─────────────────────────────────────────────────────────────

data class CapabilityManifest(
    val toolId: String,
    val requiredScope: String,
    val description: String
)

@Serializable
data class ToolCall(
    val toolId: String,
    val params: Map<String, String> = emptyMap()
)

sealed class ToolResult {
    data class Success(val data: String, val latencyMs: Long) : ToolResult()
    data class Blocked(val reason: String) : ToolResult()
    data class Unauthorized(val requiredScope: String) : ToolResult()
    data class Error(val message: String) : ToolResult()
}

// ── Audit ─────────────────────────────────────────────────────────────────

enum class AuditEventType { INFERENCE, BLOCKED, TOOL_CALL, TRAINING, TOKEN_ISSUED }

@Entity(tableName = "audit_entries")
data class AuditEntry(
    @PrimaryKey val id: String,
    val agentId: String,
    val timestamp: Long,
    val eventType: String,
    val inferencePathName: String?,
    val latencyMs: Long?,
    val confidence: Float?,
    val scopesJson: String,
    val detail: String,
    val signature: String
)

// ── Interactions (training data) ──────────────────────────────────────────

@Entity(tableName = "interactions")
data class InteractionRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val prompt: String,
    val response: String,
    val pathName: String,
    val confidence: Float,
    val accepted: Boolean = true,
    val anonymised: Boolean = false
)

// ── UI State ──────────────────────────────────────────────────────────────

data class MetricsState(
    val activePath: InferencePath? = null,
    val latencyMs: Long = 0,
    val confidence: Float = 0f,
    val npuUtilPct: Int = 0,
    val ramUsageMb: Int = 0,
    val tokenTtlSeconds: Long = 0,
    val guardStatus: GuardStatus = GuardStatus.IDLE,
    val guardBlockReason: String? = null
)

enum class GuardStatus { IDLE, PASS, BLOCK }

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val role: MessageRole,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val path: InferencePath? = null,
    val confidence: Float? = null
)

enum class MessageRole { USER, ASSISTANT, SYSTEM }

data class AssistantUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isProcessing: Boolean = false,
    val metrics: MetricsState = MetricsState(),
    val auditLog: List<AuditEntry> = emptyList(),
    val showAuditLog: Boolean = false,
    val error: String? = null
)

// ── Routing Decision ─────────────────────────────────────────────────────

data class RoutingDecision(
    val path: InferencePath,
    val agentId: String,
    val token: CapabilityToken
)

// ── Scopes ────────────────────────────────────────────────────────────────

object Scopes {
    const val INFERENCE_SLM = "INFERENCE_SLM"
    const val INFERENCE_LLM = "INFERENCE_LLM"
    const val KNOWLEDGE_READ = "KNOWLEDGE_READ"
    const val SENSOR_MIC = "SENSOR_MIC"
    const val AUDIT_WRITE = "AUDIT_WRITE"

    val DEFAULT_AGENT = setOf(INFERENCE_SLM, KNOWLEDGE_READ)
    val ELEVATED_AGENT = setOf(INFERENCE_SLM, INFERENCE_LLM, KNOWLEDGE_READ)
    val SYSTEM_AGENT = setOf(INFERENCE_SLM, INFERENCE_LLM, KNOWLEDGE_READ, SENSOR_MIC, AUDIT_WRITE)

    fun forRole(role: AgentRole): Set<String> = when (role) {
        AgentRole.DEFAULT -> DEFAULT_AGENT
        AgentRole.ELEVATED -> ELEVATED_AGENT
        AgentRole.SYSTEM -> SYSTEM_AGENT
    }
}
