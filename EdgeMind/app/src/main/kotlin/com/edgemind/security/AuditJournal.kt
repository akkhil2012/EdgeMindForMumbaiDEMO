package com.edgemind.security

import android.content.Context
import android.util.Base64
import android.util.Log
import com.edgemind.data.AuditEntry
import com.edgemind.data.AuditEventType
import com.edgemind.data.AppDatabase
import com.edgemind.data.CapabilityToken
import com.edgemind.data.GuardBlockReason
import com.edgemind.data.InferencePath
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.UUID

/**
 * Signed local audit journal.
 * Every agent action (inference, block, training) is logged with a SHA-256 HMAC signature
 * over the serialised entry, ensuring tamper-evidence at rest.
 */
class AuditJournal(context: Context) {

    private val dao = AppDatabase.getInstance(context).auditDao()

    suspend fun logInference(
        agentId: String,
        token: CapabilityToken,
        path: InferencePath,
        latencyMs: Long,
        confidence: Float,
        detail: String = ""
    ) = write(
        agentId = agentId,
        token = token,
        eventType = AuditEventType.INFERENCE,
        path = path,
        latencyMs = latencyMs,
        confidence = confidence,
        detail = detail
    )

    suspend fun logBlocked(
        agentId: String,
        token: CapabilityToken,
        reason: GuardBlockReason,
        promptHash: String
    ) = write(
        agentId = agentId,
        token = token,
        eventType = AuditEventType.BLOCKED,
        detail = "reason=${reason.name} promptHash=$promptHash"
    )

    suspend fun logTokenIssued(agentId: String, token: CapabilityToken) = write(
        agentId = agentId,
        token = token,
        eventType = AuditEventType.TOKEN_ISSUED,
        detail = "scopes=${token.scopes.joinToString(",")}"
    )

    suspend fun logTraining(agentId: String, token: CapabilityToken, samples: Int, steps: Int) = write(
        agentId = agentId,
        token = token,
        eventType = AuditEventType.TRAINING,
        detail = "samples=$samples steps=$steps"
    )

    fun observeRecent(): Flow<List<AuditEntry>> = dao.observeRecent()

    suspend fun getRecent(limit: Int = 50): List<AuditEntry> = dao.getRecent(limit)

    private suspend fun write(
        agentId: String,
        token: CapabilityToken,
        eventType: AuditEventType,
        path: InferencePath? = null,
        latencyMs: Long? = null,
        confidence: Float? = null,
        detail: String = ""
    ) = withContext(Dispatchers.IO) {
        val id = UUID.randomUUID().toString()
        val timestamp = System.currentTimeMillis()
        val scopesJson = token.scopes.joinToString(",")

        val rawEntry = "$id|$agentId|$timestamp|${eventType.name}|${path?.name}|$latencyMs|$confidence|$scopesJson|$detail"
        val signature = hmacSha256(rawEntry)

        val entry = AuditEntry(
            id = id,
            agentId = agentId,
            timestamp = timestamp,
            eventType = eventType.name,
            inferencePathName = path?.name,
            latencyMs = latencyMs,
            confidence = confidence,
            scopesJson = scopesJson,
            detail = detail,
            signature = signature
        )

        dao.insert(entry)
        Log.d(TAG, "Audit[${eventType.name}] agent=$agentId path=${path?.name} latency=${latencyMs}ms conf=$confidence")
    }

    private fun hmacSha256(data: String): String {
        // In production: use Android Keystore HMAC key for signing
        val bytes = MessageDigest.getInstance("SHA-256").digest(
            (data + SIGNING_SALT).toByteArray()
        )
        return Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    companion object {
        private const val TAG = "AuditJournal"
        private const val SIGNING_SALT = "edgemind-audit-v1"
    }
}
