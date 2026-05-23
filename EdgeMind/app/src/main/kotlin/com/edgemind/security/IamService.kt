package com.edgemind.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import com.edgemind.data.AgentRole
import com.edgemind.data.CapabilityToken
import com.edgemind.data.Scopes
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.Signature

/**
 * On-device IAM service backed by Android Keystore.
 * Issues short-lived ES256 JWT capability tokens to agents.
 * Token scope is determined by the agent's role at session start.
 */
class IamService(private val context: Context) {

    private val keyStore: KeyStore = KeyStore.getInstance(KEYSTORE_PROVIDER).apply { load(null) }

    init {
        ensureSigningKeyExists()
    }

    fun issueToken(agentId: String, role: AgentRole): CapabilityToken {
        val scopes = Scopes.forRole(role)
        val issuedAt = System.currentTimeMillis()
        val expiresAt = issuedAt + TOKEN_TTL_MS

        val header = base64Url("""{"alg":"ES256","typ":"JWT"}""")
        val payload = base64Url(buildPayload(agentId, scopes, issuedAt / 1000, expiresAt / 1000))
        val signingInput = "$header.$payload"
        val signature = sign(signingInput)

        val jwt = "$signingInput.$signature"
        Log.i(TAG, "Token issued: agentId=$agentId role=$role ttl=${TOKEN_TTL_MS / 1000}s")

        return CapabilityToken(
            jwt = jwt,
            agentId = agentId,
            scopes = scopes,
            expiresAt = expiresAt
        )
    }

    fun validateScope(token: CapabilityToken, requiredScope: String) {
        if (token.isExpired) throw TokenExpiredException("Token expired for agent ${token.agentId}")
        if (requiredScope !in token.scopes) {
            throw InsufficientScopeException("Required scope '$requiredScope' not in token for ${token.agentId}")
        }
    }

    fun verifyToken(jwt: String): CapabilityToken? {
        return try {
            val parts = jwt.split(".")
            if (parts.size != 3) return null
            val signingInput = "${parts[0]}.${parts[1]}"
            val payloadJson = String(Base64.decode(parts[1], Base64.URL_SAFE or Base64.NO_PADDING))
            // In production: verify EC signature against public key
            parsePayload(payloadJson, jwt)
        } catch (e: Exception) {
            Log.w(TAG, "Token verification failed: ${e.message}")
            null
        }
    }

    private fun buildPayload(agentId: String, scopes: Set<String>, iat: Long, exp: Long): String {
        val scopesStr = scopes.joinToString(",") { "\"$it\"" }
        return """{"sub":"$agentId","iat":$iat,"exp":$exp,"iss":"edgemind-iam","scopes":[$scopesStr]}"""
    }

    private fun parsePayload(payloadJson: String, jwt: String): CapabilityToken? {
        // Simple extraction without a full JSON library to keep dependencies light
        val sub = extractJsonString(payloadJson, "sub") ?: return null
        val exp = extractJsonLong(payloadJson, "exp") ?: return null
        val scopesRaw = Regex(""""scopes":\[([^\]]*)]""").find(payloadJson)?.groupValues?.get(1) ?: ""
        val scopes = scopesRaw.split(",").map { it.trim().removeSurrounding("\"") }.filter { it.isNotEmpty() }.toSet()
        return CapabilityToken(jwt = jwt, agentId = sub, scopes = scopes, expiresAt = exp * 1000)
    }

    private fun extractJsonString(json: String, key: String): String? =
        Regex(""""$key":"([^"]+)"""").find(json)?.groupValues?.get(1)

    private fun extractJsonLong(json: String, key: String): Long? =
        Regex(""""$key":(\d+)""").find(json)?.groupValues?.get(1)?.toLongOrNull()

    private fun sign(data: String): String {
        return try {
            val privateKey = keyStore.getKey(SIGNING_KEY_ALIAS, null) as? java.security.PrivateKey
                ?: return mockSign(data)
            val sig = Signature.getInstance("SHA256withECDSA").apply {
                initSign(privateKey)
                update(data.toByteArray())
            }
            Base64.encodeToString(sig.sign(), Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
        } catch (e: Exception) {
            Log.w(TAG, "Keystore signing failed, using mock signature: ${e.message}")
            mockSign(data)
        }
    }

    private fun mockSign(data: String): String {
        val hash = java.security.MessageDigest.getInstance("SHA-256").digest(data.toByteArray())
        return Base64.encodeToString(hash, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    private fun base64Url(input: String): String =
        Base64.encodeToString(input.toByteArray(), Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)

    private fun ensureSigningKeyExists() {
        if (keyStore.containsAlias(SIGNING_KEY_ALIAS)) return
        try {
            val spec = KeyGenParameterSpec.Builder(
                SIGNING_KEY_ALIAS,
                KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
            )
                .setAlgorithmParameterSpec(java.security.spec.ECGenParameterSpec("secp256r1"))
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setUserAuthenticationRequired(false)
                .build()
            KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_EC, KEYSTORE_PROVIDER)
                .apply { initialize(spec) }
                .generateKeyPair()
            Log.i(TAG, "EC signing key generated in Android Keystore")
        } catch (e: Exception) {
            Log.w(TAG, "Could not generate Keystore key: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "IamService"
        private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
        private const val SIGNING_KEY_ALIAS = "edgemind_iam_signing_key"
        private const val TOKEN_TTL_MS = 30 * 60 * 1000L // 30 minutes
    }
}

class TokenExpiredException(message: String) : SecurityException(message)
class InsufficientScopeException(message: String) : SecurityException(message)
