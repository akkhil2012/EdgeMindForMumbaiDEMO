package com.edgemind.training

import android.content.Context
import android.util.Log
import com.edgemind.data.AppDatabase
import com.edgemind.data.InteractionRecord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.regex.Pattern

/**
 * Local training data manager.
 * Collects accepted inference pairs (prompt → response), anonymises PII,
 * and provides batched samples for LoRA fine-tuning.
 */
class InteractionDataset(context: Context) {

    private val dao = AppDatabase.getInstance(context).interactionDao()

    suspend fun record(
        prompt: String,
        response: String,
        pathName: String,
        confidence: Float
    ) = withContext(Dispatchers.IO) {
        val record = InteractionRecord(
            timestamp = System.currentTimeMillis(),
            prompt = anonymise(prompt),
            response = anonymise(response),
            pathName = pathName,
            confidence = confidence,
            accepted = true,
            anonymised = true
        )
        dao.insert(record)
    }

    suspend fun getSamples(maxSamples: Int = 500): List<TrainingSample> =
        withContext(Dispatchers.IO) {
            dao.getTrainingSamples(maxSamples).map { record ->
                TrainingSample(
                    inputText = record.prompt,
                    targetText = record.response,
                    weight = record.confidence
                )
            }
        }

    suspend fun count(): Int = dao.count()

    /**
     * Strip PII patterns before storing interaction data.
     * Names and IDs are replaced with placeholders.
     */
    private fun anonymise(text: String): String {
        var result = text
        // SSN / Aadhaar / NIC patterns
        result = Regex("\\b\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{4}\\b").replace(result, "[ID_REDACTED]")
        result = Regex("\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b").replace(result, "[ID_REDACTED]")
        // Phone numbers
        result = Regex("\\b[+]?[0-9]{10,13}\\b").replace(result, "[PHONE_REDACTED]")
        // Email addresses
        result = Regex("[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}").replace(result, "[EMAIL_REDACTED]")
        return result
    }

    data class TrainingSample(
        val inputText: String,
        val targetText: String,
        val weight: Float = 1.0f
    )

    companion object {
        private const val TAG = "InteractionDataset"
    }
}
