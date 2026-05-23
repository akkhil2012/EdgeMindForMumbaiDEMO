package com.edgemind.training

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.edgemind.data.AgentRole
import com.edgemind.data.Scopes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

/**
 * WorkManager idle-session trigger.
 * Schedules LoRA fine-tuning when the device is charging and idle for 10 minutes.
 * Training happens entirely on-device — no data egress.
 */
class IdleScheduler(private val context: Context) {

    fun schedule() {
        val constraints = Constraints.Builder()
            .setRequiresCharging(true)
            .setRequiresBatteryNotLow(true)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()

        val request = PeriodicWorkRequestBuilder<TrainingWorker>(
            repeatInterval = 6,
            repeatIntervalTimeUnit = TimeUnit.HOURS,
            flexTimeInterval = 30,
            flexTimeIntervalUnit = TimeUnit.MINUTES
        )
            .setConstraints(constraints)
            .setInitialDelay(10, TimeUnit.MINUTES)
            .build()

        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            WORK_NAME,
            ExistingPeriodicWorkPolicy.KEEP,
            request
        )

        Log.i(TAG, "Idle training scheduled (charging + idle)")
    }

    fun cancel() {
        WorkManager.getInstance(context).cancelUniqueWork(WORK_NAME)
        Log.i(TAG, "Idle training cancelled")
    }

    companion object {
        private const val TAG = "IdleScheduler"
        private const val WORK_NAME = "edgemind_lora_training"
    }
}

class TrainingWorker(
    private val context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.Default) {
        Log.i(TAG, "Idle training session starting")
        return@withContext try {
            val dataset = InteractionDataset(context)
            val sampleCount = dataset.count()

            if (sampleCount < MIN_SAMPLES) {
                Log.i(TAG, "Not enough samples ($sampleCount < $MIN_SAMPLES), skipping")
                return@withContext Result.success()
            }

            val samples = dataset.getSamples(500)
            val trainer = LoraTrainer(context)
            val result = trainer.train(samples) { step, loss ->
                if (step % 25 == 0) Log.d(TAG, "Training step $step loss=$loss")
            }

            Log.i(TAG, "Training complete: ${result.stepsCompleted} steps, loss=${result.finalLoss}, duration=${result.durationMs}ms")
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Training failed: ${e.message}", e)
            Result.retry()
        }
    }

    companion object {
        private const val TAG = "TrainingWorker"
        private const val MIN_SAMPLES = 50
    }
}
