package com.edgemind

import android.app.Application
import android.util.Log
import androidx.work.Configuration
import androidx.work.WorkManager
import com.edgemind.training.TrainingWorker

class EdgeMindApp : Application(), Configuration.Provider {

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "EdgeMind application starting")
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setMinimumLoggingLevel(Log.INFO)
            .build()

    companion object {
        private const val TAG = "EdgeMindApp"
    }
}
