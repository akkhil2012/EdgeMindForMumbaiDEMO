package com.edgemind.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface AuditDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entry: AuditEntry)

    @Query("SELECT * FROM audit_entries ORDER BY timestamp DESC LIMIT 100")
    fun observeRecent(): Flow<List<AuditEntry>>

    @Query("SELECT * FROM audit_entries ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecent(limit: Int = 50): List<AuditEntry>

    @Query("DELETE FROM audit_entries WHERE timestamp < :before")
    suspend fun pruneBefore(before: Long)
}

@Dao
interface InteractionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(record: InteractionRecord): Long

    @Query("SELECT * FROM interactions WHERE accepted = 1 ORDER BY timestamp DESC LIMIT :maxSamples")
    suspend fun getTrainingSamples(maxSamples: Int = 500): List<InteractionRecord>

    @Query("SELECT COUNT(*) FROM interactions")
    suspend fun count(): Int

    @Query("UPDATE interactions SET anonymised = 1 WHERE id = :id")
    suspend fun markAnonymised(id: Long)
}
