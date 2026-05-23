package com.edgemind.mcp.tools

import com.edgemind.data.ToolCall
import com.edgemind.data.ToolResult
import com.edgemind.mcp.ToolContext
import com.edgemind.mcp.ToolHandler
import com.edgemind.rag.RagPipeline
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * vector_search tool handler.
 * Queries the on-device FAISS / Kotlin-fallback vector store for RAG augmentation.
 */
class VectorTool(private val rag: RagPipeline) : ToolHandler {

    override suspend fun handle(call: ToolCall, ctx: ToolContext): ToolResult {
        val query = call.params["query"] ?: return ToolResult.Error("Missing 'query' parameter")
        val k = call.params["k"]?.toIntOrNull()?.coerceIn(1, 20) ?: 5

        val start = System.currentTimeMillis()
        val chunks = rag.retrieve(query, k)
        val latency = System.currentTimeMillis() - start

        val json = buildString {
            append("""{"chunks":[""")
            chunks.forEachIndexed { i, chunk ->
                if (i > 0) append(",")
                append("""{"id":${chunk.id},"score":${chunk.score},"text":${Json.encodeToString(chunk.text)}}""")
            }
            append("""],"count":${chunks.size}}""")
        }

        return ToolResult.Success(data = json, latencyMs = latency)
    }
}
