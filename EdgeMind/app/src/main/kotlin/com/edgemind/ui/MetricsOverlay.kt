package com.edgemind.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.edgemind.data.GuardStatus
import com.edgemind.data.InferencePath
import com.edgemind.data.MetricsState

@Composable
fun MetricsOverlay(metrics: MetricsState, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = Color(0xFF0D1117),
        shape = RoundedCornerShape(bottomStart = 12.dp, bottomEnd = 12.dp)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "LIVE METRICS",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF58A6FF),
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                )
                GuardStatusChip(metrics.guardStatus, metrics.guardBlockReason)
            }

            Spacer(modifier = Modifier.height(2.dp))

            // Row 1: Path + Latency + Confidence
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricItem("PATH", metrics.activePath?.name ?: "–", pathColor(metrics.activePath))
                MetricItem("LATENCY", if (metrics.latencyMs > 0) "${metrics.latencyMs} ms" else "–", Color(0xFFE6EDF3))
                MetricItem("CONF", if (metrics.confidence > 0) "%.2f".format(metrics.confidence) else "–", confidenceColor(metrics.confidence))
            }

            // Confidence progress bar
            if (metrics.confidence > 0f) {
                val animConf by animateFloatAsState(
                    targetValue = metrics.confidence,
                    animationSpec = tween(400),
                    label = "confidence"
                )
                LinearProgressIndicator(
                    progress = { animConf },
                    modifier = Modifier.fillMaxWidth().height(3.dp).clip(RoundedCornerShape(2.dp)),
                    color = confidenceColor(metrics.confidence),
                    trackColor = Color(0xFF21262D)
                )
            }

            // Row 2: NPU + RAM + Token TTL
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricItem("NPU", "${metrics.npuUtilPct}%", Color(0xFF79C0FF))
                MetricItem("RAM", "${metrics.ramUsageMb} MB", Color(0xFFE6EDF3))
                MetricItem("TOKEN TTL", "${metrics.tokenTtlSeconds}s", tokenTtlColor(metrics.tokenTtlSeconds))
            }
        }
    }
}

@Composable
private fun MetricItem(label: String, value: String, valueColor: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = label,
            fontSize = 8.sp,
            fontFamily = FontFamily.Monospace,
            color = Color(0xFF8B949E),
            letterSpacing = 1.sp
        )
        Text(
            text = value,
            fontSize = 13.sp,
            fontFamily = FontFamily.Monospace,
            color = valueColor,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun GuardStatusChip(status: GuardStatus, blockReason: String?) {
    val (text, color) = when (status) {
        GuardStatus.IDLE -> "GUARD IDLE" to Color(0xFF8B949E)
        GuardStatus.PASS -> "GUARD PASS" to Color(0xFF3FB950)
        GuardStatus.BLOCK -> "GUARD BLOCK" to Color(0xFFF85149)
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(6.dp)
                .clip(CircleShape)
                .background(color)
        )
        Text(
            text = if (status == GuardStatus.BLOCK && blockReason != null) "BLOCK: ${blockReason.take(8)}" else text,
            fontSize = 9.sp,
            fontFamily = FontFamily.Monospace,
            color = color,
            fontWeight = FontWeight.Medium,
            letterSpacing = 0.5.sp
        )
    }
}

private fun pathColor(path: InferencePath?) = when (path) {
    InferencePath.SLM -> Color(0xFF3FB950)
    InferencePath.RAG -> Color(0xFFD29922)
    InferencePath.LLM -> Color(0xFF79C0FF)
    null -> Color(0xFF8B949E)
}

private fun confidenceColor(conf: Float) = when {
    conf >= 0.75f -> Color(0xFF3FB950)
    conf >= 0.5f -> Color(0xFFD29922)
    else -> Color(0xFFF85149)
}

private fun tokenTtlColor(ttl: Long) = when {
    ttl > 300 -> Color(0xFF3FB950)
    ttl > 60 -> Color(0xFFD29922)
    else -> Color(0xFFF85149)
}
