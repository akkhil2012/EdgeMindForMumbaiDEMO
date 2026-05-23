package com.edgemind.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.clickable
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Assessment
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgemind.data.AuditEntry
import com.edgemind.data.ChatMessage
import com.edgemind.data.InferencePath
import com.edgemind.data.MessageRole
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun AssistantScreen(vm: AssistantViewModel = viewModel()) {
    val state by vm.uiState.collectAsState()
    val listState = rememberLazyListState()
    val snackbarState = remember { SnackbarHostState() }
    var inputText by remember { mutableStateOf("") }

    // Auto-scroll to latest message
    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size - 1)
        }
    }

    // Show errors in snackbar
    LaunchedEffect(state.error) {
        state.error?.let {
            snackbarState.showSnackbar(it)
            vm.dismissError()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarState) },
        containerColor = Color(0xFF0D1117)
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .statusBarsPadding()
                .imePadding()
        ) {
            // ── Header ────────────────────────────────────────────────────
            AppHeader(
                showAuditLog = state.showAuditLog,
                onToggleAudit = vm::toggleAuditLog
            )

            // ── Metrics overlay ───────────────────────────────────────────
            MetricsOverlay(metrics = state.metrics)

            Spacer(modifier = Modifier.height(1.dp))

            // ── Content: chat or audit log ────────────────────────────────
            Box(modifier = Modifier.weight(1f)) {
                if (state.showAuditLog) {
                    AuditLogPanel(entries = state.auditLog)
                } else {
                    ChatPanel(
                        messages = state.messages,
                        listState = listState,
                        isProcessing = state.isProcessing
                    )
                }
            }

            // ── Quick prompt chips ────────────────────────────────────────
            if (!state.showAuditLog) {
                QuickPromptRow { prompt ->
                    inputText = prompt
                }
            }

            // ── Input bar ─────────────────────────────────────────────────
            InputBar(
                value = inputText,
                onValueChange = { inputText = it },
                onSend = {
                    if (inputText.isNotBlank()) {
                        vm.sendMessage(inputText)
                        inputText = ""
                    }
                },
                isProcessing = state.isProcessing,
                modifier = Modifier.navigationBarsPadding()
            )
        }
    }
}

@Composable
private fun AppHeader(showAuditLog: Boolean, onToggleAudit: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF161B22))
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "EdgeMind",
                color = Color(0xFFE6EDF3),
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
            )
            Text(
                text = "On-Device · MCP · ExecuTorch",
                color = Color(0xFF58A6FF),
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace
            )
        }
        Row {
            IconButton(onClick = onToggleAudit) {
                Icon(
                    imageVector = if (showAuditLog) Icons.Default.Close else Icons.Default.Assessment,
                    contentDescription = "Toggle audit log",
                    tint = if (showAuditLog) Color(0xFF58A6FF) else Color(0xFF8B949E)
                )
            }
        }
    }
}

@Composable
private fun ChatPanel(
    messages: List<ChatMessage>,
    listState: androidx.compose.foundation.lazy.LazyListState,
    isProcessing: Boolean
) {
    LazyColumn(
        state = listState,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxSize()
    ) {
        items(messages, key = { it.id }) { msg ->
            MessageBubble(msg)
        }
        if (isProcessing) {
            item {
                TypingIndicator()
            }
        }
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    val isUser = msg.role == MessageRole.USER
    val isSystem = msg.role == MessageRole.SYSTEM

    if (isSystem) {
        Text(
            text = msg.text,
            color = Color(0xFF8B949E),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
        )
        return
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Column(
            modifier = Modifier.widthIn(max = 300.dp),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            Surface(
                shape = RoundedCornerShape(
                    topStart = if (isUser) 16.dp else 4.dp,
                    topEnd = if (isUser) 4.dp else 16.dp,
                    bottomStart = 16.dp,
                    bottomEnd = 16.dp
                ),
                color = if (isUser) Color(0xFF1F6FEB) else Color(0xFF161B22)
            ) {
                Text(
                    text = msg.text,
                    color = Color(0xFFE6EDF3),
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
                )
            }

            if (!isUser && (msg.path != null || msg.confidence != null)) {
                Spacer(Modifier.height(3.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    msg.path?.let { path ->
                        PathBadge(path)
                    }
                    msg.confidence?.let { conf ->
                        Text(
                            text = "%.0f%%".format(conf * 100),
                            fontSize = 10.sp,
                            color = confidenceColor(conf),
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PathBadge(path: InferencePath) {
    val (label, color) = when (path) {
        InferencePath.SLM -> "SLM" to Color(0xFF3FB950)
        InferencePath.RAG -> "RAG" to Color(0xFFD29922)
        InferencePath.LLM -> "LLM" to Color(0xFF79C0FF)
    }
    Surface(
        shape = RoundedCornerShape(4.dp),
        color = color.copy(alpha = 0.15f)
    ) {
        Text(
            text = label,
            fontSize = 9.sp,
            color = color,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}

@Composable
private fun TypingIndicator() {
    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(14.dp),
            color = Color(0xFF58A6FF),
            strokeWidth = 2.dp
        )
        Text("Processing on-device...", fontSize = 12.sp, color = Color(0xFF8B949E))
    }
}

@Composable
private fun QuickPromptRow(onSelect: (String) -> Unit) {
    val prompts = listOf(
        "Chest pain + dyspnoea",
        "Differential diagnosis?",
        "Triage severity",
        "Drug interactions"
    )
    LazyRow(
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        items(prompts) { prompt ->
            Surface(
                shape = RoundedCornerShape(20.dp),
                color = Color(0xFF21262D),
                modifier = Modifier.clickable { onSelect(prompt) }
            ) {
                Text(
                    text = prompt,
                    fontSize = 12.sp,
                    color = Color(0xFF8B949E),
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                )
            }
        }
    }
}

@Composable
private fun InputBar(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    isProcessing: Boolean,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(Color(0xFF161B22))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.weight(1f),
            placeholder = {
                Text("Describe symptoms or ask a clinical question...", fontSize = 13.sp, color = Color(0xFF8B949E))
            },
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF30363D),
                unfocusedBorderColor = Color(0xFF21262D),
                focusedContainerColor = Color(0xFF0D1117),
                unfocusedContainerColor = Color(0xFF0D1117),
                focusedTextColor = Color(0xFFE6EDF3),
                unfocusedTextColor = Color(0xFFE6EDF3),
                cursorColor = Color(0xFF58A6FF)
            ),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { onSend() }),
            maxLines = 4,
            shape = RoundedCornerShape(12.dp)
        )
        IconButton(
            onClick = onSend,
            enabled = value.isNotBlank() && !isProcessing,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(if (value.isNotBlank() && !isProcessing) Color(0xFF1F6FEB) else Color(0xFF21262D))
        ) {
            Icon(
                imageVector = Icons.Default.Send,
                contentDescription = "Send",
                tint = Color.White,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
private fun AuditLogPanel(entries: List<AuditEntry>) {
    val fmt = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }

    if (entries.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("No audit entries yet", color = Color(0xFF8B949E), fontSize = 13.sp)
        }
        return
    }

    LazyColumn(
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxSize()
    ) {
        item {
            Text(
                "AUDIT JOURNAL — Signed & Timestamped",
                color = Color(0xFF58A6FF),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp,
                modifier = Modifier.padding(bottom = 4.dp)
            )
        }
        items(entries, key = { it.id }) { entry ->
            AuditEntryRow(entry, fmt)
        }
    }
}

@Composable
private fun AuditEntryRow(entry: AuditEntry, fmt: SimpleDateFormat) {
    Surface(
        shape = RoundedCornerShape(6.dp),
        color = Color(0xFF161B22),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = entry.eventType,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = eventTypeColor(entry.eventType),
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = fmt.format(Date(entry.timestamp)),
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF8B949E)
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                entry.inferencePathName?.let {
                    MonoLabel("path", it)
                }
                entry.latencyMs?.let {
                    MonoLabel("latency", "${it}ms")
                }
                entry.confidence?.let {
                    MonoLabel("conf", "%.2f".format(it))
                }
            }
            Text(
                text = "agent: ${entry.agentId.take(20)}  sig: ${entry.signature.take(12)}…",
                fontSize = 9.sp,
                fontFamily = FontFamily.Monospace,
                color = Color(0xFF6E7681)
            )
        }
    }
}

@Composable
private fun MonoLabel(key: String, value: String) {
    Text(
        text = "$key=$value",
        fontSize = 10.sp,
        fontFamily = FontFamily.Monospace,
        color = Color(0xFFE6EDF3)
    )
}

private fun eventTypeColor(type: String) = when (type) {
    "INFERENCE" -> Color(0xFF3FB950)
    "BLOCKED" -> Color(0xFFF85149)
    "TRAINING" -> Color(0xFF79C0FF)
    "TOKEN_ISSUED" -> Color(0xFFD29922)
    else -> Color(0xFF8B949E)
}

private fun confidenceColor(conf: Float) = when {
    conf >= 0.75f -> Color(0xFF3FB950)
    conf >= 0.5f -> Color(0xFFD29922)
    else -> Color(0xFFF85149)
}
