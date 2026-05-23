# EdgeMind — Complete Sequence Diagram

## 1. App Startup & Initialization

```mermaid
sequenceDiagram
    autonumber
    participant OS as Android OS
    participant App as EdgeMindApp
    participant MA as MainActivity
    participant AVM as AssistantViewModel
    participant TR as ToolRegistry
    participant IAM as IamService
    participant GRD as GuardModel
    participant SLM as SlmInferenceEngine
    participant LLM as LlmInferenceEngine
    participant RAG as RagPipeline
    participant AUD as AuditJournal
    participant MCS as McpServer
    participant MCC as McpClient
    participant SCH as IdleScheduler
    participant WM as WorkManager

    OS->>App: onCreate()
    App->>WM: setLoggingLevel(INFO)
    App-->>OS: ready

    OS->>MA: onCreate()
    MA->>MA: enableEdgeToEdge()
    MA->>AVM: new AssistantViewModel(app)

    AVM->>AVM: init { viewModelScope.launch { initialise() } }

    par Infrastructure init
        AVM->>SLM: new SlmInferenceEngine(context)
        SLM->>SLM: load Phi-3 Mini int4 (.pte)
    and
        AVM->>LLM: new LlmInferenceEngine(context)
        LLM->>LLM: load 7B int4 (.pte)
    and
        AVM->>RAG: new RagPipeline(context)
        RAG->>RAG: load EmbedModel (MiniLM-L6-v2)
        RAG->>RAG: load FaissIndex from assets
    and
        AVM->>GRD: new GuardModel(context)
        GRD->>GRD: load guard classifier (.pte, 60M)
    and
        AVM->>IAM: new IamService(context)
        IAM->>IAM: init Android Keystore (ES256)
    and
        AVM->>AUD: new AuditJournal(db)
        AUD->>AUD: init Room database (AppDatabase)
    end

    AVM->>TR: new ToolRegistry()
    AVM->>TR: register(RUN_INFERENCE_SLM/LLM, InferenceTool)
    AVM->>TR: register(VECTOR_SEARCH, VectorTool)
    AVM->>TR: register(AUDIT_LOG_WRITE, AuditTool)

    AVM->>MCS: new McpServer(registry, guard, iam, audit)
    AVM->>MCC: new McpClient(server, agentRole=ELEVATED)
    MCC->>IAM: issueToken(agentId, ELEVATED scopes)
    IAM-->>MCC: CapabilityToken (JWT, ES256, 30 min TTL)

    AVM->>MCS: markInitialized()

    AVM->>AUD: observeRecent() [Flow]
    AUD-->>AVM: Flow<List<AuditEntry>> (ongoing)

    AVM->>SCH: schedule(context)
    SCH->>WM: enqueueUniquePeriodicWork(6h, charging+idle)

    AVM-->>MA: initialized
    MA->>MA: setContent { AssistantScreen(viewModel) }
```

---

## 2. User Message — Main Processing Flow

```mermaid
sequenceDiagram
    autonumber
    participant UI as AssistantScreen
    participant AVM as AssistantViewModel
    participant MCC as McpClient
    participant IAM as IamService
    participant MCS as McpServer
    participant GRD as GuardModel
    participant TR as ToolRegistry
    participant IT as InferenceTool
    participant SLM as SlmInferenceEngine
    participant ET as ExecuTorchRunner
    participant TK as Tokenizer
    participant RAG as RagPipeline
    participant EM as EmbedModel
    participant FI as FaissIndex
    participant LLM as LlmInferenceEngine
    participant AUD as AuditJournal
    participant IDS as InteractionDataset

    UI->>AVM: sendMessage(text)
    AVM->>AVM: append ChatMessage(USER, text)
    AVM->>AVM: _uiState: isProcessing = true

    AVM->>AVM: detectTaskType(text) → NER/TRIAGE/INTENT/FORM_EXTRACT/AUTO
    AVM->>AVM: preferLlm = text contains "differential"/"reasoning"/"explain"/…
    AVM->>AVM: create ToolCall(RUN_INFERENCE_SLM, {prompt, prefer_llm, task_type})

    AVM->>MCC: invokeTool(call, prompt)

    MCC->>MCC: ensureFreshToken()
    alt Token expired
        MCC->>IAM: issueToken(agentId, scopes)
        IAM-->>MCC: new CapabilityToken (JWT, 30 min)
    end

    MCC->>MCS: handleToolCall(call, token, prompt)

    %% ── STEP 1: Guardrail ──────────────────────────────────────────
    rect rgb(255, 240, 240)
        note over MCS,GRD: Step 1 — Guardrail Check
        MCS->>GRD: classify(prompt)
        GRD->>GRD: SHA-256 hash prompt
        GRD->>GRD: heuristicClassify() — regex patterns:<br/>jailbreak / PII (SSN, Aadhaar, CC) /<br/>self-harm / instruction override
        alt Native model available
            GRD->>GRD: nativeClassify() via ExecuTorch JNI
        end
        GRD-->>MCS: GuardResult(isBlocked, reason, promptHash)

        alt isBlocked = true
            MCS->>AUD: logBlocked(agentId, reason, promptHash)
            MCS-->>MCC: ToolResult.Blocked(reason)
            MCC-->>AVM: ToolResult.Blocked
            AVM->>UI: show block reason banner
        end
    end

    %% ── STEP 2: IAM Validation ──────────────────────────────────────
    rect rgb(240, 255, 240)
        note over MCS,IAM: Step 2 — IAM / Scope Validation
        MCS->>IAM: validateScope(token, INFERENCE_SLM)
        IAM->>IAM: check token.expiresAt > now
        IAM->>IAM: check INFERENCE_SLM ∈ token.scopes
        alt Expired
            IAM-->>MCS: TokenExpiredException
            MCS-->>MCC: ToolResult.Unauthorized("token expired")
        else Insufficient scope
            IAM-->>MCS: InsufficientScopeException
            MCS-->>MCC: ToolResult.Unauthorized("missing scope")
        end
        IAM-->>MCS: valid
    end

    %% ── STEP 3: Route to handler ────────────────────────────────────
    rect rgb(240, 240, 255)
        note over MCS,TR: Step 3 — Registry Lookup & Dispatch
        MCS->>TR: getHandler(RUN_INFERENCE_SLM)
        TR-->>MCS: InferenceTool handler
        MCS->>IT: handle(call, ToolContext(agentId, sessionId))
    end

    %% ── InferenceTool: Three-Path Routing ───────────────────────────
    rect rgb(255, 255, 220)
        note over IT,LLM: InferenceTool — Three-Path Routing

        alt prefer_llm = true
            IT->>LLM: runInference(prompt, context)
            note right of IT: skip to LLM directly
        else SLM first
            IT->>SLM: runInference(prompt, taskType, maxTokens)

            SLM->>TK: encode(systemPrompt + userPrompt)
            TK->>TK: SentencePiece tokenize → token IDs
            TK-->>SLM: IntArray tokens

            SLM->>ET: forward(tokens)
            ET->>ET: call native executorch_forward() JNI
            ET-->>SLM: FloatArray logits

            SLM->>SLM: greedyDecode(logits) → token IDs
            SLM->>TK: decode(token IDs)
            TK-->>SLM: response text

            SLM-->>IT: InferenceResult(text, confidence, latency, path=SLM)

            alt confidence ≥ 0.75
                IT->>AUD: logInference(agentId, SLM, confidence, latency)
                IT-->>MCS: ToolResult.Success(json, latency)
            else confidence < 0.75 — RAG fallback
                note over IT,FI: RAG Fallback Path
                IT->>RAG: retrieve(query=prompt, k=5)

                RAG->>EM: embed(prompt)
                EM->>EM: MiniLM forward → 384-dim FloatArray
                EM-->>RAG: queryVector

                RAG->>FI: search(queryVector, k=5)
                FI->>FI: FAISS IndexFlatL2 search (JNI)
                FI-->>RAG: top-5 chunks + distances

                RAG-->>IT: List<RagChunk>(text, similarity)
                IT->>RAG: buildAugmentedPrompt(prompt, chunks)
                RAG-->>IT: augmentedPrompt

                IT->>SLM: runInference(augmentedPrompt, taskType)
                SLM-->>IT: InferenceResult(text, confidence, path=RAG)

                alt confidence ≥ 0.75
                    IT->>AUD: logInference(agentId, RAG, confidence, latency)
                    IT-->>MCS: ToolResult.Success(json, latency)
                else confidence still < 0.75 — LLM escalation
                    note over IT,LLM: LLM Escalation Path
                    IT->>LLM: runInference(prompt, augmentedContext)

                    LLM->>TK: encode(systemPrompt + augmentedPrompt)
                    TK-->>LLM: tokens
                    LLM->>ET: forward(tokens)
                    ET-->>LLM: logits
                    LLM->>LLM: greedyDecode → token IDs
                    LLM->>TK: decode(token IDs)
                    TK-->>LLM: response text

                    LLM-->>IT: InferenceResult(text, confidence, path=LLM)
                    IT->>AUD: logInference(agentId, LLM, confidence, latency)
                    IT-->>MCS: ToolResult.Success(json, latency)
                end
            end
        end
    end

    IT->>IDS: record(prompt, response, path, confidence)
    IDS->>IDS: anonymize PII (SSN/phone/email → [REDACTED])
    IDS->>IDS: Room INSERT INTO interactions

    AUD->>AUD: HMAC-SHA256 sign entry
    AUD->>AUD: Room INSERT INTO audit_entries

    MCS-->>MCC: ToolResult.Success(data, latencyMs)
    MCC-->>AVM: ToolResult.Success

    AVM->>AVM: parse JSON response
    AVM->>AVM: append ChatMessage(ASSISTANT, text, path, confidence)
    AVM->>AVM: _uiState: isProcessing = false
    AVM-->>UI: updated UiState

    UI->>UI: render MessageBubble (path badge + confidence)
    UI->>UI: render MetricsOverlay (latency/conf/NPU/RAM/tokens)
```

---

## 3. Background Idle Training Flow

```mermaid
sequenceDiagram
    autonumber
    participant WM as WorkManager
    participant TW as TrainingWorker
    participant IDS as InteractionDataset
    participant LT as LoraTrainer
    participant ET as ExecuTorchRunner
    participant AUD as AuditJournal

    note over WM: Triggers when: device charging + idle ≥ 10 min<br/>Interval: every 6 hours

    WM->>TW: doWork()
    TW->>IDS: getSamples(max=500)
    IDS->>IDS: Room SELECT accepted interactions
    IDS-->>TW: List<InteractionRecord>

    alt samples < 50
        TW-->>WM: Result.success() [skip — not enough data]
    else samples ≥ 50
        TW->>LT: train(samples, config={rank=8, alpha=16, lr=1e-4, epochs=3})
        LT->>LT: prepare LoRA adapter (rank-8, alpha-16)
        loop each epoch
            LT->>LT: forward pass (ExecuTorch)
            LT->>LT: compute loss
            LT->>LT: backward + AdamW update
            LT-->>TW: onProgress(epoch, loss)
        end
        LT->>ET: save updated adapter weights
        LT-->>TW: training complete

        TW->>AUD: logTraining(samplesCount, finalLoss, latency)
        AUD->>AUD: HMAC-SHA256 sign + Room INSERT

        TW-->>WM: Result.success()
    end
```

---

## 4. Audit Log Observation Flow (Reactive)

```mermaid
sequenceDiagram
    autonumber
    participant AVM as AssistantViewModel
    participant AUD as AuditJournal
    participant DAO as AuditDao
    participant DB as Room (AppDatabase)
    participant UI as AssistantScreen

    note over AVM: Runs continuously in viewModelScope

    AVM->>AUD: observeRecent() → Flow<List<AuditEntry>>
    AUD->>DAO: getRecentFlow(limit=50)
    DAO->>DB: SELECT * FROM audit_entries ORDER BY timestamp DESC LIMIT 50
    DB-->>DAO: LiveData / Flow<List<AuditEntry>>
    DAO-->>AUD: Flow
    AUD-->>AVM: Flow<List<AuditEntry>>

    loop on each new AuditEntry
        DB-->>AVM: emit updated list
        AVM->>AVM: _uiState.update { copy(auditLog = entries) }
        AVM-->>UI: recompose audit log panel
    end
```

---

## 5. VectorSearch Tool Flow

```mermaid
sequenceDiagram
    autonumber
    participant AVM as AssistantViewModel
    participant MCC as McpClient
    participant MCS as McpServer
    participant VT as VectorTool
    participant RAG as RagPipeline
    participant EM as EmbedModel
    participant FI as FaissIndex

    AVM->>MCC: invokeTool(ToolCall(VECTOR_SEARCH, {query, k}), prompt)
    MCC->>MCS: handleToolCall(call, token, prompt)
    MCS->>MCS: validateScope(token, KNOWLEDGE_READ)
    MCS->>VT: handle(call, ctx)

    VT->>RAG: retrieve(query, k)
    RAG->>EM: embed(query)
    EM-->>RAG: FloatArray(384)
    RAG->>FI: search(vector, k)
    FI-->>RAG: List<(chunkId, distance)>
    RAG->>RAG: map chunk IDs → text + compute similarity
    RAG-->>VT: List<RagChunk>

    VT->>VT: serialize chunks to JSON
    VT-->>MCS: ToolResult.Success(json)
    MCS-->>MCC: ToolResult.Success
    MCC-->>AVM: ToolResult.Success
```

---

## Architecture Overview (Component Map)

```
┌─────────────────────────────────────────────────────┐
│                    UI Layer                          │
│  AssistantScreen ◄──► AssistantViewModel             │
│  MetricsOverlay                                      │
└─────────────┬───────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────┐
│                  MCP Orchestration                   │
│  McpClient ──► McpServer ──► ToolRegistry            │
│                    │                                 │
│         ┌──────────┴──────────┐                      │
│         ▼                     ▼                      │
│    GuardModel             IamService                 │
└─────────┬────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────┐
│                  Tool Handlers                        │
│  InferenceTool   VectorTool   AuditTool              │
└──┬──────────┬──────────────────────────────────────┘
   │          │
   │    ┌─────▼──────────────────────┐
   │    │     RAG Pipeline           │
   │    │  EmbedModel + FaissIndex   │
   │    └────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────┐
│              Inference Engines                       │
│  SlmInferenceEngine (Phi-3 Mini, <300ms)            │
│  LlmInferenceEngine (7B int4, 1-5s)                 │
│  ExecuTorchRunner (JNI) + Tokenizer (JNI)           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              Persistence & Training                  │
│  AuditJournal → Room (audit_entries)                │
│  InteractionDataset → Room (interactions)           │
│  LoraTrainer ← TrainingWorker ← WorkManager (6h)    │
└─────────────────────────────────────────────────────┘
```
