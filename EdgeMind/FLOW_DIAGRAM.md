# EdgeMind — Flow Diagrams

## 1. System Architecture Flow

```mermaid
flowchart TB
    subgraph UI["UI Layer"]
        AS[AssistantScreen]
        MO[MetricsOverlay]
        AS <--> MO
    end

    subgraph VM["ViewModel"]
        AVM[AssistantViewModel]
    end

    subgraph MCP["MCP Orchestration"]
        MCC[McpClient]
        MCS[McpServer]
        TR[ToolRegistry]
        MCC --> MCS
        MCS --> TR
    end

    subgraph SEC["Security"]
        GRD[GuardModel\nPrompt Guardrail]
        IAM[IamService\nJWT · ES256]
        AUD[AuditJournal\nHMAC-SHA256]
    end

    subgraph TOOLS["Tool Handlers"]
        IT[InferenceTool]
        VT[VectorTool]
        AT[AuditTool]
    end

    subgraph INF["Inference Engines"]
        SLM[SlmInferenceEngine\nPhi-3 Mini · <300ms]
        LLM[LlmInferenceEngine\n7B int4 · 1-5s]
        ET[ExecuTorchRunner\nJNI Bridge]
        TK[Tokenizer\nSentencePiece JNI]
        SLM --> ET
        LLM --> ET
        SLM --> TK
        LLM --> TK
    end

    subgraph RAG["RAG Pipeline"]
        RP[RagPipeline]
        EM[EmbedModel\nMiniLM-L6-v2 · 384-dim]
        FI[FaissIndex\nIndexFlatL2 JNI]
        RP --> EM
        RP --> FI
    end

    subgraph DB["Persistence (Room)"]
        DB1[(audit_entries)]
        DB2[(interactions)]
    end

    subgraph TRN["Background Training"]
        WM[WorkManager\n6h · charging+idle]
        TW[TrainingWorker]
        LT[LoraTrainer\nrank-8 · alpha-16]
        IDS[InteractionDataset]
        WM --> TW
        TW --> LT
        TW --> IDS
        LT --> ET
    end

    AS <--> AVM
    AVM --> MCC
    MCC --> IAM
    MCS --> GRD
    MCS --> IAM
    MCS --> AUD
    TR --> IT
    TR --> VT
    TR --> AT
    IT --> SLM
    IT --> LLM
    IT --> RP
    VT --> RP
    AUD --> DB1
    IDS --> DB2
    DB2 --> TW
    AUD -.->|Flow| AVM
```

---

## 2. Message Processing Flow

```mermaid
flowchart TD
    A([User types message]) --> B[AssistantViewModel\nsendMessage]
    B --> C[Detect TaskType\nNER / TRIAGE / INTENT\nFORM_EXTRACT / AUTO]
    C --> D{Keywords:\ndifferential / reasoning\nexplain / compare?}
    D -- yes --> E[preferLlm = true]
    D -- no --> F[preferLlm = false]
    E & F --> G[Build ToolCall\nRUN_INFERENCE_SLM]
    G --> H[McpClient\nensureFreshToken]

    H --> I{Token\nexpired?}
    I -- yes --> J[IamService\nissueToken · ES256]
    J --> K[New JWT\n30 min TTL]
    K --> L[McpServer\nhandleToolCall]
    I -- no --> L

    subgraph GATE["Security Gates"]
        L --> M[GuardModel\nclassify prompt]
        M --> N{Blocked?}
        N -- yes\nJAILBREAK/PII\nSELF_HARM --> O[AuditJournal\nlogBlocked]
        O --> P([Show block\nreason in UI])
        N -- no --> Q[IamService\nvalidateScope]
        Q --> R{Token\nvalid?}
        R -- expired --> S([ToolResult\n.Unauthorized])
        R -- missing scope --> S
        R -- valid --> T[ToolRegistry\ngetHandler]
    end

    T --> U[InferenceTool\nroute]
```

---

## 3. Three-Path Inference Routing

```mermaid
flowchart TD
    A[InferenceTool\nroute] --> B{prefer_llm\n= true?}

    B -- yes --> LLM_PATH

    B -- no --> C[SlmInferenceEngine\nrunInference]
    C --> D[Tokenizer\nencode prompt]
    D --> E[ExecuTorchRunner\nforward · JNI]
    E --> F[greedyDecode\nlogits → tokens]
    F --> G[Tokenizer\ndecode → text]
    G --> H{confidence\n≥ 0.75?}

    H -- yes --> DONE_SLM([Return\npath = SLM])

    H -- no --> I[RagPipeline\nretrieve k=5]
    I --> J[EmbedModel\nembed query\n384-dim]
    J --> K[FaissIndex\nsearch IndexFlatL2]
    K --> L[Top-5 chunks\n+ similarity scores]
    L --> M[buildAugmentedPrompt\nchunks + original]
    M --> N[SlmInferenceEngine\nre-run with RAG context]
    N --> O{confidence\n≥ 0.75?}

    O -- yes --> DONE_RAG([Return\npath = RAG])

    O -- no --> LLM_PATH[LlmInferenceEngine\nrunInference + augmented ctx]
    LLM_PATH --> P[Tokenizer encode]
    P --> Q[ExecuTorchRunner forward]
    Q --> R[greedyDecode + decode]
    R --> DONE_LLM([Return\npath = LLM])

    DONE_SLM & DONE_RAG & DONE_LLM --> AUDIT[AuditJournal\nlogInference\nagentId · path · conf · latency]
    AUDIT --> RECORD[InteractionDataset\nrecord + anonymize PII]
    RECORD --> RESULT([ToolResult.Success\nJSON response])
```

---

## 4. Security & IAM Flow

```mermaid
flowchart LR
    subgraph ROLES["Agent Roles → Scopes"]
        R1[DEFAULT] --> S1[INFERENCE_SLM\nKNOWLEDGE_READ]
        R2[ELEVATED] --> S2[INFERENCE_SLM\nINFERENCE_LLM\nKNOWLEDGE_READ]
        R3[SYSTEM] --> S3[INFERENCE_SLM\nINFERENCE_LLM\nKNOWLEDGE_READ\nSENSOR_MIC\nAUDIT_WRITE]
    end

    subgraph TOKEN["Token Lifecycle"]
        T1[issueToken\nAgentRole + scopes] --> T2[ES256 sign\nAndroid Keystore]
        T2 --> T3[CapabilityToken\nJWT · 30 min TTL]
        T3 --> T4{expired?}
        T4 -- yes --> T1
        T4 -- no --> T5[validateScope\ncheck required scope ∈ token]
    end

    subgraph GUARD["GuardModel Pipeline"]
        G1[Prompt] --> G2[SHA-256 hash]
        G2 --> G3[Heuristic regex\njailbreak · PII\nself-harm · override]
        G3 --> G4{Native model\navailable?}
        G4 -- yes --> G5[ExecuTorch JNI\n60M classifier]
        G4 -- no --> G6[Heuristic result]
        G5 & G6 --> G7{isBlocked?}
        G7 -- yes --> G8[GuardResult\nBLOCKED + reason]
        G7 -- no --> G9[GuardResult\nPASSED]
    end

    subgraph AUDIT["Audit Trail"]
        A1[HMAC-SHA256\nsign entry] --> A2[(Room DB\naudit_entries)]
        A2 --> A3[Flow emissions\nto ViewModel]
    end

    T5 --> GUARD
    G9 --> T5
    G8 --> AUDIT
    T5 --> AUDIT
```

---

## 5. Background Training Flow

```mermaid
flowchart TD
    A([Device:\ncharging + idle ≥ 10 min]) --> B[WorkManager\ntrigger every 6h]
    B --> C[TrainingWorker\ndoWork]
    C --> D[InteractionDataset\ngetSamples max=500]
    D --> E[Room SELECT\naccepted interactions]
    E --> F{samples\n≥ 50?}

    F -- no --> G([Result.success\nskip — not enough data])

    F -- yes --> H[Anonymize PII\nSSN · phone · email\n→ REDACTED]
    H --> I[LoraTrainer\ntrain rank-8 · alpha-16\nlr=1e-4 · epochs=3]

    I --> J[Prepare LoRA\nadapter weights]
    J --> K{epoch loop}
    K --> L[Forward pass\nExecuTorch JNI]
    L --> M[Compute loss]
    M --> N[Backward\nAdamW update]
    N --> O{more\nepochs?}
    O -- yes --> K
    O -- no --> P[Save adapter\nweights to disk]

    P --> Q[AuditJournal\nlogTraining\nsamples · loss · latency]
    Q --> R([Result.success\nmodel updated locally\nno data egress])
```

---

## 6. RAG Pipeline Detail

```mermaid
flowchart TD
    A([Query text]) --> B[EmbedModel\nall-MiniLM-L6-v2]
    B --> C[MiniLM forward pass\nJNI or mock]
    C --> D[384-dim\nFloatArray]
    D --> E[FaissIndex\nIndexFlatL2 search]

    subgraph KB["Knowledge Base (23 clinical chunks)"]
        KB1[ESI Triage levels]
        KB2[ACS red flags]
        KB3[PE signs]
        KB4[Drug interactions]
        KB5[Vital signs]
    end

    E <-.->|indexed| KB
    E --> F[Top-k chunks\n+ L2 distances]
    F --> G[Compute cosine\nsimilarity scores]
    G --> H[buildAugmentedPrompt\nchunks + original query]
    H --> I([Augmented prompt\n→ SLM / LLM])
```
