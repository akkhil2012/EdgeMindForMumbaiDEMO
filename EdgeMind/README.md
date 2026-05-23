# EdgeMind — MCP-Orchestrated On-Device AI

**MCP Summit Mumbai 2025 · Android · ExecuTorch · Hybrid Inference · Federated Learning**

A fully self-contained Android application demonstrating Model Context Protocol (MCP) orchestration running entirely on-device, powered by ExecuTorch with Qualcomm HTP and XNNPACK backends.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Application (Jetpack Compose)                  │
│  AssistantScreen ← AssistantViewModel           │
│  MetricsOverlay (live NPU/RAM/latency/conf)     │
├─────────────────────────────────────────────────┤
│  MCP Orchestration                              │
│  McpClient → McpServer → ToolRegistry           │
│    ├── GuardModel (prompt guardrails)           │
│    ├── IamService (JWT / Android Keystore)      │
│    └── AuditJournal (signed local log)          │
├─────────────────────────────────────────────────┤
│  Inference (ExecuTorch .pte)                    │
│  Path 1 — SLM  (Phi-3 Mini int4, < 300ms NPU)  │
│  Path 2 — LLM  (7B int4, 1–5s)                 │
│  Path 3 — RAG  (SLM + FAISS, SLM + ~50ms)      │
├─────────────────────────────────────────────────┤
│  Hardware                                       │
│  Qualcomm HTP (NPU) · XNNPACK (CPU) · Vulkan   │
└─────────────────────────────────────────────────┘
```

## Three-Path Routing Decision

```
Prompt
  │
  ▼
GuardModel.classify()          ← blocks jailbreaks, PII, safety
  │ PASS
  ▼
IamService.validateScope()     ← JWT capability token check
  │ OK
  ▼
SLM inference (Phi-3 Mini)     ← Path 1: < 300ms on NPU
  │
  ├─ confidence ≥ 0.75 → return result
  │
  ▼
RAG fallback                   ← Path 3: embed + FAISS + re-run SLM
  │
  ├─ confidence ≥ 0.75 → return result
  │
  ▼
LLM escalation                 ← Path 2: 7B model, 1–5s
  │
  ▼
AuditJournal.log()             ← every decision signed + timestamped
```

## Quick Start

### 1. Prerequisites

```
Android Studio Hedgehog (2023.1.1) or newer
NDK r25c
CMake 3.22+
Python 3.10+ (for model export)
Snapdragon 8-series device (or any Android 10+ for mock mode)
```

### 2. Run in Mock Mode (no native libraries needed)

The app compiles and runs on any Android device without `.pte` model files or native libraries. All inference, FAISS search, and guardrails fall back to Kotlin implementations that simulate the demo scenario.

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 3. Export Real Models (for production)

```bash
pip install executorch torchao qai-hub-models sentence-transformers

# SLM — Phi-3 Mini int4 for Snapdragon 8 Gen 3
python scripts/export_phi3.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --quant int4 --group-size 128 \
    --backend qualcomm --soc SM8650 \
    --output app/src/main/assets/models/phi3_mini.pte

# Speech-to-text
python scripts/export_whisper.py \
    --model openai/whisper-small \
    --backend qualcomm \
    --output app/src/main/assets/models/whisper_small.pte

# Clinical knowledge base FAISS index
python scripts/build_kb_index.py \
    --corpus data/clinical_kb.jsonl \
    --output-index app/src/main/assets/models/clinical_kb.faiss
```

### 4. Native Libraries (ExecuTorch + FAISS)

Place prebuilt `.so` files in `app/jniLibs/arm64-v8a/`:

| Library | Source |
|---|---|
| `libexecutorch.so` | ExecuTorch Android AAR from [pytorch/executorch releases](https://github.com/pytorch/executorch/releases) |
| `libfaiss.so` | Build FAISS for Android: [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |

Then uncomment the link lines in `app/CMakeLists.txt`.

### 5. Stream live logs

```bash
adb logcat -s EdgeMind:D McpServer:D IamService:D GuardModel:D ExecuTorchRunner:D
```

## Demo Scenario (Section 7 from architecture doc)

Try these prompts in sequence:

1. **"I have chest pain and shortness of breath"**
   → NER extracts symptoms → Triage gives 0.62 conf → RAG triggered → 0.91 conf → ESI Level 2

2. **"What is the differential diagnosis?"**
   → LLM path escalation → ACS / PE / Aortic dissection with reasoning

3. Open the audit log (chart icon) → view every signed decision

## Project Structure

```
app/src/main/kotlin/com/edgemind/
├── mcp/             # McpServer, McpClient, ToolRegistry, tools/
├── inference/       # ExecuTorchRunner, SlmInferenceEngine, LlmInferenceEngine, Tokenizer
├── rag/             # EmbedModel, FaissIndex, RagPipeline
├── security/        # GuardModel, IamService, AuditJournal
├── training/        # LoraTrainer, InteractionDataset, IdleScheduler
├── ui/              # AssistantScreen, MetricsOverlay, AssistantViewModel
└── data/            # Models, AppDatabase, Room DAOs

cpp/
├── executorch_jni.cpp   # ExecuTorch JNI bridge
└── faiss_jni.cpp        # FAISS JNI bridge

scripts/
├── export_phi3.py       # torch.export + quantize + ExecuTorch lowering
├── export_whisper.py    # Whisper Small export
└── build_kb_index.py    # Clinical KB FAISS index builder
```

## Performance Benchmarks (Snapdragon 8 Gen 3, HTP)

| Workload | Latency |
|---|---|
| Phi-3 Mini prefill 128 tok | ~180 ms |
| Phi-3 Mini decode 64 tok | ~90 ms |
| Whisper Small 5s audio | ~1.4 s |
| MiniLM embed 128 tok | ~18 ms |
| FAISS search k=5 / 5k vectors | ~4 ms |
| Guard model classify | ~22 ms |

## Security

- **Prompt guardrails**: on-device classifier blocks jailbreaks, PII, medical safety red-flags before any inference
- **IAM**: ES256 JWT tokens signed by Android Keystore; short-lived (30 min), scoped per tool
- **Audit journal**: every agent action signed with SHA-256 HMAC, persisted to Room database
- **Zero cloud dependency**: no API keys, no data egress, GDPR/DPDP compliant by design

## Federated Learning

LoRA adapters (rank 8, alpha 16) fine-tuned locally on anonymised interaction history during device idle sessions (charging + idle ≥ 10 min). Training via ExecuTorch training API with gradient checkpointing for low-RAM devices.
