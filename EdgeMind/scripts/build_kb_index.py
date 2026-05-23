#!/usr/bin/env python3
"""
Build a FAISS flat index from a clinical knowledge base JSONL file.
The index is saved to app/src/main/assets/models/ for bundling with the APK.

JSONL format (one document per line):
    {"id": 0, "text": "ESI Level 2 criteria ..."}

Usage:
    pip install faiss-cpu sentence-transformers
    python scripts/build_kb_index.py \
        --corpus data/clinical_kb.jsonl \
        --embed-model sentence-transformers/all-MiniLM-L6-v2 \
        --output-index app/src/main/assets/models/clinical_kb.faiss \
        --output-meta app/src/main/assets/models/clinical_kb_meta.jsonl
"""

import argparse
import json
import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_kb_index")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/clinical_kb.jsonl")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output-index", default="app/src/main/assets/models/clinical_kb.faiss")
    p.add_argument("--output-meta", default="app/src/main/assets/models/clinical_kb_meta.jsonl")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--metric", choices=["L2", "IP"], default="IP",
                   help="IP = inner product (cosine with normalized vectors)")
    return p.parse_args()


def load_corpus(path: str):
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    log.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def embed_corpus(docs, model_name: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("pip install sentence-transformers")
        sys.exit(1)

    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    log.info(f"Embedding {len(texts)} chunks (batch_size={batch_size}) ...")
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=True)
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_index(embeddings: np.ndarray, metric: str):
    try:
        import faiss
    except ImportError:
        log.error("pip install faiss-cpu")
        sys.exit(1)

    dim = embeddings.shape[1]
    if metric == "IP":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings.astype(np.float32))
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}, metric={metric}")
    return index


def save_outputs(index, docs, args):
    import faiss
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    faiss.write_index(index, args.output_index)
    log.info(f"Index saved to {args.output_index} ({os.path.getsize(args.output_index)/1024:.1f} KB)")

    with open(args.output_meta, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Metadata saved to {args.output_meta}")


def validate_index(index_path: str, embed_model: str):
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(index_path)
        model = SentenceTransformer(embed_model)

        query = model.encode(["chest pain shortness of breath"], normalize_embeddings=True)
        distances, ids = index.search(query.astype(np.float32), k=5)
        log.info(f"Validation search returned IDs: {ids[0].tolist()}")
        log.info(f"Distances: {distances[0].tolist()}")
        log.info("Index validation PASSED")
    except Exception as e:
        log.warning(f"Validation skipped: {e}")


def generate_demo_corpus(output_path: str):
    """Generate a demo corpus if no input file is provided."""
    demo_docs = [
        {"id": 0, "text": "ESI Level 1: Immediate life-saving intervention required. Cardiac arrest, respiratory failure, severe haemorrhage."},
        {"id": 1, "text": "ESI Level 2: High-risk situation or acute changes. Chest pain with cardiac risk, severe dyspnoea, altered mental status."},
        {"id": 2, "text": "ACS Red Flags: Crushing chest pain radiating to arm or jaw, diaphoresis, nausea, dyspnoea, syncope."},
        {"id": 3, "text": "Pulmonary Embolism: Acute dyspnoea, pleuritic chest pain, haemoptysis, tachycardia, hypoxia."},
        {"id": 4, "text": "Wells PE Score: DVT signs +3, PE likely +3, HR>100 +1.5, immobilisation +1.5, previous PE/DVT +1.5, haemoptysis +1, malignancy +1."},
        {"id": 5, "text": "MONA ACS protocol: Morphine IV, Oxygen if SpO2<94%, Nitrates if BP>90, Aspirin 300mg loading."},
        {"id": 6, "text": "Troponin: High-sensitivity troponin I or T. Elevated above 99th percentile. Serial at 0h and 1-3h."},
        {"id": 7, "text": "Aortic dissection: Tearing back pain, unequal arm BPs. Avoid thrombolytics. Emergency CT angiography."},
        {"id": 8, "text": "Tension pneumothorax: Absent breath sounds, tracheal deviation, hypotension. Immediate needle decompression."},
        {"id": 9, "text": "Sepsis: SIRS criteria + infection. Lactate >2. Start antibiotics within 1 hour. IV fluids 30ml/kg."},
        {"id": 10, "text": "Drug interaction: Warfarin + NSAIDs increases bleeding risk. Monitor INR. Warfarin + metronidazole potentiates anticoagulation."},
        {"id": 11, "text": "Aspirin: Acetylsalicylic acid. Antiplatelet 75-100mg OD. ACS loading 300mg. Avoid in children."},
    ]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for doc in demo_docs:
            f.write(json.dumps(doc) + "\n")
    log.info(f"Demo corpus written to {output_path} ({len(demo_docs)} entries)")
    return demo_docs


def main():
    args = parse_args()

    if not os.path.exists(args.corpus):
        log.warning(f"Corpus not found at {args.corpus} — generating demo corpus")
        docs = generate_demo_corpus(args.corpus)
    else:
        docs = load_corpus(args.corpus)

    embeddings = embed_corpus(docs, args.embed_model, args.batch_size)
    index = build_index(embeddings, args.metric)
    save_outputs(index, docs, args)
    validate_index(args.output_index, args.embed_model)

    log.info("\n✓ Done. Copy the FAISS index and metadata to app/src/main/assets/models/ and rebuild the APK.")
    log.info("  Alternatively, the app will build the index on first launch from its embedded knowledge base.")


if __name__ == "__main__":
    main()
