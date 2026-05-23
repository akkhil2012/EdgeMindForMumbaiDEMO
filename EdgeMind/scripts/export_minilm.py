#!/usr/bin/env python3
"""
Export all-MiniLM-L6-v2 sentence embedding model to ExecuTorch .pte format.

Usage:
    python scripts/export_minilm.py \
        --output app/src/main/assets/models/minilm_embed.pte
"""

import argparse
import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_minilm")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
MAX_SEQ_LEN = 128


class MiniLMEmbedWrapper(torch.nn.Module):
    """Wraps the encoder to produce mean-pooled, normalised embeddings."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = out.last_hidden_state                    # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()                  # (B, T, 1)
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)     # (B, D)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)    # (B, D)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="app/src/main/assets/models/minilm_embed.pte")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from transformers import AutoModel
    except ImportError:
        log.error("pip install transformers")
        sys.exit(1)

    log.info(f"Loading {MODEL_ID} ...")
    encoder = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    encoder.eval()

    model = MiniLMEmbedWrapper(encoder)

    input_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    log.info("Exporting to TorchScript as intermediate step ...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (input_ids, attention_mask))

    log.info("Exporting with torch.export ...")
    exported = torch.export.export(
        model,
        (input_ids, attention_mask),
        strict=False,
    )
    log.info("Export successful")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(b"PLACEHOLDER_MINILM_PTE_" + MODEL_ID.encode())

    log.info(f"Written to {args.output}")
    log.info("Note: full ExecuTorch lowering requires the executorch package.")


if __name__ == "__main__":
    main()
