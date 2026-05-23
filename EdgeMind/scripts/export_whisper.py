#!/usr/bin/env python3
"""
Export Whisper Small to ExecuTorch .pte format.

Usage:
    python scripts/export_whisper.py \
        --model openai/whisper-small \
        --backend qualcomm \
        --output app/src/main/assets/models/whisper_small.pte
"""

import argparse
import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_whisper")


def parse_args():
    p = argparse.ArgumentParser(description="Export Whisper to ExecuTorch .pte")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--backend", choices=["qualcomm", "xnnpack"], default="qualcomm")
    p.add_argument("--output", default="app/src/main/assets/models/whisper_small.pte")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        log.error("pip install transformers")
        sys.exit(1)

    log.info(f"Loading Whisper: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    # Export encoder and decoder separately for streaming on-device.
    # float32 avoids the dtype guard in WhisperEncoderLayer.forward that
    # torch.export cannot resolve; quantisation is handled by the lowering step.
    log.info("Exporting encoder ...")
    mel_input = torch.zeros(1, 80, 3000, dtype=torch.float32)
    encoder_exported = torch.export.export(model.get_encoder(), (mel_input,), strict=False)

    log.info("Exporting decoder ...")
    enc_out = torch.zeros(1, 1500, model.config.d_model, dtype=torch.float32)
    dec_ids = torch.zeros(1, 4, dtype=torch.long)
    cache_position = torch.arange(4, dtype=torch.long)

    # transformers 4.44 _update_causal_mask broadcasts cross-attention shapes
    # that torch.export cannot reconcile. Patch it out for tracing — attention
    # layers handle causal_mask=None correctly, and use_cache=False gives a
    # fully static graph with no KV-cache dynamic shapes.
    decoder = model.get_decoder()
    decoder._update_causal_mask = lambda *args, **kwargs: None

    decoder_exported = torch.export.export(
        decoder,
        (dec_ids, enc_out),
        kwargs={"cache_position": cache_position, "use_cache": False},
        strict=False,
    )

    # In production: lower to ExecuTorch with Qualcomm HTP or XNNPACK backend
    # For now, write placeholder .pte files
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(b"PLACEHOLDER_WHISPER_PTE_" + args.model.encode())

    log.info(f"Written to {args.output}")
    log.info("Note: Full ExecuTorch lowering requires executorch package. Placeholder written.")


if __name__ == "__main__":
    main()
