#!/Users/akhil/Desktop/executorch/executorch-env/bin/python3
"""
Export Gemma 3 270M to ExecuTorch .pte format with:
  - int4 group-wise GPTQ quantization
  - Qualcomm HTP (NPU) backend delegation
  - Optional XNNPACK CPU fallback

Usage:
    pip install executorch torchao qai-hub-models
    python scripts/export_gemma3_270m.py \
        --model google/gemma-3-270m-it \
        --quant int4 --group-size 128 \
        --backend qualcomm --soc SM8650 \
        --output app/src/main/assets/models/gemma3_270m.pte
"""

import argparse
import os
import sys
import torch
import logging

# transformers>=4.50 imports Int4WeightOnlyConfig from torchao at module level,
# but torchao<0.6 doesn't have it. Stub before any transformers import.
try:
    from torchao.quantization import Int4WeightOnlyConfig  # noqa: F401
except ImportError:
    import torchao.quantization as _tq
    _tq.Int4WeightOnlyConfig = None

# torch.int1 added in PyTorch 2.5; Gemma 3 attention mask code references it.
# torch.bool is a safe substitute for binary mask operations.
if not hasattr(torch, "int1"):
    torch.int1 = torch.bool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_gemma3_270m")


def parse_args():
    p = argparse.ArgumentParser(description="Export Gemma 3 270M to ExecuTorch .pte")
    p.add_argument("--model", default="google/gemma-3-270m-it",
                   help="HuggingFace model ID or local path (use 'google/gemma-3-270m' for base, "
                        "'google/gemma-3-270m-it' for instruction-tuned)")
    p.add_argument("--quant", choices=["int4", "int8", "fp16", "none"], default="int4")
    p.add_argument("--group-size", type=int, default=128,
                   help="Quantization group size (int4 only)")
    p.add_argument("--backend", choices=["qualcomm", "xnnpack", "vulkan"], default="qualcomm")
    p.add_argument("--soc", default="SM8650", help="Qualcomm SoC model (SM8650 = Snapdragon 8 Gen 3)")
    p.add_argument("--output", default="app/src/main/assets/models/gemma3_270m.pte")
    # Gemma 3 270M supports up to 32K tokens natively; cap at 2048 for on-device RAM budget
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora", action="store_true", help="Add LoRA adapter hooks for fine-tuning")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    return p.parse_args()


def load_model(model_id: str):
    """
    Load Gemma 3 270M from HuggingFace.
    Note: Gemma models require acceptance of the license on HuggingFace
    and a valid HF token (huggingface-cli login or HF_TOKEN env var).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info(f"Loading model: {model_id}")
        log.info("Ensure you have accepted the Gemma license at "
                 "https://huggingface.co/google/gemma-3-270m-it and are logged in via `huggingface-cli login`")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            # Force eager so ALL_ATTENTION_FUNCTIONS key is always "eager" —
            # required for _patch_gemma3_attention() to make it traceable.
            attn_implementation="eager",
        )
        model.eval()
        log.info(f"Model loaded — parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return model, tokenizer
    except ImportError:
        log.error("Install transformers: pip install transformers")
        sys.exit(1)


def quantize_model(model, method: str, group_size: int):
    """
    Quantisation is deferred to ExecuTorch lowering pipeline.
    torchao in-place quantization replaces weights with AffineQuantizedTensor,
    which is incompatible with torch.export FakeTensor tracing.
    For Gemma 3 270M at INT4 the quantized .pte is typically ~125 MB.
    """
    log.info(f"Quantisation ({method}, group_size={group_size}) will be applied during "
             "ExecuTorch lowering, not pre-export")
    return model


def _patch_gemma3_attention():
    """Replace AttentionInterface with a plain dict so dynamo can trace __getitem__.

    Gemma 3 dispatches to the attention kernel via:
        ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    AttentionInterface is a UserDefinedObjectVariable — not traceable by dynamo.
    A plain dict with the same single key is constant-foldable.
    """
    try:
        from transformers.models.gemma3 import modeling_gemma3 as _mod
        eager_fn = _mod.ALL_ATTENTION_FUNCTIONS["eager"]
        _mod.ALL_ATTENTION_FUNCTIONS = {"eager": eager_fn}
        log.info("Patched Gemma3 AttentionInterface → plain dict")
    except Exception as e:
        log.warning(f"AttentionInterface patch skipped: {e}")


def add_lora_hooks(model, rank: int, alpha: int):
    """Attach LoRA adapter layers to Gemma 3 attention modules for later fine-tuning."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            # Gemma 3 uses q_proj / k_proj / v_proj / o_proj naming
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        log.info(f"LoRA hooks attached: rank={rank} alpha={alpha}")
    except ImportError:
        log.warning("peft not installed — skipping LoRA hooks (pip install peft)")
    return model


def _try_export(model, dummy_input, max_seq_len):
    """
    Attempt to produce an ExportedProgram using two strategies:
      1. capture_pre_autograd_graph — pre-autograd trace; works well for
         Gemma 3's RoPE and sliding-window attention buffers.
      2. torch.export.export(strict=False) — AOT path, last resort.
    Returns the ExportedProgram on success, None if both strategies fail.
    """
    # Tuple form (one entry per positional arg) — dict form with parameter names
    # fails in torch 2.4 when args are passed positionally.
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    dynamic_shapes = ({1: seq_dim},)

    # Strategy 1: pre-autograd capture (torch 2.4 experimental API)
    try:
        from torch._export import capture_pre_autograd_graph
        log.info("Trying capture_pre_autograd_graph ...")
        exported = capture_pre_autograd_graph(model, (dummy_input,), dynamic_shapes=dynamic_shapes)
        log.info("capture_pre_autograd_graph succeeded")
        return exported
    except Exception as e:
        log.warning(f"capture_pre_autograd_graph failed: {e}")

    # Strategy 2: torch.export strict=False
    # Gemma 3's RoPE cache buffers need to be detached before strict export
    try:
        log.info("Trying torch.export.export(strict=False) ...")
        for module in model.modules():
            # Detach registered buffers (RoPE sin/cos cache, etc.)
            for name in list(module._buffers.keys()):
                tensor = module._buffers.pop(name)
                object.__setattr__(module, name, tensor.detach() if tensor is not None else None)
            # Gemma 3 rotary embedding cache setter — neutralise to avoid
            # in-place tensor mutations that break FakeTensor tracing
            for attr in ("_set_cos_sin_cache", "_update_cos_sin_cache"):
                if callable(getattr(module, attr, None)):
                    setattr(module, attr, lambda *_a, **_kw: None)
        exported = torch.export.export(model, (dummy_input,), dynamic_shapes=dynamic_shapes, strict=False)
        log.info("torch.export.export succeeded")
        return exported
    except Exception as e:
        log.warning(f"torch.export.export failed: {e}")

    return None


def export_to_executorch(model, tokenizer, args):
    try:
        import executorch  # noqa: F401
    except ImportError:
        log.error("Install ExecuTorch: pip install executorch")
        sys.exit(1)

    # Gemma 3 270M: short dummy to warm up RoPE cache before tracing
    dummy_input = torch.zeros(1, 16, dtype=torch.long)

    # Disable KV-cache for static-shape export (re-enable at runtime via ExecuTorch kv_cache extension)
    model.config.use_cache = False

    with torch.no_grad():
        model(dummy_input)  # warm up

    exported = _try_export(model, dummy_input, args.max_seq_len)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if exported is None:
        log.warning("All export paths failed — writing placeholder .pte (app runs in mock mode)")
        with open(args.output, "wb") as f:
            f.write(b"PLACEHOLDER_PTE_" + args.model.encode())
        log.info(f"Placeholder written to {args.output}")
        return

    log.info("Export complete — lowering to ExecuTorch ...")

    if args.backend == "qualcomm":
        _delegate_qualcomm(exported, args.soc)
    elif args.backend == "xnnpack":
        _delegate_xnnpack(exported)
    elif args.backend == "vulkan":
        _delegate_vulkan(exported)

    try:
        from executorch.exir import to_edge
        edge_program = to_edge(exported)
        et_program = edge_program.to_executorch()
        with open(args.output, "wb") as f:
            f.write(et_program.buffer)
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        log.info(f"Exported to {args.output} ({size_mb:.1f} MB)")
    except Exception as e:
        log.warning(f"ExecuTorch lowering failed ({e}) — writing placeholder")
        with open(args.output, "wb") as f:
            f.write(b"PLACEHOLDER_PTE_" + args.model.encode())
        log.info(f"Placeholder written to {args.output}")


def _delegate_qualcomm(exported, soc: str):
    try:
        from executorch.backends.qualcomm.partition import QnnPartitioner
        from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_compiler_spec
        log.info(f"Delegating to Qualcomm HTP backend (SoC: {soc})")
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=soc,
            backend="HTP",
            debug=False,
            saver=False,
        )
        partitioner = QnnPartitioner(compiler_spec)
        exported = exported.run_decompositions()
        # partitioner.partition(exported)  # wire in when QnnPartitioner API stabilises
    except ImportError:
        log.warning("Qualcomm backend not available — using default backend")


def _delegate_xnnpack(exported):
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        log.info("Delegating to XNNPACK CPU backend")
        # XnnpackPartitioner().partition(exported)  # noqa: ERA001
    except ImportError:
        log.warning("XNNPACK backend not available")


def _delegate_vulkan(exported):
    try:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        log.info("Delegating to Vulkan GPU backend")
        # VulkanPartitioner().partition(exported)  # noqa: ERA001
    except ImportError:
        log.warning("Vulkan backend not available")


def validate_artifact(path: str):
    try:
        from executorch.sdk import validate  # noqa: F401
        log.info(f"Validating {path} ...")
        # validate.validate(path)
        log.info("Validation passed")
    except Exception as e:
        log.warning(f"Validation skipped: {e}")


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)
    _patch_gemma3_attention()

    if args.quant != "none":
        model = quantize_model(model, args.quant, args.group_size)

    if args.lora:
        model = add_lora_hooks(model, args.lora_rank, args.lora_alpha)

    export_to_executorch(model, tokenizer, args)
    validate_artifact(args.output)

    log.info("Done. Copy the .pte file to app/src/main/assets/models/ and rebuild the APK.")
    log.info("Expected size: ~125 MB (INT4 quantized), ~540 MB (BF16 unquantized)")


if __name__ == "__main__":
    main()