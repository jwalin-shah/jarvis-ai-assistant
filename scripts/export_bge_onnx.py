#!/usr/bin/env python3
"""Export BGE model to ONNX format for CPU inference.

This creates a lightweight ONNX model that can be used by CPUEmbedder.
Much simpler than using optimum-cli.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModel, AutoTokenizer


def export_bge_onnx():
    """Export BGE-small to ONNX format."""
    model_name = "BAAI/bge-small-en-v1.5"
    output_dir = Path("models/bge-small-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    dummy_text = "This is a test"
    inputs = tokenizer(
        dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]

    # Export
    print("Exporting to ONNX...")
    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"},
        },
        opset_version=14,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Exported to {onnx_path}")
    print(f"  Model size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Tokenizer saved to {output_dir}")

    # Test
    print("\nTesting ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_texts = ["Hello world", "Test sentence"]
    test_inputs = tokenizer(test_texts, return_tensors="np", padding=True, truncation=True)

    outputs = session.run(
        None,
        {
            "input_ids": test_inputs["input_ids"],
            "attention_mask": test_inputs["attention_mask"],
        },
    )

    print("✓ Test successful")
    print(f"  Output shape: {outputs[0].shape}")

    return True


if __name__ == "__main__":
    try:
        export_bge_onnx()
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
