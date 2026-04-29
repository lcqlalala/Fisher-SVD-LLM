#!/usr/bin/env python3
# coding: utf-8
"""
Load a saved Fisher-SVD checkpoint and print per-layer compression ratios.

Example:
  python print_layer_compression_from_checkpoint.py \
      --ckpt /path/to/MODEL_ID_fisher_svd_0.6.pt
"""

import argparse
import re
from collections import defaultdict
from typing import Dict, Tuple

import torch

from fisher_svd import SVDLinear, SVDLinearWithDenseBlocks


def _load_checkpoint(ckpt_path: str):
    # Compatible with different torch versions / defaults.
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    return obj


def _extract_layer_idx(module_name: str) -> int:
    # LLaMA/Mistral style: model.layers.{idx}.*
    m = re.search(r"\.layers\.(\d+)\.", module_name)
    if m:
        return int(m.group(1))
    # OPT style: model.decoder.layers.{idx}.*
    m = re.search(r"\.decoder\.layers\.(\d+)\.", module_name)
    if m:
        return int(m.group(1))
    return -1


def _module_compression_params(module) -> Tuple[int, int]:
    """
    Returns:
      (orig_params, compressed_params)

    Definition aligned with Fisher-SVD logs (weight matrices only, bias excluded):
      original: out_features * in_features
      compressed low-rank: rank * (in_features + out_features)
      plus residual block params if present
    """
    if isinstance(module, (SVDLinear, SVDLinearWithDenseBlocks)):
        v_proj = module.v_proj
        u_proj = module.u_proj
        in_features = v_proj.in_features
        rank = v_proj.out_features
        out_features = u_proj.out_features

        original = out_features * in_features
        compressed = rank * (in_features + out_features)

        # Add residual block params for SVDLinearWithDenseBlocks.
        if isinstance(module, SVDLinearWithDenseBlocks):
            block_params = 0
            for gi in range(module.num_groups):
                blocks_t = getattr(module, f"g{gi}_blocks_T")
                block_params += blocks_t.numel()
            compressed += block_params

        return original, compressed

    return 0, 0


def print_layer_compression(model) -> None:
    layer_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {
        "orig": 0,
        "comp": 0,
        "num_proj": 0,
    })

    for name, module in model.named_modules():
        if not isinstance(module, (SVDLinear, SVDLinearWithDenseBlocks)):
            continue

        layer_idx = _extract_layer_idx(name)
        if layer_idx < 0:
            # Ignore non-transformer-layer modules.
            continue

        orig, comp = _module_compression_params(module)
        layer_stats[layer_idx]["orig"] += orig
        layer_stats[layer_idx]["comp"] += comp
        layer_stats[layer_idx]["num_proj"] += 1

    if not layer_stats:
        print("No SVDLinear/SVDLinearWithDenseBlocks modules found in checkpoint model.")
        return

    total_orig = 0
    total_comp = 0

    print("Layer-wise compression ratios:")
    for layer_idx in sorted(layer_stats.keys()):
        orig = layer_stats[layer_idx]["orig"]
        comp = layer_stats[layer_idx]["comp"]
        nproj = layer_stats[layer_idx]["num_proj"]
        total_orig += orig
        total_comp += comp

        ratio = comp / max(1, orig)
        reduction = 1.0 - ratio
        print(
            f"  L{layer_idx:02d}: projections={nproj}, "
            f"compressed/original={comp:,}/{orig:,} ({ratio:.2%}), "
            f"reduction={reduction:.2%}"
        )

    total_ratio = total_comp / max(1, total_orig)
    total_reduction = 1.0 - total_ratio
    print("Total:")
    print(
        f"  compressed/original={total_comp:,}/{total_orig:,} "
        f"({total_ratio:.2%}), reduction={total_reduction:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Print per-layer compression ratios from Fisher-SVD checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to saved checkpoint .pt")
    args = parser.parse_args()

    obj = _load_checkpoint(args.ckpt)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError("Checkpoint format not recognized: expected dict with key 'model'.")

    model = obj["model"]
    print_layer_compression(model)


if __name__ == "__main__":
    main()

