"""Post-train LoRA adapter surgery for Nemotron-3-Nano submission.

Mirrors the 3 phases from `amplifying-top-50-singular-values.ipynb`:
  1. Force `target_modules` in adapter_config.json to the canonical vLLM names.
  2. Rename safetensors keys: `base_model.model.model` -> `base_model.model.backbone`.
  3a. Unfuse fused MoE experts `experts.w1` / `experts.w2` ->
      per-expert `experts.{i}.up_proj` / `experts.{i}.down_proj` LoRA pairs.
  3b. Fuse split Mamba `gate_proj` + `x_proj` LoRA pairs (rank R each) into a
      single fused `in_proj` LoRA (rank R) via QR-then-SVD (Eckart-Young),
      optionally boosting the top `--top-frac` of singular values by `--boost`.

CLI:
  python tools/postprocess_adapter.py \\
    --in  outputs/raft_adapter \\
    --out outputs/raft_adapter_processed \\
    --boost 1.12 --top-frac 0.5 \\
    [--rank 32] [--verify-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Canonical target_modules string list the eval / vLLM stack expects.
# Source: huikang `amplifying-top-50-singular-values.ipynb` cell 10.
CANONICAL_TARGET_MODULES = [
    "k_proj",
    "o_proj",
    "in_proj",
    "q_proj",
    "up_proj",
    "v_proj",
    "down_proj",
    "out_proj",
    "lm_head",
]


def rename_key(key: str) -> str:
    """Rename adapter key prefix to match vLLM's `backbone` namespace."""
    return key.replace("base_model.model.model", "base_model.model.backbone")


def _print_diff(label: str, a: set[str], b: set[str], head: int = 10) -> None:
    only_a = sorted(a - b)[:head]
    only_b = sorted(b - a)[:head]
    print(f"  [{label}] in_a_not_b={len(a - b)}  in_b_not_a={len(b - a)}")
    if only_a:
        print(f"    sample (a-b): {only_a}")
    if only_b:
        print(f"    sample (b-a): {only_b}")


def process(
    in_dir: Path,
    out_dir: Path,
    rank: int,
    boost: float,
    top_frac: float,
    verify_only: bool,
) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg_path = in_dir / "adapter_config.json"
    st_path = in_dir / "adapter_model.safetensors"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        return 2
    if not st_path.exists():
        print(f"ERROR: {st_path} not found", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text())
    original_targets = cfg.get("target_modules")
    print(f"== in_dir = {in_dir}")
    print(f"   adapter_config.target_modules (before) = {original_targets}")

    # Load all tensors
    adapter_tensors: dict[str, "torch.Tensor"] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            adapter_tensors[key] = f.get_tensor(key)
    in_keys = set(adapter_tensors.keys())
    print(f"   loaded {len(adapter_tensors)} tensors")

    # Collect adapter base names (strip .lora_A/.lora_B.weight)
    base_names: set[str] = set()
    for key in adapter_tensors:
        base_names.add(re.sub(r"\.lora_[AB]\.weight$", "", key))

    # Identify split Mamba pairs (gate_proj + x_proj per layer) to fuse -> in_proj
    mamba_merge: dict[str, dict[str, str]] = {}
    for base in base_names:
        for proj in ("gate_proj", "x_proj"):
            # Match only when proj is the final segment of the module path
            if base.endswith(f".{proj}"):
                layer_path = base.rsplit(f".{proj}", 1)[0]
                mamba_merge.setdefault(layer_path, {})[proj] = base
    # Only treat a layer as a Mamba merge target if BOTH gate_proj and x_proj
    # adapters are present.
    mamba_merge = {lp: m for lp, m in mamba_merge.items() if set(m) == {"gate_proj", "x_proj"}}
    mamba_merge_bases: set[str] = set()
    for projs in mamba_merge.values():
        mamba_merge_bases.update(projs.values())
    print(f"   detected {len(mamba_merge)} Mamba layers with split gate/x to fuse")

    # Build output tensors
    out: dict[str, "torch.Tensor"] = {}
    n_unfused_experts = 0
    n_passthrough = 0

    for base in sorted(base_names):
        lora_A = adapter_tensors[f"{base}.lora_A.weight"]
        lora_B = adapter_tensors[f"{base}.lora_B.weight"]
        renamed = rename_key(base)

        # Skip empty w3 expert placeholders (Nemotron MoE quirk)
        if ".experts.w3" in base and lora_A.numel() == 0:
            continue

        # Mamba split pairs handled separately below
        if base in mamba_merge_bases:
            continue

        # Unfuse expert tensors w1 -> up_proj, w2 -> down_proj per expert
        if ".experts.w1" in base or ".experts.w2" in base:
            # Broadcast singleton expert dim if one side has shape[0]==1
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

            num_experts = lora_A.shape[0]
            proj_name = "up_proj" if ".w1" in base else "down_proj"
            for i in range(num_experts):
                exp_renamed = re.sub(
                    r"\.experts\.w[12]",
                    f".experts.{i}.{proj_name}",
                    renamed,
                )
                out[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
                out[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
            n_unfused_experts += 1
            continue

        # Direct passthrough rename
        out[f"{renamed}.lora_A.weight"] = lora_A
        out[f"{renamed}.lora_B.weight"] = lora_B
        n_passthrough += 1

    print(f"   passthrough renames = {n_passthrough}")
    print(f"   expert unfusions    = {n_unfused_experts} (each expanded to N per-expert pairs)")

    # Mamba fuse: gate_proj + x_proj -> in_proj via QR-then-SVD
    n_fused = 0
    for layer_path, projs in sorted(mamba_merge.items()):
        renamed_layer = rename_key(layer_path)
        in_proj_base = f"{renamed_layer}.in_proj"

        gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()

        # Concatenated rank = 2 * adapter rank
        cat_rank = gate_A.shape[0] + x_A.shape[0]
        # in_proj output dim = gate_B rows + x_B rows
        in_proj_dim = gate_B.shape[0] + x_B.shape[0]

        # Stack A side along rank dim
        A_cat = torch.cat([gate_A, x_A], dim=0)  # (cat_rank, in_dim)
        # Block-diag B side
        B_block = torch.zeros(in_proj_dim, cat_rank, dtype=A_cat.dtype)
        B_block[: gate_B.shape[0], : gate_A.shape[0]] = gate_B
        B_block[gate_B.shape[0] :, gate_A.shape[0] :] = x_B

        Q_B, R_B = torch.linalg.qr(B_block)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        core = R_B @ R_A.T
        U, S, Vh = torch.linalg.svd(core, full_matrices=False)

        k = min(rank, S.shape[0])
        S_scaled = S[:k].clone()
        if boost != 1.0 and top_frac > 0:
            n_boost = max(1, int(k * top_frac))
            S_scaled[:n_boost] *= boost

        new_B = (Q_B @ U[:, :k]) * S_scaled.unsqueeze(0)
        new_A = Vh[:k, :] @ Q_A.T

        out[f"{in_proj_base}.lora_A.weight"] = new_A
        out[f"{in_proj_base}.lora_B.weight"] = new_B

        kept = float(S_scaled.sum())
        total = float(S.sum())
        print(
            f"   fused {layer_path}: kept rank-{k} singular-value mass "
            f"{kept:.2f}/{total:.2f} ({100 * kept / max(total, 1e-9):.1f}%)"
        )
        n_fused += 1
    print(f"   mamba fusions = {n_fused}")

    out_keys = set(out.keys())
    print(f"   output tensors = {len(out)}")
    _print_diff("rename-only diff vs input (post-rename)", in_keys, out_keys)

    if verify_only:
        print("verify-only: not writing anything.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    # Patch + write config
    cfg["target_modules"] = list(CANONICAL_TARGET_MODULES)
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    # Write tensors
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    print(f"== wrote {out_dir / 'adapter_config.json'}")
    print(f"== wrote {out_dir / 'adapter_model.safetensors'}  "
          f"({(out_dir / 'adapter_model.safetensors').stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   adapter_config.target_modules (after) = {cfg['target_modules']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True, type=Path,
                   help="input adapter dir (contains adapter_config.json + adapter_model.safetensors)")
    p.add_argument("--out", dest="out_dir", required=True, type=Path,
                   help="output dir for the processed adapter (created)")
    p.add_argument("--rank", type=int, default=32,
                   help="competition LoRA rank cap (default 32)")
    p.add_argument("--boost", type=float, default=1.0,
                   help="multiplier applied to top --top-frac singular values during Mamba fuse "
                        "(1.0 = no boost; reference notebook used 1.12)")
    p.add_argument("--top-frac", type=float, default=0.5,
                   help="fraction of singular values to boost when --boost != 1.0")
    p.add_argument("--verify-only", action="store_true",
                   help="do all the work, print diffs, but do not write outputs")
    args = p.parse_args()

    if not args.in_dir.is_dir():
        print(f"ERROR: --in {args.in_dir} is not a directory", file=sys.stderr)
        return 2
    return process(args.in_dir, args.out_dir, args.rank, args.boost, args.top_frac, args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
