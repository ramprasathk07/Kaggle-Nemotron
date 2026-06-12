#!/usr/bin/env python3
"""
lora_merge.py — merge LoRA adapters for the Nemotron Reasoning Challenge.

Merges N LoRA adapters trained on the SAME base model (your SFT/GRPO/RAFT
runs, category-expert adapters, public adapters) WITHOUT loading the 30B
base model and WITHOUT torch: pure numpy on CPU, with a built-in
safetensors reader/writer (F32/F16/BF16/F64).

Method: for each adapted module, reconstruct the full-rank update
ΔW_i = scale_i * B_i @ A_i, combine the ΔW_i (linear soup, TIES, or DARE),
then re-factorize the merged ΔW back to a uniform rank-r adapter
(default 32 — the competition cap) via truncated SVD. Output alpha is set
equal to rank so the effective scale is exactly 1.0 and all scaling is
folded into the factors.

Why delta space: averaging A and B matrices directly is wrong for
independently-trained adapters (avg(B)avg(A) != avg(BA)); concatenating
ranks violates the rank<=32 rule. SVD on the combined ΔW is the correct,
constraint-respecting merge, and the per-module "energy" report tells you
how lossy the rank-32 truncation was (energy ~1.0 = lossless).

USAGE
  python3 lora_merge.py \
      --adapter runs/sft_best:0.5 --adapter runs/grpo:0.3 --adapter runs/raft:0.2 \
      --mode ties --density 0.5 --rank 32 --out merged_ties/

  modes: linear | ties | dare_linear | dare_ties
  ALWAYS evaluate the merged adapter locally (official-metric replica)
  before submitting. Merges are hypotheses, not upgrades.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------
# Minimal safetensors I/O (read F32/F16/BF16/F64 -> float32; write F32/BF16)
# --------------------------------------------------------------------------

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def _decode(dtype: str, buf: bytes, shape) -> np.ndarray:
    if dtype == "F32":
        arr = np.frombuffer(buf, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(buf, dtype="<f2").astype(np.float32)
    elif dtype == "F64":
        arr = np.frombuffer(buf, dtype="<f8").astype(np.float32)
    elif dtype == "BF16":
        u16 = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"unsupported safetensors dtype {dtype}")
    return np.ascontiguousarray(arr.astype(np.float32)).reshape(shape)


def _encode(arr: np.ndarray, dtype: str) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if dtype == "F32":
        return a.astype("<f4").tobytes()
    if dtype == "BF16":
        u32 = a.view(np.uint32)
        return ((u32 >> 16).astype("<u2")).tobytes()
    raise ValueError(f"unsupported output dtype {dtype}")


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n).decode("utf-8"))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = _decode(meta["dtype"], data[s:e], meta["shape"])
    return out


def save_safetensors(path: str, tensors: dict[str, np.ndarray],
                     dtype: str = "F32") -> None:
    header: dict = {"__metadata__": {"format": "pt"}}
    blobs, off = [], 0
    for name in tensors:  # contiguous, in insertion order
        b = _encode(tensors[name], dtype)
        header[name] = {"dtype": dtype, "shape": list(tensors[name].shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(hj).to_bytes(8, "little"))
        f.write(hj)
        for b in blobs:
            f.write(b)


# --------------------------------------------------------------------------
# Adapter loading / key normalization
# --------------------------------------------------------------------------

_LORA_RE = re.compile(r"^(?P<path>.+?)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")


def canon_id(module_path: str) -> str:
    """Stable id for matching the same module across export styles."""
    p = module_path
    for prefix in ("base_model.model.", "base_model."):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return p


def load_adapter(adir: str) -> dict:
    cfg_path = os.path.join(adir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg.get("lora_alpha", cfg["r"])
    scale = alpha / (np.sqrt(r) if cfg.get("use_rslora") else r)
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        print(f"  WARNING {adir}: per-module rank/alpha patterns present; "
              "using global r/alpha for all modules.", file=sys.stderr)

    st_path = os.path.join(adir, "adapter_model.safetensors")
    tensors = load_safetensors(st_path)
    modules: dict[str, dict] = {}
    for key, arr in tensors.items():
        m = _LORA_RE.match(key)
        if not m:
            print(f"  WARNING {adir}: skipping non-LoRA tensor '{key}' "
                  "(modules_to_save are not merged — avoid them for this comp).",
                  file=sys.stderr)
            continue
        cid = canon_id(m.group("path"))
        slot = modules.setdefault(cid, {"orig_path": m.group("path")})
        slot["A" if m.group("ab") == "A" else "B"] = arr
    for cid, slot in modules.items():
        if "A" not in slot or "B" not in slot:
            raise ValueError(f"{adir}: module {cid} missing lora_A or lora_B")
    return {"dir": adir, "config": cfg, "scale": float(scale), "modules": modules}


# --------------------------------------------------------------------------
# Combination rules
# --------------------------------------------------------------------------


def trim_topk(delta: np.ndarray, density: float) -> np.ndarray:
    if density >= 1.0:
        return delta
    k = int(np.ceil(density * delta.size))
    if k <= 0:
        return np.zeros_like(delta)
    flat = np.abs(delta).ravel()
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    return np.where(np.abs(delta) >= thresh, delta, 0.0)


def dare_drop(delta: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    if p <= 0.0:
        return delta
    keep = rng.random(delta.shape) >= p
    return delta * keep / (1.0 - p)


def combine(deltas: list[np.ndarray], weights: list[float], mode: str,
            density: float, drop_p: float, rng: np.random.Generator) -> np.ndarray:
    if mode.startswith("dare"):
        deltas = [dare_drop(d, drop_p, rng) for d in deltas]
    if mode.endswith("ties"):
        deltas = [trim_topk(d, density) for d in deltas]
        total = sum(w * d for w, d in zip(weights, deltas))
        elect = np.sign(total)
        num = np.zeros_like(total)
        den = np.zeros_like(total)
        for w, d in zip(weights, deltas):
            m = (np.sign(d) == elect) & (d != 0)
            num += w * d * m
            den += w * m
        merged = num / np.where(den == 0, 1.0, den)
        merged[den == 0] = 0.0
        return merged
    return sum(w * d for w, d in zip(weights, deltas))  # linear


def truncated_svd(M: np.ndarray, rank: int, rng: np.random.Generator):
    """Return A' (rank,in), B' (out,rank), energy in [0,1]."""
    out_d, in_d = M.shape
    k = min(rank, out_d, in_d)
    if min(out_d, in_d) <= max(4 * k, 128):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    else:  # randomized range finder (Halko et al.)
        G = rng.standard_normal((in_d, k + 10)).astype(np.float32)
        Q, _ = np.linalg.qr(M @ G)
        U2, s, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)
        U = Q @ U2
    fro2 = float((M * M).sum())
    energy = float((s[:k] ** 2).sum() / fro2) if fro2 > 0 else 1.0
    sq = np.sqrt(s[:k]).astype(np.float32)
    Bp = (U[:, :k] * sq).astype(np.float32)
    Ap = (sq[:, None] * Vt[:k]).astype(np.float32)
    if k < rank:  # zero-pad so every module has the same uniform rank
        Ap = np.concatenate([Ap, np.zeros((rank - k, in_d), np.float32)], axis=0)
        Bp = np.concatenate([Bp, np.zeros((out_d, rank - k), np.float32)], axis=1)
    return Ap, Bp, energy


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def merge(adapter_specs: list[tuple[str, float]], mode: str, density: float,
          drop_p: float, rank: int, out_dir: str, seed: int,
          normalize: bool = True, save_dtype: str = "F32") -> dict:
    rng = np.random.default_rng(seed)
    adapters = [load_adapter(d) for d, _ in adapter_specs]
    weights = [w for _, w in adapter_specs]
    if normalize:
        tot = sum(weights)
        weights = [w / tot for w in weights]

    all_ids: list[str] = []
    for ad in adapters:
        for cid in ad["modules"]:
            if cid not in all_ids:
                all_ids.append(cid)
    print(f"Merging {len(adapters)} adapters | mode={mode} rank={rank} "
          f"weights={['%.3f' % w for w in weights]} | {len(all_ids)} modules")

    out_tensors: dict[str, np.ndarray] = {}
    energies: list[tuple[str, float]] = []
    for i, cid in enumerate(all_ids):
        deltas, ws, orig_path = [], [], None
        for ad, w in zip(adapters, weights):
            slot = ad["modules"].get(cid)
            if slot is None:
                continue
            if orig_path is None:
                orig_path = slot["orig_path"]
            deltas.append(ad["scale"] * (slot["B"].astype(np.float32)
                                         @ slot["A"].astype(np.float32)))
            ws.append(w)
        merged = combine(deltas, ws, mode, density, drop_p, rng)
        Ap, Bp, energy = truncated_svd(merged, rank, rng)
        out_tensors[f"{orig_path}.lora_A.weight"] = Ap
        out_tensors[f"{orig_path}.lora_B.weight"] = Bp
        energies.append((cid, energy))
        if (i + 1) % 25 == 0 or i + 1 == len(all_ids):
            print(f"  [{i + 1}/{len(all_ids)}] modules merged")

    os.makedirs(out_dir, exist_ok=True)
    save_safetensors(os.path.join(out_dir, "adapter_model.safetensors"),
                     out_tensors, dtype=save_dtype)

    cfg = dict(adapters[0]["config"])
    cfg["r"] = rank
    cfg["lora_alpha"] = rank          # -> effective scale exactly 1.0
    cfg["use_rslora"] = False
    cfg.pop("rank_pattern", None)
    cfg.pop("alpha_pattern", None)
    cfg["target_modules"] = sorted({cid.rsplit(".", 1)[-1] for cid in all_ids})
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    evals = sorted(energies, key=lambda t: t[1])
    report = {"modules": len(all_ids),
              "energy_mean": float(np.mean([e for _, e in energies])),
              "energy_min": float(evals[0][1]),
              "worst_5": evals[:5]}
    print(f"\nDone -> {out_dir}")
    print(f"  rank-{rank} energy: mean={report['energy_mean']:.4f} "
          f"min={report['energy_min']:.4f}")
    for cid, e in report["worst_5"]:
        print(f"    lossiest: {cid}  energy={e:.4f}")
    print("  (energy << 1.0 means rank truncation discarded signal — consider "
          "fewer/closer adapters or check those modules' categories in eval)")
    return report


def _parse_spec(s: str) -> tuple[str, float]:
    if ":" in s:
        path, w = s.rsplit(":", 1)
        return path, float(w)
    return s, 1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", action="append", required=True,
                    metavar="DIR[:WEIGHT]", help="repeatable; weight defaults to 1")
    ap.add_argument("--mode", default="ties",
                    choices=["linear", "ties", "dare_linear", "dare_ties"])
    ap.add_argument("--density", type=float, default=0.5,
                    help="TIES trim: fraction of largest-|delta| params kept")
    ap.add_argument("--drop-p", type=float, default=0.3,
                    help="DARE random drop probability")
    ap.add_argument("--rank", type=int, default=32,
                    help="output rank (competition cap: 32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-normalize", action="store_true",
                    help="use weights as given instead of normalizing to sum 1")
    args = ap.parse_args()

    if args.rank > 32:
        print("WARNING: rank > 32 violates the competition constraint.",
              file=sys.stderr)
    merge([_parse_spec(s) for s in args.adapter], args.mode, args.density,
          args.drop_p, args.rank, args.out, args.seed,
          normalize=not args.no_normalize)
