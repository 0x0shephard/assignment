"""Generic hyperparameter sweep runner for Task C7 (§4.11).

Reuses the existing training scripts:
  train_dpo.py    — DPO β sweep
  train_ppo.py    — PPO β / ε / K sweeps
  train_grpo.py   — GRPO β / ε / K sweeps

Each run gets its own output directory `runs/<name>/<sweep_val>/` so we can
diff the resulting `log.json` / `summary.json`. After all runs finish, the
aggregator prints a summary table.

The four ablations from the manual:
  1. KL coefficient (PPO or GRPO):  --beta ∈ {0, 0.05, 0.1, 0.5}
  2. Clipping (PPO or GRPO):        --eps_clip ∈ {0.05, 0.2, 0.5, 1e9}  (last = unclipped)
  3. Group size K (GRPO):           --K ∈ {1, 2, 4, 8}, adjust prompts_per_step to keep 8·K_base RM calls per step constant
  4. DPO β:                         --beta ∈ {0.01, 0.1, 0.5, 1.0}

The runner is intentionally simple: it constructs argv and calls the entry
point via subprocess so each run is isolated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "dpo_beta": {
        "script": "train_dpo.py",
        "flag": "--beta",
        "values": [0.01, 0.1, 0.5, 1.0],
        "extra": ["--epochs", "1"],
    },
    "ppo_beta": {
        "script": "train_ppo.py",
        "flag": "--beta",
        "values": [0.0, 0.05, 0.1, 0.5],
        "extra": ["--steps", "200"],
    },
    "grpo_beta": {
        "script": "train_grpo.py",
        "flag": "--beta",
        "values": [0.0, 0.05, 0.1, 0.5],
        "extra": ["--steps", "200"],
    },
    "ppo_eps": {
        "script": "train_ppo.py",
        "flag": "--eps_clip",
        # 1e9 = effectively unclipped (CPI)
        "values": [0.05, 0.2, 0.5, 1e9],
        "extra": ["--steps", "200"],
    },
    "grpo_eps": {
        "script": "train_grpo.py",
        "flag": "--eps_clip",
        "values": [0.05, 0.2, 0.5, 1e9],
        "extra": ["--steps", "200"],
    },
    "grpo_K": {
        "script": "train_grpo.py",
        "flag": "--K",
        "values": [1, 2, 4, 8],
        # Keep 8*K_base = 32 RM calls per step constant:
        # prompts_per_step = 32 / K
        "extra_per_val": lambda k: ["--prompts_per_step", str(max(1, 32 // int(k))),
                                   "--steps", "200"],
    },
}


def _slug(v):
    return str(v).replace(".", "p").replace("+", "").replace("-", "n")


def run_sweep(name: str, out_root: Path, dry_run: bool = False, extra_argv=None):
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name}. options: {list(PRESETS)}")
    spec = PRESETS[name]
    results = []
    for v in spec["values"]:
        run_dir = out_root / name / _slug(v)
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, spec["script"], spec["flag"], str(v),
                "--out", str(run_dir)]
        if "extra" in spec:
            argv += list(spec["extra"])
        if "extra_per_val" in spec:
            argv += list(spec["extra_per_val"](v))
        if extra_argv:
            argv += list(extra_argv)
        print("\n===", name, "=", v, "===")
        print("$", " ".join(argv))
        if dry_run:
            results.append({"value": v, "cmd": argv, "dir": str(run_dir)})
            continue
        # stream stdout so we see progress
        rc = subprocess.call(argv)
        results.append({"value": v, "returncode": rc, "dir": str(run_dir)})

    (out_root / f"{name}_manifest.json").write_text(json.dumps(results, indent=2))
    return results


def summarize(sweep_dirs):
    """Print a small table across a completed sweep. Reads whichever of
    log.json / summary.json each run wrote."""
    rows = []
    for d in sweep_dirs:
        d = Path(d)
        log_path = d / "log.json"
        summary_path = d / "summary.json"
        row = {"run": str(d)}
        if log_path.exists():
            data = json.loads(log_path.read_text())
            log = data.get("log", data) if isinstance(data, dict) else data
            if log:
                last = log[-1]
                row.update({k: last.get(k) for k in
                            ("rm_mean", "kl_term", "eval_rm", "eval_kl",
                             "clip_frac", "mean_resp_len", "eval_pass1",
                             "eval_format_ok")})
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            for k in ("final_pref_acc", "final_response_ppl",
                      "final_test_pref_accuracy"):
                if k in s:
                    row[k] = s[k]
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    print("\n=== sweep summary ===")
    print(", ".join(keys))
    for r in rows:
        print(", ".join(f"{r.get(k, '')}" for k in keys))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS.keys()) + ["summarize"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--sweep_dirs", nargs="+",
                    help="For 'summarize' preset: directories to summarize.")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="Extra args forwarded to the training script "
                         "(everything after -- is passed through).")
    args = ap.parse_args()

    if args.preset == "summarize":
        if not args.sweep_dirs:
            raise SystemExit("--sweep_dirs required for summarize")
        summarize(args.sweep_dirs)
        return

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_sweep(args.preset, out_root, dry_run=args.dry_run, extra_argv=extra)


if __name__ == "__main__":
    main()
