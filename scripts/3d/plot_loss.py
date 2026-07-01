"""Plot the training loss curve from an exp7 tee'd log.

Parses the HF Trainer `{'loss': ..., 'epoch': ...}` lines (logging_steps=1 -> one per
optimizer step) and saves a PNG. Re-runnable mid-training.

Usage:
  python scripts/3d/plot_loss.py --log ckpt/exp7-scanqa-dinov3-fuse.log
"""
import argparse
import ast
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(log_path):
    steps, losses, epochs = [], [], []
    line_re = re.compile(r"\{.*'loss':.*\}")
    for line in open(log_path, errors="ignore"):
        line = line.strip()
        m = line_re.search(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        if "loss" in d and "train_loss" not in d:  # skip the final summary line
            losses.append(float(d["loss"]))
            epochs.append(float(d.get("epoch", 0)))
            steps.append(len(losses))
    return steps, losses, epochs


def ema(x, alpha=0.05):
    out, m = [], None
    for v in x:
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="ckpt/exp7-scanqa-dinov3-fuse.log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.log.rsplit(".", 1)[0] + "_loss.png"

    steps, losses, epochs = parse(args.log)
    if not losses:
        raise SystemExit(f"no loss lines found in {args.log}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, lw=0.6, alpha=0.35, color="#1f77b4", label="loss (per step)")
    ax.plot(steps, ema(losses), lw=2.0, color="#d62728", label="EMA (α=0.05)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss")
    ax.set_title(f"exp7 ScanQA + DINOv3 fusion — {len(losses)} steps, {epochs[-1]:.2f} epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # epoch boundaries as light vlines
    for e in range(1, int(epochs[-1]) + 1):
        xs = [s for s, ep in zip(steps, epochs) if ep >= e]
        if xs:
            ax.axvline(xs[0], color="gray", ls="--", lw=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"final loss {losses[-1]:.4f} | min {min(losses):.4f} | steps {len(losses)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
