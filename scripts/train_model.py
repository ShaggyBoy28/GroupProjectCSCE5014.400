"""Train the activity LSTM and evaluate it with cross-validation and quality/scale breakdowns.

Usage (project folder, .venv active, dataset zip in data/):

    # Retrain on the original train/test split (checks the training code reproduces the shipped model)
    python scripts/train_model.py standard

    # Block cross-validation: k folds of contiguous videos per class (≈ unseen people/views)
    python scripts/train_model.py cv --folds 5

    # Also run the same CV with random window folds, to show how much a leaky split inflates scores
    python scripts/train_model.py cv --folds 5 --compare-random

    # Add new recordings / classes (made with scripts/extract_poses.py); repeat --extra per index
    python scripts/train_model.py cv --extra data/extracted/index.csv

    # Train the model to ship on everything (train + test + extra) and save models/training/final.ckpt
    python scripts/train_model.py final --extra data/extracted/index.csv

Outputs go to models/training/ (git-ignored): checkpoints and a JSON report per run.
Training runs on the CPU; the model is small (~17.5k parameters).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*LightningModule.*hparams.*")

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.lstm import ActionClassificationLSTM  # noqa: E402
from src.metrics import calculate_classification_metrics  # noqa: E402
from src.pose_dataset import (DEFAULT_ZIP, LABELS, block_folds, load_extracted, load_split,  # noqa: E402
                              missing_joint_share, person_height, video_ids)

OUT_DIR = ROOT / "models" / "training"


def loader(windows, labels, batch_size, shuffle):
    data = TensorDataset(torch.from_numpy(np.ascontiguousarray(windows)), torch.from_numpy(labels))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train(windows, labels, args, val=None, lr_monitor="train_loss"):
    """Train one model for a fixed number of epochs; `val` (windows, labels) is only logged."""
    pl.seed_everything(args.seed, workers=True)
    model = ActionClassificationLSTM(34, args.hidden, learning_rate=args.lr, num_classes=len(args.labels),
                                     class_names=list(args.labels), lr_monitor=lr_monitor)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="cpu", deterministic=True, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    val_loader = loader(*val, args.batch_size, False) if val is not None else None
    trainer.fit(model, loader(windows, labels, args.batch_size, True), val_loader)
    return trainer, model


def predict(model, windows):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(np.ascontiguousarray(windows[i:i + 4096]))).argmax(1)
                          for i in range(0, len(windows), 4096)]).numpy()


def metrics(y_true, y_pred, labels):
    return calculate_classification_metrics(y_true.tolist(), y_pred.tolist(), labels)


def summary(y_true, y_pred, labels):
    m = metrics(y_true, y_pred, labels)
    return {k: m[k] for k in ("accuracy", "macroPrecision", "macroRecall", "macroF1", "sampleCount")} | {
        "perClass": m["perClass"], "confusionMatrix": m["confusionMatrix"]}


def buckets(values, y_true, y_pred, names, labels):
    """Accuracy and macro F1 in three equal-size buckets (terciles) of `values`."""
    edges = np.nanquantile(values, [1 / 3, 2 / 3])
    which = np.digitize(values, edges)
    out = []
    for b, name in enumerate(names):
        idx = which == b
        lo = values[idx].min() if idx.any() else float("nan")
        hi = values[idx].max() if idx.any() else float("nan")
        m = metrics(y_true[idx], y_pred[idx], labels)
        out.append({"bucket": name, "range": [round(float(lo), 3), round(float(hi), 3)],
                    "windows": int(idx.sum()), "accuracy": m["accuracy"], "macroF1": m["macroF1"]})
    return out


def add_extra(args, windows, labels):
    """Append windows from --extra indexes; new activity names become new classes (added to args.labels)."""
    if not args.extra:
        return windows, labels, None
    parts = [load_extracted(path) for path in args.extra]
    x = np.concatenate([p[0] for p in parts])
    names = np.concatenate([p[1] for p in parts])
    info = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    args.labels = tuple(args.labels) + tuple(str(n) for n in sorted(set(names) - set(args.labels)))
    y = np.array([args.labels.index(n) for n in names], dtype=np.int64)
    print(f"Extra: {len(x):,} windows from {len(set(info['clip']))} clips, {len(set(info['subject']))} subjects; "
          f"classes now {', '.join(args.labels)}")
    return np.concatenate([windows, x]), np.concatenate([labels, y]), info


def run_standard(args):
    x_tr, y_tr = load_split(args.data, "train")
    x_te, y_te = load_split(args.data, "test")
    print(f"Standard split: {len(x_tr):,} train / {len(x_te):,} test windows, {args.epochs} epochs")
    start = time.time()
    trainer, model = train(x_tr, y_tr, args, val=(x_te, y_te), lr_monitor="val_loss")
    report = {"mode": "standard", "epochs": args.epochs, "trainSeconds": round(time.time() - start, 1),
              "test": summary(y_te, predict(model, x_te), args.labels)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(OUT_DIR / "standard.ckpt")
    return report


def run_final(args):
    """Train on every labeled window (no held-out data) for the checkpoint the app would load."""
    x_tr, y_tr = load_split(args.data, "train")
    x_te, y_te = load_split(args.data, "test")
    windows, labels, _ = add_extra(args, np.concatenate([x_tr, x_te]), np.concatenate([y_tr, y_te]))
    print(f"Final model: {len(windows):,} windows, {len(args.labels)} classes, {args.epochs} epochs")
    start = time.time()
    trainer, _ = train(windows, labels, args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(OUT_DIR / "final.ckpt")
    return {"mode": "final", "epochs": args.epochs, "trainSeconds": round(time.time() - start, 1),
            "labels": list(args.labels), "windows": len(windows),
            "windowsPerClass": np.bincount(labels, minlength=len(args.labels)).tolist(),
            "checkpoint": "models/training/final.ckpt"}


def run_cv(args, random_folds=False):
    windows, labels = load_split(args.data, "train")
    n_orig = len(windows)
    videos = video_ids(windows)
    if random_folds:
        folds = np.random.default_rng(args.seed).integers(0, args.folds, n_orig)
        kind = "random windows (leaky: overlapping windows of one video land in train and test)"
    else:
        folds = block_folds(videos, labels, args.folds)
        kind = "contiguous video blocks per class (≈ unseen people/camera views)"
    windows, labels, extra = add_extra(args, windows, labels)
    if extra is not None:
        # Recordings carry real subject IDs: every clip of a person goes to the same fold.
        subjects = sorted(set(extra["subject"]))
        fold_of = {s: i % args.folds for i, s in enumerate(subjects)}
        folds = np.concatenate([folds, [fold_of[s] for s in extra["subject"]]])
        kind += "; added recordings folded by subject (unseen people)"
    print(f"{args.folds}-fold CV over {len(windows):,} windows — {kind}")
    y_pred = np.empty_like(labels)
    per_fold = []
    start = time.time()
    for k in range(args.folds):
        test = folds == k
        _, model = train(windows[~test], labels[~test], args)
        y_pred[test] = predict(model, windows[test])
        m = metrics(labels[test], y_pred[test], args.labels)
        per_fold.append({"fold": k, "windows": int(test.sum()), "accuracy": m["accuracy"], "macroF1": m["macroF1"]})
        print(f"  fold {k + 1}/{args.folds}: accuracy {m['accuracy']:.1%}, macro F1 {m['macroF1']:.1%} "
              f"({time.time() - start:.0f} s)")
    accs = np.array([f["accuracy"] for f in per_fold])
    f1s = np.array([f["macroF1"] for f in per_fold])
    o = slice(0, n_orig)  # quality and size buckets describe the original dataset
    report = {
        "mode": "cv-random" if random_folds else "cv-block", "folds": args.folds, "epochs": args.epochs,
        "labels": list(args.labels), "splitting": kind, "trainSeconds": round(time.time() - start, 1),
        "accuracyMean": float(accs.mean()), "accuracyStd": float(accs.std()),
        "macroF1Mean": float(f1s.mean()), "macroF1Std": float(f1s.std()),
        "perFold": per_fold,
        "pooled": summary(labels, y_pred, args.labels),
        "byQuality": buckets(missing_joint_share(windows[o]), labels[o], y_pred[o],
                             ["few missing joints", "some missing joints", "many missing joints"], args.labels),
        "byPersonSize": buckets(person_height(windows[o], videos), labels[o], y_pred[o],
                                ["small (far)", "medium", "large (near)"], args.labels),
    }
    if extra is not None:
        env = np.concatenate([np.full(n_orig, "original dataset (MHAD lab, OpenPose)"),
                              [f"{s}: {e}" for s, e in zip(extra["source"], extra["environment"])]])
        report["byEnvironment"] = []
        for name in sorted(set(env)):
            idx = env == name
            m = metrics(labels[idx], y_pred[idx], args.labels)
            report["byEnvironment"].append({"bucket": name, "windows": int(idx.sum()),
                                            "accuracy": m["accuracy"], "macroF1": m["macroF1"]})
    return report


def print_report(r):
    if r["mode"] == "final":
        print(f"\nFinal model on {r['windows']:,} windows ({r['trainSeconds']:.0f} s): {r['checkpoint']}")
        return
    if r["mode"] == "standard":
        t = r["test"]
        print(f"\nTest split: accuracy {t['accuracy']:.1%}, macro F1 {t['macroF1']:.1%} ({r['trainSeconds']:.0f} s)")
        return
    print(f"\n{r['mode']}: accuracy {r['accuracyMean']:.1%} ± {r['accuracyStd']:.1%}, "
          f"macro F1 {r['macroF1Mean']:.1%} ± {r['macroF1Std']:.1%}")
    for name in ("byQuality", "byPersonSize", "byEnvironment"):
        for b in r.get(name, []):
            print(f"  {b['bucket']:<40} {b['windows']:>5} windows  accuracy {b['accuracy']:.1%}  F1 {b['macroF1']:.1%}")
    for c in r["pooled"]["perClass"]:
        print(f"  {c['label']:<15} F1 {c['f1']:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["standard", "cv", "final"])
    parser.add_argument("--data", default=str(DEFAULT_ZIP), help="dataset zip or extracted folder")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400, help="the shipped model trained ~392 epochs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--compare-random", action="store_true", help="also run CV with random window folds")
    parser.add_argument("--extra", action="append", default=[],
                        help="index.csv from scripts/extract_poses.py (new recordings/classes); repeatable")
    args = parser.parse_args()
    args.labels = LABELS

    if args.mode == "standard":
        reports = [run_standard(args)]
    elif args.mode == "final":
        reports = [run_final(args)]
    else:
        reports = [run_cv(args)]
        if args.compare_random:
            args.labels = LABELS
            reports.append(run_cv(args, random_folds=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in reports:
        print_report(r)
        path = OUT_DIR / f"{r['mode']}_report.json"
        path.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
