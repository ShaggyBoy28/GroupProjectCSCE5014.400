"""Evaluate the trained LSTM on the labeled pose windows and write the artifact /api/model serves.

Usage (from the project folder, with the .venv active):
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --data data/RNN-HAR-2D-Pose-database.zip --out models/evaluation.json

Input: the dataset zip from lstm_train.ipynb (RNN-HAR-2D-Pose-database.zip), or a folder holding
X_test.txt / Y_test.txt. Rows are loaded exactly as in training (src/lstm.py: neck dropped, keypoints
reordered to Detectron2's 17-point layout, 32 rows per window, labels made 0-based).

Output: models/evaluation.json, read by app.py when MODEL_EVALUATION_PATH points to it (run.sh does this
automatically when the file exists). The file is checked with src/evaluation.load_evaluation before it is
written, so the app will accept it.

Note: the original training used X_test/Y_test as its validation set and kept the checkpoint with the lowest
validation loss, so these scores describe a validation split, not an untouched test set.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluation import load_evaluation  # noqa: E402
from src.lstm import WINDOW_SIZE, ActionClassificationLSTM, PoseDataModule  # noqa: E402
from src.metrics import ACTIVITY_LABELS, calculate_classification_metrics  # noqa: E402

SPLITS = {"test": ("X_test.txt", "Y_test.txt"), "train": ("X_train.txt", "Y_train.txt")}


def find_split_files(data, split, workdir):
    """Return paths to the X/Y files of a split, extracting them from the zip if needed."""
    x_name, y_name = SPLITS[split]
    data = Path(data)
    if data.is_dir():
        found = {p.name: p for p in data.rglob("*.txt") if p.name in (x_name, y_name)}
    elif zipfile.is_zipfile(data):
        with zipfile.ZipFile(data) as archive:
            members = {Path(m).name: m for m in archive.namelist()
                       if Path(m).name in (x_name, y_name) and not Path(m).name.startswith("._")}
            found = {name: Path(archive.extract(member, workdir)) for name, member in members.items()}
    else:
        raise SystemExit(f"{data} is not a folder or a zip file")
    missing = [n for n in (x_name, y_name) if n not in found]
    if missing:
        raise SystemExit(f"Could not find {', '.join(missing)} in {data}")
    return found[x_name], found[y_name]


def load_windows(x_path, y_path):
    """Load windows and labels with the same code the model was trained with."""
    loader = PoseDataModule(data_root="", batch_size=1)
    windows = loader.load_X(str(x_path))  # (n, 32, 34) float32
    labels = loader.load_y(str(y_path)).reshape(-1)  # (n,) 0-based
    if len(windows) != len(labels):
        raise SystemExit(f"{len(windows)} windows but {len(labels)} labels; files don't match")
    if windows.shape[1:] != (WINDOW_SIZE, 34):
        raise SystemExit(f"Unexpected window shape {windows.shape[1:]}, expected ({WINDOW_SIZE}, 34)")
    return windows, labels


def load_model(checkpoint):
    model = ActionClassificationLSTM.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    model.eval()
    return model


def predict(model, windows, batch_size=1024):
    """Top class for each window, using the checkpoint the app loads."""
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start:start + batch_size]))
            predictions.append(model(batch).argmax(dim=1).numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(ROOT / "data" / "RNN-HAR-2D-Pose-database.zip"),
                        help="dataset zip or folder with X_test.txt / Y_test.txt")
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "saved_model.ckpt"))
    parser.add_argument("--out", default=str(ROOT / "models" / "evaluation.json"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        x_path, y_path = find_split_files(args.data, args.split, workdir)
        windows, y_true = load_windows(x_path, y_path)
    print(f"Loaded {len(windows):,} windows of {WINDOW_SIZE} frames from the {args.split} split")

    model = load_model(args.checkpoint)
    labels = list(model.hparams.get("class_names") or ACTIVITY_LABELS)  # retrained models store their classes
    y_pred = predict(model, windows)
    metrics = calculate_classification_metrics(y_true.tolist(), y_pred.tolist(), labels)
    split_desc = {
        "test": "X_test.txt / Y_test.txt (validation split: used to pick the best checkpoint during training)",
        "train": "X_train.txt / Y_train.txt (training split: the model has seen these windows)",
    }[args.split]
    metrics["dataset"] = f"Berkeley MHAD 2D poses, {args.split} split"
    artifact = {
        "checkpointSha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "labels": labels,
        "windowSize": WINDOW_SIZE,
        "split": split_desc,
        "provenance": (
            f"RNN-HAR-2D-Pose-database ({Path(args.data).name}): OpenPose 2D poses from Berkeley MHAD "
            "(12 subjects, 4 views, 22 Hz; train/test separated by activity repetition per the dataset README). "
            "Rows loaded with src/lstm.py PoseDataModule.load_X/load_y (neck dropped, reordered to Detectron2 "
            f"keypoints, {WINDOW_SIZE}-frame windows, labels 1-6 made 0-based). Predictions are the argmax of "
            f"{Path(args.checkpoint).name}; metrics from src/metrics.calculate_classification_metrics, "
            "generated by scripts/evaluate_model.py."
        ),
        "metrics": metrics,
    }

    out = Path(args.out)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Same check the app runs on startup, so a bad file is never left in place.
    load_evaluation(str(tmp), args.checkpoint, labels, WINDOW_SIZE)
    os.replace(tmp, out)

    print(f"Accuracy {metrics['accuracy']:.1%} · macro precision {metrics['macroPrecision']:.1%} · "
          f"macro recall {metrics['macroRecall']:.1%} · macro F1 {metrics['macroF1']:.1%}")
    for row in metrics["perClass"]:
        print(f"  {row['label']:<15} P {row['precision']:.1%}  R {row['recall']:.1%}  "
              f"F1 {row['f1']:.1%}  n={row['support']}")
    print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
