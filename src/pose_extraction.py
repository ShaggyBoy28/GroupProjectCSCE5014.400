"""Turning recorded videos into pose sequences for training (shared by the web page and the CLI).

Recordings live in data/recordings/<person>/<activity>/<clip> (optionally
<person>/<environment>/<activity>/<clip>). Each extracted clip becomes data/extracted/<clip id>.npy
(frames × 34 float32: the 17 Detectron2 keypoints' raw pixel x, y) plus one row in
data/extracted/index.csv, which scripts/train_model.py --extra reads.

Pose detection is the app's own: the Detectron2 predictor from app.py, people kept above 0.9, the
top-scoring person used (src/utils.filter_persons), frames with no person skipped, as the app does.
Clips are resampled to 22 frames per second, the training data's rate, so a 32-frame window covers
the same ≈1.45 s.
"""

import csv
import re
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "data" / "recordings"
EXTRACTED_DIR = ROOT / "data" / "extracted"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
TARGET_FPS = 22.0
WINDOW, STRIDE = 32, 6  # training window and spacing (src/pose_dataset.py)

# The group's planned new classes, offered first in the web page's activity list.
NEW_CLASSES = ("SIT_DOWN_STAND_UP", "WALKING", "THROWING")
# Friendly folder names → class names. The first six must match the trained model's classes exactly.
ALIASES = {
    "JUMPING": "JUMPING", "JUMP": "JUMPING",
    "JUMPING_JACKS": "JUMPING_JACKS", "JUMPING_JACK": "JUMPING_JACKS", "JACKS": "JUMPING_JACKS",
    "BOXING": "BOXING", "PUNCHING": "BOXING",
    "WAVING_2HANDS": "WAVING_2HANDS", "WAVING_TWO_HANDS": "WAVING_2HANDS", "WAVE_2": "WAVING_2HANDS",
    "WAVING_1HAND": "WAVING_1HAND", "WAVING_ONE_HAND": "WAVING_1HAND", "WAVE_1": "WAVING_1HAND",
    "CLAPPING_HANDS": "CLAPPING_HANDS", "CLAPPING": "CLAPPING_HANDS", "CLAP": "CLAPPING_HANDS",
    "SIT_DOWN_STAND_UP": "SIT_DOWN_STAND_UP", "SIT_STAND": "SIT_DOWN_STAND_UP", "SIT_DOWN": "SIT_DOWN_STAND_UP",
    "STAND_UP": "SIT_DOWN_STAND_UP",
    "WALKING": "WALKING", "WALK": "WALKING",
    "THROWING": "THROWING", "THROW": "THROWING",
}
FIELDS = ["id", "path", "activity", "subject", "environment", "take", "source", "sourceFps", "sourceFrames",
          "framesUsed", "framesNoPerson", "poseFrames", "windows", "meanKeypointScore", "personHeightPx",
          "targetFps"]
_index_lock = threading.Lock()


def activity_name(text):
    key = re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")
    return ALIASES.get(key, key)


def slug(text):
    """Folder-safe lower-case name: 'Josh R.' → 'josh-r'."""
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def clip_id(subject, environment, activity, take):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join((subject, environment, activity, take)))


def describe(path, root=RECORDINGS_DIR):
    """Person, environment, activity and take from a recording's place in the folders, or None."""
    parts = Path(path).relative_to(root).parts[:-1]
    if len(parts) == 2:
        person, environment, activity = parts[0], "unspecified", parts[1]
    elif len(parts) == 3:
        person, environment, activity = parts
    else:
        return None
    activity = activity_name(activity)
    take = Path(path).stem
    return {"path": str(Path(path).relative_to(root)), "subject": person, "environment": environment,
            "activity": activity, "take": take, "id": clip_id(person, environment, activity, take)}


def find_recordings(root=RECORDINGS_DIR):
    """(clips, skipped paths) for every video under root."""
    clips, skipped = [], []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or path.name.startswith("."):
            continue
        clip = describe(path, root)
        (clips if clip else skipped).append({**clip, "file": path} if clip else path)
    return clips, skipped


def build_pose_detector():
    """The same Detectron2 configuration app.py uses (for the command-line script)."""
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def extract(detector, path, preview, target_fps=TARGET_FPS, on_progress=None):
    """Keypoints of the main person in each kept frame: ((n, 34) float32, quality figures).

    on_progress(done, total) is called after every processed frame; total is the expected number of
    frames after resampling.
    """
    import cv2

    from .utils import filter_persons
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("could not open the video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = fps / target_fps if target_fps and fps > target_fps else 1.0
    expected = max(1, int(total / step)) if total > 0 else 0
    next_keep, i, kept, no_person, rows, scores = 0.0, 0, 0, 0, [], []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            if i - 1 + 1e-6 < next_keep:
                continue
            next_keep += step
            kept += 1
            persons, idx = filter_persons(detector(img))
            if kept == 1 or (not rows and persons):
                cv2.imwrite(str(preview), img)
            if persons:
                person = persons[idx[0]].numpy()  # (17, 3): x, y, keypoint score
                rows.append(person[:, :2].reshape(-1))
                scores.append(float(person[:, 2].mean()))
            else:
                no_person += 1
            if on_progress is not None:
                on_progress(kept, max(expected, kept))
    finally:
        cap.release()
    if kept == 0:
        raise ValueError("the video has no readable frames")
    keypoints = np.asarray(rows, dtype=np.float32).reshape(-1, 34)
    height = ""
    if len(keypoints):
        k = keypoints.reshape(-1, 17, 2)
        height = round(float(np.median(k[:, 15:17, 1].mean(1) - k[:, 5:7, 1].mean(1))), 1)  # shoulders → ankles
    return keypoints, {"sourceFps": round(fps, 2), "sourceFrames": total, "framesUsed": kept,
                       "framesNoPerson": no_person, "poseFrames": len(keypoints),
                       "windows": max(0, (len(keypoints) - WINDOW) // STRIDE + 1),
                       "meanKeypointScore": round(float(np.mean(scores)), 3) if scores else "",
                       "personHeightPx": height, "targetFps": target_fps}


def read_index(out=EXTRACTED_DIR):
    path = Path(out) / "index.csv"
    if not path.exists():
        return []
    with _index_lock, path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_clip(clip, keypoints, info, out=EXTRACTED_DIR):
    """Write the keypoints and add (or replace) the clip's row in index.csv."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{clip['id']}.npy", keypoints)
    row = {"id": clip["id"], **{k: clip[k] for k in ("path", "activity", "subject", "environment", "take")},
           "source": "group", **info}
    _rewrite_index(out, lambda rows: [r for r in rows if r["id"] != clip["id"]] + [row])
    return row


def remove_clip(clip_id_, out=EXTRACTED_DIR):
    """Drop a clip's row and files from the extracted data. Returns the removed row or None."""
    out = Path(out)
    removed = []
    _rewrite_index(out, lambda rows: [r for r in rows if r["id"] != clip_id_ or removed.append(r)])
    for name in (f"{clip_id_}.npy", f"previews/{clip_id_}.jpg"):
        (out / name).unlink(missing_ok=True)
    return removed[0] if removed else None


def _rewrite_index(out, change):
    path = out / "index.csv"
    with _index_lock:
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        rows = change(rows)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
