"""Loading and describing the 2D pose dataset (RNN-HAR-2D-Pose-database) for training and evaluation.

The dataset files hold windows of 32 frames, one frame per line, with no video, subject or camera IDs.
Consecutive windows of the same video overlap by 26 frames (a new window every 6 frames), so
`video_ids` links windows back into their source videos: 1,150 train + 288 test = the 1,438 videos
the dataset README reports.

Rows are converted exactly like training always did (src/lstm.py PoseDataModule): neck dropped,
keypoints reordered to Detectron2's 17-point layout, labels made 0-based.
"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .lstm import WINDOW_SIZE, PoseDataModule

LABELS = ("JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS")
DEFAULT_ZIP = Path(__file__).resolve().parent.parent / "data" / "RNN-HAR-2D-Pose-database.zip"
_FILES = {"train": ("X_train.txt", "Y_train.txt"), "test": ("X_test.txt", "Y_test.txt")}
OVERLAP = 26  # frames shared by consecutive windows of one video (dataset README)


def load_split(data, split):
    """Return (windows (n, 32, 34) float32, labels (n,) int64) for 'train' or 'test'."""
    x_name, y_name = _FILES[split]
    data = Path(data)
    loader = PoseDataModule(data_root="", batch_size=1)
    with tempfile.TemporaryDirectory() as workdir:
        if data.is_dir():
            found = {p.name: p for p in data.rglob("*.txt") if p.name in (x_name, y_name)}
        else:
            with zipfile.ZipFile(data) as archive:
                found = {Path(m).name: Path(archive.extract(m, workdir)) for m in archive.namelist()
                         if Path(m).name in (x_name, y_name) and not Path(m).name.startswith("._")}
        if len(found) != 2:
            raise FileNotFoundError(f"{x_name} / {y_name} not found in {data}")
        windows = loader.load_X(str(found[x_name]))
        labels = loader.load_y(str(found[y_name])).reshape(-1).astype(np.int64)
    if len(windows) != len(labels) or windows.shape[1:] != (WINDOW_SIZE, 34):
        raise ValueError(f"Unexpected data shape {windows.shape} for {len(labels)} labels")
    return windows, labels


def video_ids(windows):
    """Number each window by the video it came from (consecutive windows overlapping by 26 frames)."""
    shift = WINDOW_SIZE - OVERLAP
    same_video = np.all(windows[1:, :OVERLAP] == windows[:-1, shift:], axis=(1, 2))
    return np.concatenate([[0], np.cumsum(~same_video)])


def block_folds(videos, labels, k):
    """Assign each video to one of k folds as contiguous blocks, separately per class.

    The files list videos of a class in recording order, and neighboring videos are repetitions by the
    same person from the same camera. Keeping contiguous blocks together therefore approximates leaving
    out whole people/views (exact person and camera IDs are not in the files). Returns a fold per window.
    """
    fold_of_video = {}
    for label in np.unique(labels):
        class_videos = list(dict.fromkeys(videos[labels == label]))  # file order, unique
        for fold, block in enumerate(np.array_split(np.arange(len(class_videos)), k)):
            for i in block:
                fold_of_video[class_videos[i]] = fold
    return np.array([fold_of_video[v] for v in videos])


def missing_joint_share(windows):
    """Share of keypoints the pose detector did not find (stored as 0,0) in each window."""
    joints = windows.reshape(len(windows), WINDOW_SIZE, -1, 2)
    return (joints == 0).all(axis=-1).mean(axis=(1, 2))


def person_height(windows, videos):
    """Average shoulder-to-ankle height in pixels per window's video: how large/near the person appears."""
    joints = windows.reshape(len(windows), WINDOW_SIZE, -1, 2)
    shoulders, ankles = joints[:, :, 5:7, 1], joints[:, :, 15:17, 1]  # Detectron2: 5,6 shoulders; 15,16 ankles
    valid = (shoulders > 0).all(-1) & (ankles > 0).all(-1)
    span = np.where(valid, ankles.mean(-1) - shoulders.mean(-1), np.nan)
    per_window = np.nanmean(span, axis=1)
    out = np.empty(len(windows))
    for v in np.unique(videos):
        idx = videos == v
        out[idx] = np.nanmean(per_window[idx])
    return out


STRIDE = WINDOW_SIZE - OVERLAP  # 6 frames, the same window spacing as the dataset


def load_extracted(index_csv):
    """Windows from clips made by scripts/extract_poses.py.

    Returns (windows (n, 32, 34) float32, activity names (n,), info) where info holds per-window arrays
    'clip', 'subject', 'environment' and 'source'. Clips shorter than 32 pose frames give no windows.
    """
    import csv
    index_csv = Path(index_csv)
    with index_csv.open(newline="", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    windows, names, info = [], [], {"clip": [], "subject": [], "environment": [], "source": []}
    for clip in clips:
        frames = np.load(index_csv.parent / f"{clip['id']}.npy")
        for start in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
            windows.append(frames[start:start + WINDOW_SIZE])
            names.append(clip["activity"])
            for key in info:
                info[key].append(clip[{"clip": "id"}.get(key, key)])
    if not windows:
        return np.empty((0, WINDOW_SIZE, 34), np.float32), np.array([], str), {k: np.array([], str) for k in info}
    return (np.asarray(windows, np.float32), np.asarray(names),
            {k: np.asarray(v) for k, v in info.items()})
