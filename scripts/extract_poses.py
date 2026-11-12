"""Turn the group's activity recordings into pose sequences the LSTM can train on (command-line version).

The app's Dataset page (/dataset) does the same from the browser. Here the folder a video sits in says
what it is:

    data/recordings/<person>/<activity>/<any name>.mp4
    data/recordings/<person>/<environment>/<activity>/<any name>.mp4     (optional room level)

Activity folders can use friendly names (walking, sit_stand, clap…; see src/pose_extraction.ALIASES);
any other folder name becomes a new class. Videos already extracted are skipped.

Usage (project folder, .venv active):
    python scripts/extract_poses.py                       # everything new under data/recordings/
    python scripts/extract_poses.py path/to/clip.mp4 --person josh --activity walking   # one video
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pose_extraction as px  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", nargs="?", default=str(px.RECORDINGS_DIR),
                        help="recordings folder (default data/recordings) or a single video")
    parser.add_argument("--person", help="single video: who is in it")
    parser.add_argument("--activity", help="single video: what they are doing")
    parser.add_argument("--environment", default="", help="single video: room/setting (optional)")
    args = parser.parse_args()

    source = Path(args.input)
    if source.is_file():
        # Copy it into data/recordings/ so the folders stay the single record of what was collected.
        if not (args.person and args.activity):
            raise SystemExit("for a single video, also pass --person and --activity")
        parts = [px.slug(args.person)] + ([px.slug(args.environment)] if args.environment else [])
        target = px.RECORDINGS_DIR.joinpath(*parts, px.activity_name(args.activity).lower(), source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        clips = [{**px.describe(target), "file": target}]
    elif source.is_dir():
        clips, skipped = px.find_recordings(source)
        for path in skipped:
            print(f"skipped {path}: put it in <person>/<activity>/ or <person>/<environment>/<activity>/")
    else:
        raise SystemExit(f"{source} not found. Put videos in data/recordings/<person>/<activity>/")
    if not clips:
        raise SystemExit(f"No videos found under {source}. Layout: <person>/<activity>/<clip>.mp4")

    done = {r["id"] for r in px.read_index()}
    todo = [c for c in clips if c["id"] not in done]
    print(f"{len(clips)} videos found, {len(clips) - len(todo)} already extracted, {len(todo)} to do")
    if not todo:
        return

    detector = px.build_pose_detector()
    (px.EXTRACTED_DIR / "previews").mkdir(parents=True, exist_ok=True)
    for n, clip in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {clip['path']} → {clip['activity']} ({clip['subject']}, {clip['environment']})")
        start = time.time()
        try:
            keypoints, info = px.extract(
                detector, clip["file"], px.EXTRACTED_DIR / "previews" / f"{clip['id']}.jpg",
                on_progress=lambda k, t: k % 50 == 0 and print(f"    {k}/{t} frames…", flush=True))
        except ValueError as error:
            print(f"    skipped: {error}")
            continue
        px.save_clip(clip, keypoints, info)
        warn = "  ⚠ under 32 pose frames, too short to train on" if info["windows"] == 0 else ""
        print(f"    {info['poseFrames']} pose frames ({info['framesNoPerson']} without a person) → "
              f"{info['windows']} training windows, {time.time() - start:.0f} s{warn}")
    print(f"Next: python scripts/train_model.py cv --extra {px.EXTRACTED_DIR / 'index.csv'}")


if __name__ == "__main__":
    main()
