"""JSON API behind the Dataset page: upload recordings and turn them into training data.

    GET    /api/dataset                    clips, per-class totals and processing jobs
    POST   /api/dataset/clips              multipart: video, person, activity, environment (optional)
    DELETE /api/dataset/clips/<id>         remove a clip, its recording and its pose data
    GET    /api/dataset/clips/<id>/preview first frame with a person in it (JPEG)

Uploads are saved to data/recordings/<person>/[<environment>/]<activity>/ and queued. One background
thread runs pose extraction (src/pose_extraction.py) with the app's own Detectron2 predictor, one clip
at a time, so the page can be closed while it works. The result is the same as
scripts/extract_poses.py, and scripts/train_model.py --extra data/extracted/index.csv trains on it.
"""

import itertools
import queue
import threading
import time
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file
from werkzeug.utils import secure_filename

from . import pose_extraction as px

KEEP_FINISHED_JOBS = 50


class ExtractionQueue:
    """A single worker thread that extracts poses from uploaded clips in order."""

    def __init__(self, get_detector, out_dir):
        self._get_detector = get_detector
        self._out = Path(out_dir)
        self._queue = queue.Queue()
        self._jobs = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._thread = None

    def submit(self, clip):
        job = {"id": next(self._ids), "clipId": clip["id"], "file": clip["path"], "activity": clip["activity"],
               "person": clip["subject"], "environment": clip["environment"], "status": "queued",
               "framesDone": 0, "framesTotal": 0, "error": None, "submittedAt": time.time()}
        with self._lock:
            self._jobs[job["id"]] = job
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="pose-extraction", daemon=True)
                self._thread.start()
        self._queue.put((job["id"], clip))
        return dict(job)

    def jobs(self):
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(), key=lambda j: j["id"])]

    def busy_with(self, clip_id):
        with self._lock:
            return any(j["clipId"] == clip_id and j["status"] in ("queued", "processing")
                       for j in self._jobs.values())

    def _update(self, job_id, **changes):
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self):
        while True:
            job_id, clip = self._queue.get()
            self._update(job_id, status="processing")
            try:
                (self._out / "previews").mkdir(parents=True, exist_ok=True)
                keypoints, info = px.extract(
                    self._get_detector(), clip["file"], self._out / "previews" / f"{clip['id']}.jpg",
                    on_progress=lambda done, total: self._update(job_id, framesDone=done, framesTotal=total))
                px.save_clip(clip, keypoints, info, self._out)
                self._update(job_id, status="done", windows=info["windows"])
            except Exception as error:  # report any failure on the page instead of killing the worker
                self._update(job_id, status="error", error=str(error) or type(error).__name__)
            finally:
                self._prune()

    def _prune(self):
        with self._lock:
            finished = [j for j in self._jobs.values() if j["status"] in ("done", "error")]
            for job in sorted(finished, key=lambda j: j["id"])[:-KEEP_FINISHED_JOBS]:
                del self._jobs[job["id"]]


def _number(value, cast=int):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _clip_payload(row):
    return {
        "id": row["id"], "activity": row["activity"], "person": row["subject"],
        "environment": row["environment"], "take": row["take"], "file": row["path"],
        "poseFrames": _number(row.get("poseFrames")), "windows": _number(row.get("windows")),
        "framesNoPerson": _number(row.get("framesNoPerson")), "framesUsed": _number(row.get("framesUsed")),
        "durationS": (round(_number(row.get("sourceFrames")) / _number(row.get("sourceFps"), float), 1)
                      if _number(row.get("sourceFrames")) and _number(row.get("sourceFps"), float) else None),
    }


def create_dataset_blueprint(get_detector, get_model_labels, *, recordings_dir=px.RECORDINGS_DIR,
                             extracted_dir=px.EXTRACTED_DIR):
    bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")
    recordings_dir, extracted_dir = Path(recordings_dir), Path(extracted_dir)
    jobs = ExtractionQueue(get_detector, extracted_dir)

    @bp.errorhandler(400)
    @bp.errorhandler(404)
    @bp.errorhandler(409)
    @bp.errorhandler(413)
    def json_error(error):
        return jsonify({"error": error.description}), error.code

    @bp.get("")
    def overview():
        clips = [_clip_payload(r) for r in px.read_index(extracted_dir)]
        model_labels = list(get_model_labels())
        activities = list(dict.fromkeys([*model_labels, *px.NEW_CLASSES, *(c["activity"] for c in clips)]))
        per_class = []
        for label in activities:
            mine = [c for c in clips if c["activity"] == label]
            per_class.append({"label": label, "inModel": label in model_labels, "clips": len(mine),
                              "people": len({c["person"] for c in mine}),
                              "windows": sum(c["windows"] or 0 for c in mine)})
        return jsonify({
            "activities": activities,
            "clips": clips,
            "jobs": jobs.jobs(),
            "summary": {"clips": len(clips), "people": len({c["person"] for c in clips}),
                        "environments": len({c["environment"] for c in clips}),
                        "windows": sum(c["windows"] or 0 for c in clips), "perClass": per_class},
            "people": sorted({c["person"] for c in clips}),
            "environments": sorted({c["environment"] for c in clips} - {"unspecified"}),
            "indexPath": "data/extracted/index.csv",
        })

    @bp.post("/clips")
    def add_clip():
        video = request.files.get("video")
        person = px.slug(request.form.get("person", ""))
        activity = px.activity_name(request.form.get("activity", ""))
        environment = px.slug(request.form.get("environment", ""))
        if video is None or not video.filename:
            abort(400, description="Choose a video file")
        if not person:
            abort(400, description="Enter who is in the video")
        if not activity:
            abort(400, description="Choose the activity")
        filename = secure_filename(video.filename)
        if not filename or Path(filename).suffix.lower() not in px.VIDEO_SUFFIXES:
            abort(400, description="Video must be .mp4, .mov, .m4v, .avi or .mkv")
        folder = recordings_dir.joinpath(person, *([environment] if environment else []), activity.lower())
        target = folder / filename
        clip = px.describe(target, recordings_dir)
        if target.exists() or any(r["id"] == clip["id"] for r in px.read_index(extracted_dir)):
            abort(409, description=f"{person} already has a {filename} for this activity; rename the file")
        folder.mkdir(parents=True, exist_ok=True)
        video.save(target)
        return jsonify(jobs.submit({**clip, "file": target})), 202

    @bp.delete("/clips/<clip_id>")
    def delete_clip(clip_id):
        if jobs.busy_with(clip_id):
            abort(409, description="That clip is still being processed")
        row = px.remove_clip(clip_id, extracted_dir)
        if row is None:
            abort(404, description="Clip not found")
        recording = (recordings_dir / row["path"]).resolve()
        if recording.is_relative_to(recordings_dir.resolve()):
            recording.unlink(missing_ok=True)
            for parent in recording.parents:  # tidy up now-empty person/activity folders
                if parent == recordings_dir.resolve() or any(parent.iterdir()):
                    break
                parent.rmdir()
        return jsonify({"deleted": clip_id})

    @bp.get("/clips/<clip_id>/preview")
    def preview(clip_id):
        path = extracted_dir / "previews" / f"{secure_filename(clip_id)}.jpg"
        if not path.is_file():
            abort(404, description="No preview for that clip")
        return send_file(path, mimetype="image/jpeg", max_age=0)

    return bp

