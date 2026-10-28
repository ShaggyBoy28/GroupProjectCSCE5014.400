"""JSON API used by the React frontend (see docs/API.md).

The blueprint doesn't import torch, Detectron2 or OpenCV. The heavy work is
injected through ``services`` so the API can be tested with fakes:

    services.probe(path) -> {"fps", "frameCount", "width", "height", "durationS"}
    services.analyze(path) -> iterator of event dicts, ending with {"type": "done"}

It also builds the Activity Breakdown data from the saved predictions. A
prediction is one LSTM output for one frame: {"frame": int, "probabilities": [6 floats]}.
"""

import json
import os

from flask import Blueprint, Response, abort, jsonify, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

# ---------- Activity Breakdown ----------

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS",
]


def _round(value, digits=4):
    return round(float(value), digits)


def label_of(probabilities):
    """Return the label with the highest probability and that probability."""
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return LABELS[best], float(probabilities[best])


def build_segments(predictions, fps):
    """Group consecutive predictions with the same top label into segments.

    A segment runs from its first predicted frame up to the first frame of the
    next segment (or one frame past the last prediction for the final segment).
    """
    segments = []
    for prediction in predictions:
        label, confidence = label_of(prediction["probabilities"])
        frame = prediction["frame"]
        if segments and segments[-1]["label"] == label:
            current = segments[-1]
            current["endFrame"] = frame + 1
            current["_confidences"].append(confidence)
        else:
            if segments:
                segments[-1]["endFrame"] = frame
            segments.append(
                {"label": label, "startFrame": frame, "endFrame": frame + 1, "_confidences": [confidence]}
            )

    result = []
    for segment in segments:
        confidences = segment.pop("_confidences")
        start_s = segment["startFrame"] / fps
        end_s = segment["endFrame"] / fps
        result.append(
            {
                **segment,
                "startS": _round(start_s, 3),
                "endS": _round(end_s, 3),
                "durationS": _round(end_s - start_s, 3),
                "avgConfidence": _round(sum(confidences) / len(confidences)),
                "peakConfidence": _round(max(confidences)),
            }
        )
    return result


def time_per_activity(segments):
    """Seconds and share of analyzed time for every class, largest first."""
    seconds = {label: 0.0 for label in LABELS}
    for segment in segments:
        seconds[segment["label"]] += segment["durationS"]
    total = sum(seconds.values())
    rows = [
        {
            "label": label,
            "seconds": _round(value, 3),
            "share": _round(value / total) if total else 0.0,
        }
        for label, value in seconds.items()
    ]
    # Stable sort keeps class order for ties (e.g. several unused classes).
    return sorted(rows, key=lambda row: row["seconds"], reverse=True)


def build_breakdown(predictions, video, window_size):
    """Build the full Activity Breakdown payload returned by the API.

    ``video`` holds the probed video metadata (name, fps, frameCount, width,
    height, durationS). ``predictions`` must be ordered by frame.
    """
    fps = video["fps"] or 30
    segments = build_segments(predictions, fps)
    activity_times = time_per_activity(segments)
    confidences = [label_of(p["probabilities"])[1] for p in predictions]

    has_predictions = bool(predictions)
    main = activity_times[0] if has_predictions else None
    summary = {
        "mainActivity": main["label"] if main else None,
        "mainActivityShare": main["share"] if main else 0.0,
        "labelChanges": max(len(segments) - 1, 0),
        "averageConfidence": _round(sum(confidences) / len(confidences)) if has_predictions else 0.0,
        "predictionCount": len(predictions),
        "firstPredictionS": _round(predictions[0]["frame"] / fps, 3) if has_predictions else None,
    }

    rows = []
    for prediction in predictions:
        label, confidence = label_of(prediction["probabilities"])
        rows.append(
            {
                "frame": prediction["frame"],
                "timeS": _round(prediction["frame"] / fps, 3),
                "label": label,
                "confidence": _round(confidence),
                "probabilities": [_round(p) for p in prediction["probabilities"]],
            }
        )

    return {
        "video": video,
        "windowSize": window_size,
        "labels": LABELS,
        "summary": summary,
        "segments": segments,
        "timePerActivity": activity_times,
        "predictions": rows,
    }


# ---------- API ----------

SAMPLE_VIDEO = "sample_video.mp4"
ALLOWED_EXTENSIONS = {"mp4"}


def result_video_name(name):
    return "res_{}".format(name)


def result_data_name(name):
    return "res_{}.json".format(os.path.splitext(name)[0])


def _is_allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_api_blueprint(services, video_folder):
    api = Blueprint("api", __name__, url_prefix="/api")
    video_folder = os.path.abspath(video_folder)

    def video_path(name):
        """Resolve a video name to a file in the video folder, or 404."""
        safe_name = secure_filename(name)
        if not safe_name or safe_name != name or not _is_allowed(safe_name):
            abort(404, description="Video not found")
        path = os.path.join(video_folder, safe_name)
        if not os.path.isfile(path):
            abort(404, description="Video not found")
        return path

    def result_data_path(name):
        return os.path.join(video_folder, result_data_name(name))

    @api.errorhandler(404)
    @api.errorhandler(400)
    @api.errorhandler(413)
    def json_error(error):
        return jsonify({"error": error.description}), error.code

    @api.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @api.get("/sample")
    def sample():
        return get_video(SAMPLE_VIDEO)

    @api.post("/videos")
    def upload_video():
        video_file = request.files.get("video")
        if video_file is None or not video_file.filename:
            abort(400, description="Choose a video file to upload")
        filename = secure_filename(video_file.filename)
        if not filename or not _is_allowed(filename):
            abort(400, description="Only .mp4 files are supported")
        if filename == SAMPLE_VIDEO or filename.startswith("res_"):
            abort(400, description="That file name is reserved; rename the file and try again")
        video_file.save(os.path.join(video_folder, filename))
        return jsonify({"name": filename}), 201

    @api.get("/videos/<name>")
    def get_video(name):
        path = video_path(name)
        info = {"name": name, **services.probe(path)}
        info["analyzed"] = os.path.isfile(result_data_path(name)) and os.path.isfile(
            os.path.join(video_folder, result_video_name(name))
        )
        return jsonify(info)

    @api.get("/videos/<name>/file")
    def get_video_file(name):
        video_path(name)
        return send_from_directory(video_folder, name, mimetype="video/mp4")

    @api.get("/videos/<name>/analyze")
    def analyze_video(name):
        path = video_path(name)

        def events():
            try:
                for event in services.analyze(path):
                    yield "data: {}\n\n".format(json.dumps(event))
            except Exception as error:  # surface the failure to the browser instead of a dropped stream
                yield "data: {}\n\n".format(json.dumps({"type": "error", "message": str(error)}))

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return Response(stream_with_context(events()), mimetype="text/event-stream", headers=headers)

    @api.get("/videos/<name>/result")
    def get_result_video(name):
        video_path(name)
        result_name = result_video_name(name)
        if not os.path.isfile(os.path.join(video_folder, result_name)):
            abort(404, description="This video hasn't been analyzed yet")
        download = request.args.get("download") == "1"
        return send_from_directory(
            video_folder, result_name, mimetype="video/mp4", as_attachment=download, download_name=result_name
        )

    @api.get("/videos/<name>/breakdown")
    def get_breakdown(name):
        video_path(name)
        data_path = result_data_path(name)
        if not os.path.isfile(data_path):
            abort(404, description="This video hasn't been analyzed yet")
        with open(data_path, "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
        return jsonify(build_breakdown(data["predictions"], data["video"], data["windowSize"]))

    return api
