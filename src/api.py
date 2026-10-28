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
import tempfile
import math

from flask import Blueprint, Response, abort, jsonify, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename
from .performance import performance_payload
from .results import read_result, video_operation, invalidate_result, ResultNotReady, VideoBusy
from .evaluation import validate_metrics

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


def create_api_blueprint(services, video_folder, *, window_size=32):
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

    @api.get("/model")
    def get_model():
        # The optional provider returns only independently validated evaluation
        # artifacts. Never estimate accuracy from an uploaded video's confidence.
        provider = getattr(services, "model_metrics", None)
        try:
            supplied = provider() if provider is not None else None
            metrics = validate_metrics(supplied, LABELS) if supplied is not None else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
            metrics = None
        return jsonify({"labels": LABELS, "windowSize": window_size, "metrics": metrics})

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
        if filename.lower() == SAMPLE_VIDEO.lower() or filename.lower().startswith("res_"):
            abort(400, description="That file name is reserved; rename the file and try again")
        path = os.path.join(video_folder, filename)
        temporary = None
        try:
            with video_operation(path):
                with tempfile.NamedTemporaryFile(dir=video_folder, prefix="res_upload_", delete=False) as staged:
                    temporary = staged.name
                    video_file.save(staged)
                invalidate_result(path)
                os.replace(temporary, path)
        except VideoBusy:
            return jsonify({"error": "Video is busy; try again after analysis finishes"}), 409
        except OSError:
            return jsonify({"error": "Video upload could not be saved"}), 500
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)
        return jsonify({"name": filename}), 201

    @api.get("/videos/<name>")
    def get_video(name):
        path = video_path(name)
        try:
            metadata = services.probe(path)
            for key in ("fps", "frameCount", "width", "height", "durationS"):
                value = metadata[key]
                if (type(value) not in (int, float) or value < 0
                        or value > 2**53 - 1 or not math.isfinite(value)):
                    raise ValueError("Invalid video metadata")
            if metadata["fps"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
                raise ValueError("Invalid video dimensions or frame rate")
            info = {"name": name, **metadata}
            json.dumps(info, allow_nan=False)
        except FileNotFoundError:
            abort(404, description="Video not found")
        except (ValueError, OverflowError):
            return jsonify({"error": "Video is invalid or unsupported"}), 422
        except OSError:
            return jsonify({"error": "Video metadata could not be read"}), 500
        except Exception:
            return jsonify({"error": "Video metadata could not be read"}), 500
        try:
            read_result(path)
            info["analyzed"] = True
        except (ResultNotReady, VideoBusy, OSError, ValueError, KeyError, TypeError, AttributeError):
            info["analyzed"] = False
        return jsonify(info)

    @api.get("/videos/<name>/file")
    def get_video_file(name):
        video_path(name)
        return send_from_directory(video_folder, name, mimetype="video/mp4")

    @api.get("/videos/<name>/analyze")
    def analyze_video(name):
        path = video_path(name)

        def events():
            analysis = None
            try:
                analysis = iter(services.analyze(path))
                for event in analysis:
                    if event.get("type") == "progress" and "metrics" in event:
                        event = {**event, "metrics": performance_payload(event["metrics"])}
                    yield "data: {}\n\n".format(json.dumps(event, allow_nan=False))
            except Exception as error:  # surface the failure to the browser instead of a dropped stream
                yield "data: {}\n\n".format(json.dumps({"type": "error", "message": str(error)}))
            finally:
                close = getattr(analysis, "close", None)
                if close is not None:
                    close()

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return Response(stream_with_context(events()), mimetype="text/event-stream", headers=headers)

    @api.get("/videos/<name>/result")
    def get_result_video(name):
        path = video_path(name)
        try:
            with video_operation(path):
                read_result(path, locked=True)
                # Open the response file while the run lock is held so the
                # response cannot select a newer video's bytes after validation.
                response = send_from_directory(
                    video_folder, result_video_name(name), mimetype="video/mp4",
                    as_attachment=request.args.get("download") == "1",
                    download_name=result_video_name(name),
                )
            return response
        except (ResultNotReady, VideoBusy, OSError, ValueError, KeyError, TypeError, AttributeError):
            abort(404, description="This video hasn't been analyzed yet")

    @api.get("/videos/<name>/breakdown")
    def get_breakdown(name):
        path = video_path(name)
        data_path = result_data_path(name)
        if not os.path.isfile(data_path):
            abort(404, description="This video hasn't been analyzed yet")
        try:
            data = read_result(path)
            result = build_breakdown(data["predictions"], data["video"], data["windowSize"])
            if isinstance(data.get("performance"), dict):
                result["performance"] = performance_payload(data["performance"], final=True)
            json.dumps(result, allow_nan=False)
        except (ResultNotReady, VideoBusy):
            abort(404, description="This video hasn't been analyzed yet")
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError):
            return jsonify({"error": "Analysis results could not be read; analyze the video again"}), 500
        return jsonify(result)

    return api
