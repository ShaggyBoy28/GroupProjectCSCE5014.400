"""API contract tests with synthetic videos/evaluations and injected services."""

import copy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from flask import Flask

from src.api import LABELS, create_api_blueprint
from src.evaluation import load_evaluation
from src.metrics import calculate_classification_metrics


class ApiMetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / "sample_video.mp4").write_bytes(b"mock video")
        self.metadata = {"fps": 30, "frameCount": 34, "width": 640, "height": 480, "durationS": 34 / 30}
        self.services = SimpleNamespace(
            probe=Mock(return_value=self.metadata),
            analyze=Mock(return_value=iter([{"type": "done"}])),
        )
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.register_blueprint(create_api_blueprint(self.services, str(self.folder)))
        self.client = self.app.test_client()

    def save_result(self, performance=None):
        data = {"video": {"name": "sample_video.mp4", **self.metadata}, "windowSize": 32,
                "predictions": [{"frame": 32, "probabilities": [1, 0, 0, 0, 0, 0]}]}
        if performance is not None:
            data["performance"] = performance
        (self.folder / "res_sample_video.json").write_text(json.dumps(data))
        (self.folder / "res_sample_video.mp4").write_bytes(b"mock output")

    def events(self):
        response = self.client.get("/api/videos/sample_video.mp4/analyze")
        self.assertEqual(response.mimetype, "text/event-stream")
        self.assertEqual(response.headers["Cache-Control"], "no-cache")
        return [json.loads(part.removeprefix("data: ")) for part in response.get_data(as_text=True).split("\n\n") if part]

    def test_model_information_without_evaluation(self):
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"labels": LABELS, "windowSize": 32, "metrics": None})
        self.services.probe.assert_not_called()
        self.services.analyze.assert_not_called()

    def test_model_explicitly_missing_evaluation(self):
        self.services.model_metrics = lambda: None
        self.assertIsNone(self.client.get("/api/model").json["metrics"])

    def test_model_receives_reviewed_evaluation_service_results(self):
        # Synthetic known-answer fixture, never published as real model results.
        metrics = {"dataset": "SYNTHETIC TEST FIXTURE", **calculate_classification_metrics(range(6), range(6))}
        self.services.model_metrics = lambda: metrics
        self.assertEqual(self.client.get("/api/model").json["metrics"], metrics)

    def test_progress_metrics_preserve_existing_fields_and_done(self):
        metrics = {"frameMs": 10, "poseMs": 7, "lstmMs": 1, "fps": 50, "memoryMb": 123}
        event = {"type": "progress", "percent": 99, "frame": 33, "totalFrames": 34, "label": "BOXING", "metrics": metrics}
        self.services.analyze.return_value = iter([event, {"type": "done"}])
        self.assertEqual(self.events(), [event, {"type": "done"}])

    def test_missing_and_invalid_live_measurements_are_null(self):
        self.services.analyze.return_value = iter([
            {"type": "progress", "metrics": {"frameMs": 10, "poseMs": -1, "memoryMb": float("nan"), "fps": True}},
            {"type": "done"},
        ])
        self.assertEqual(self.events()[0]["metrics"], {
            "frameMs": 10, "poseMs": None, "lstmMs": None, "fps": None, "memoryMb": None,
        })

    def test_older_progress_events_remain_unchanged(self):
        event = {"type": "progress", "percent": 10, "frame": 3, "totalFrames": 34, "label": None}
        self.services.analyze.return_value = iter([event, {"type": "done"}])
        self.assertEqual(self.events()[0], event)

    def test_full_performance_summary_in_breakdown(self):
        performance = {"totalS": 1, "avgFrameMs": 20, "avgPoseMs": 15, "avgLstmMs": 1,
                       "avgFps": 34, "peakMemoryMb": 123, "framesProcessed": 34, "predictionCount": 1}
        self.save_result(performance)
        result = self.client.get("/api/videos/sample_video.mp4/breakdown").json
        self.assertEqual(result["performance"], performance)
        self.assertEqual(result["summary"]["mainActivity"], "JUMPING")
        self.assertEqual(result["predictions"][0]["confidence"], 1)

    def test_partial_final_measurements_are_not_fabricated(self):
        self.save_result({"totalS": 1, "avgPoseMs": float("inf"), "predictionCount": 0})
        result = self.client.get("/api/videos/sample_video.mp4/breakdown").json["performance"]
        self.assertEqual(result, {"totalS": 1, "avgFrameMs": None, "avgPoseMs": None,
                                  "avgLstmMs": None, "avgFps": None, "peakMemoryMb": None, "predictionCount": 0})

    def test_old_sidecar_without_performance_still_works(self):
        self.save_result()
        response = self.client.get("/api/videos/sample_video.mp4/breakdown")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("performance", response.json)

    def test_existing_health_sample_upload_file_and_download(self):
        self.assertEqual(self.client.get("/api/health").json, {"status": "ok"})
        self.assertFalse(self.client.get("/api/sample").json["analyzed"])
        response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"mock"), "new.mp4")})
        self.assertEqual((response.status_code, response.json), (201, {"name": "new.mp4"}))
        self.assertEqual(self.client.get("/api/videos/new.mp4").json["name"], "new.mp4")
        with self.client.get("/api/videos/new.mp4/file") as response:
            self.assertEqual(response.data, b"mock")
        self.save_result()
        with self.client.get("/api/videos/sample_video.mp4/result?download=1") as response:
            self.assertEqual(response.data, b"mock output")
            self.assertIn("attachment", response.headers["Content-Disposition"])

    def test_error_responses(self):
        self.assertEqual(self.client.post("/api/videos").status_code, 400)
        for name in ("bad.txt", "sample_video.mp4", "res_bad.mp4"):
            response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"mock"), name)})
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json)
        for suffix in ("", "/breakdown", "/result", "/analyze"):
            response = self.client.get("/api/videos/missing.mp4" + suffix)
            self.assertEqual(response.status_code, 404)
            self.assertIn("error", response.json)
        self.assertEqual(self.client.get("/api/videos/sample_video.mp4/breakdown").status_code, 404)

    def test_corrupt_sidecar_returns_json_error(self):
        (self.folder / "res_sample_video.json").write_text("not json")
        response = self.client.get("/api/videos/sample_video.mp4/breakdown")
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)

    def test_malformed_performance_is_omitted_for_compatibility(self):
        self.save_result("not a measurement dictionary")
        response = self.client.get("/api/videos/sample_video.mp4/breakdown")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("performance", response.json)

    def test_analysis_failure_emits_error_without_done(self):
        def analyze(path):
            yield {"type": "progress", "percent": 1}
            raise RuntimeError("mock inference failure")
        self.services.analyze = analyze
        self.assertEqual(self.events(), [{"type": "progress", "percent": 1},
                                         {"type": "error", "message": "mock inference failure"}])

    def test_disconnect_closes_underlying_analysis(self):
        closed = []
        def analyze(path):
            try:
                yield {"type": "progress", "percent": 1}
                yield {"type": "done"}
            finally:
                closed.append(True)
        self.services.analyze = analyze
        response = self.client.get("/api/videos/sample_video.mp4/analyze", buffered=False)
        response.close()
        self.assertEqual(closed, [True])


class EvaluationArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.checkpoint = self.folder / "mock.ckpt"
        self.checkpoint.write_bytes(b"synthetic checkpoint identity")
        self.path = self.folder / "evaluation.json"
        self.artifact = {
            "checkpointSha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "labels": LABELS, "windowSize": 32, "split": "synthetic unit-test split",
            "provenance": "SYNTHETIC TEST FIXTURE ONLY",
            "metrics": {"dataset": "SYNTHETIC TEST FIXTURE", **calculate_classification_metrics([0, 0, 1, 2], [0, 1, 1, 0])},
        }

    def load(self, artifact=None):
        self.path.write_text(json.dumps(artifact if artifact is not None else self.artifact))
        return load_evaluation(str(self.path), self.checkpoint, LABELS, 32)

    def test_unconfigured_returns_null_without_reading_checkpoint(self):
        self.assertIsNone(load_evaluation(None, "missing.ckpt", LABELS, 32))

    def test_matching_consistent_artifact(self):
        self.assertEqual(self.load(), self.artifact["metrics"])

    def test_identity_and_provenance_must_match(self):
        for key, value in [("checkpointSha256", "wrong"), ("labels", LABELS[::-1]),
                           ("windowSize", 1), ("split", ""), ("provenance", "")]:
            with self.subTest(key=key):
                artifact = {**self.artifact, key: value}
                with self.assertRaises(ValueError):
                    self.load(artifact)

    def test_invalid_or_inconsistent_scores_and_counts_rejected(self):
        for key, value in [("accuracy", .9), ("macroF1", float("nan")), ("sampleCount", 0),
                           ("confusionMatrix", [[1]]), ("dataset", ""), ("perClass", [])]:
            with self.subTest(key=key):
                artifact = copy.deepcopy(self.artifact)
                artifact["metrics"][key] = value
                with self.assertRaises(ValueError):
                    self.load(artifact)


if __name__ == "__main__":
    unittest.main()
