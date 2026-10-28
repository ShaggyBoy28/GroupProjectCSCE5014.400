"""Targeted Phase 2E regressions; all media/checkpoints are temporary fixtures."""

import builtins
import copy
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_api_metrics as api_fixtures
from src.api import LABELS
from src.evaluation import MAX_COUNT, load_evaluation, validate_metrics
from src.results import ResultPublication, ResultNotReady, VideoBusy, read_result, result_paths, video_operation


class ResultApiRegressions(unittest.TestCase):
    setUp = api_fixtures.ApiMetricsTests.setUp
    save_result = api_fixtures.ApiMetricsTests.save_result

    def test_replacement_invalidates_old_predictions_metrics_and_video(self):
        path = self.folder / "clip.mp4"
        path.write_bytes(b"first input")
        with ResultPublication(path) as publication:
            publication.raw_path.write_bytes(b"first annotated video")
            publication.publish({"video": {"name": path.name, **self.metadata}, "windowSize": 32,
                                 "predictions": [], "performance": {"totalS": 99}}, publication.raw_path)
        self.assertTrue(self.client.get("/api/videos/clip.mp4").json["analyzed"])
        response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"replacement"), "clip.mp4")})
        self.assertEqual((response.status_code, response.json), (201, {"name": "clip.mp4"}))
        self.assertEqual(path.read_bytes(), b"replacement")
        self.assertFalse(self.client.get("/api/videos/clip.mp4").json["analyzed"])
        self.assertEqual(self.client.get("/api/videos/clip.mp4/breakdown").status_code, 404)
        self.assertEqual(self.client.get("/api/videos/clip.mp4/result").status_code, 404)
        # Old video retained, never exposed as a completed replacement analysis.
        self.assertEqual(result_paths(path)[0].read_bytes(), b"first annotated video")

    def test_legacy_result_is_invalidated_on_replacement_too(self):
        self.save_result({"totalS": 99})
        path = self.folder / "clip.mp4"
        path.write_bytes(b"old")
        (self.folder / "res_sample_video.json").replace(self.folder / "res_clip.json")
        (self.folder / "res_sample_video.mp4").replace(self.folder / "res_clip.mp4")
        response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"new"), "clip.mp4")})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.get("/api/videos/clip.mp4/breakdown").status_code, 404)

    def test_failed_upload_keeps_original_input_and_hides_old_results(self):
        path = self.folder / "clip.mp4"
        path.write_bytes(b"old")
        (self.folder / "res_clip.json").write_text("{}")
        with patch("src.api.os.replace", side_effect=OSError("private filesystem details")):
            response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"new"), "clip.mp4")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(path.read_bytes(), b"old")
        self.assertNotIn("private", response.get_data(as_text=True))
        self.assertFalse((self.folder / "res_clip.json").exists())
        self.assertEqual(list(self.folder.glob("res_upload_*")), [])

    def test_upload_rejected_while_same_video_has_active_operation(self):
        path = self.folder / "clip.mp4"
        path.write_bytes(b"old")
        with video_operation(path):
            response = self.client.post("/api/videos", data={"video": (io.BytesIO(b"new"), "clip.mp4")})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(path.read_bytes(), b"old")

    def test_metadata_errors_are_generic_json_with_appropriate_status(self):
        for error, status in [(ValueError("private decoder details"), 422),
                              (OverflowError("private overflow details"), 422),
                              (FileNotFoundError("private path"), 404),
                              (OSError("private filesystem details"), 500),
                              (RuntimeError("private decoder failure"), 500)]:
            with self.subTest(error=type(error).__name__):
                self.services.probe.side_effect = error
                response = self.client.get("/api/videos/sample_video.mp4")
                self.assertEqual(response.status_code, status)
                self.assertTrue(response.is_json)
                self.assertIn("error", response.json)
                self.assertNotIn("private", response.get_data(as_text=True))

    def test_nonfinite_metadata_is_rejected_as_json(self):
        for value in (float("nan"), float("inf"), -1, 10**400):
            with self.subTest(value=value):
                self.services.probe.return_value = {**self.metadata, "fps": value}
                response = self.client.get("/api/videos/sample_video.mp4")
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.is_json)

    def test_result_bytes_cannot_be_changed_without_invalidating_binding(self):
        path = self.folder / "sample_video.mp4"
        with ResultPublication(path) as publication:
            publication.raw_path.write_bytes(b"result")
            publication.publish({"video": {"name": path.name, **self.metadata}, "windowSize": 32,
                                 "predictions": []}, publication.raw_path)
        result_paths(path)[0].write_bytes(b"different output")
        self.assertFalse(self.client.get("/api/videos/sample_video.mp4").json["analyzed"])
        self.assertEqual(self.client.get("/api/videos/sample_video.mp4/result").status_code, 404)
        self.assertEqual(self.client.get("/api/videos/sample_video.mp4/breakdown").status_code, 404)


class PublicationFailureRegressions(unittest.TestCase):
    setUp = api_fixtures.ApiMetricsTests.setUp

    def test_publish_video_or_metadata_rename_failure_leaves_no_complete_result(self):
        path = self.folder / "sample_video.mp4"
        for fail_number in (1, 2):
            with self.subTest(rename=fail_number):
                old_video, data_path = result_paths(path)
                old_video.write_bytes(b"old output")
                data_path.write_text("{}")
                rename = os.replace
                calls = []
                def fail_one(source, destination):
                    calls.append((source, destination))
                    if len(calls) == fail_number:
                        raise OSError("mock publication failure")
                    return rename(source, destination)
                with self.assertRaisesRegex(OSError, "mock publication failure"):
                    with ResultPublication(path) as publication:
                        publication.raw_path.write_bytes(b"new output")
                        with patch("src.results.os.replace", side_effect=fail_one):
                            publication.publish({"predictions": [], "performance": {"totalS": 1}}, publication.raw_path)
                self.assertFalse(data_path.exists())
                self.assertEqual(list(self.folder.glob("res_stage_*")), [])
                self.assertEqual(list(self.folder.glob("*.lock")), [])
                with self.assertRaises(ResultNotReady):
                    read_result(path)
                self.assertEqual(self.client.get("/api/videos/sample_video.mp4/result").status_code, 404)

    def test_metadata_write_failure_keeps_old_video_unpublished(self):
        path = self.folder / "sample_video.mp4"
        old_video, data_path = result_paths(path)
        old_video.write_bytes(b"old output")
        data_path.write_text("{}")
        def fail_write(target, mode="r", *args, **kwargs):
            if "w" in mode:
                raise OSError("mock metadata write failure")
            return builtins.open(target, mode, *args, **kwargs)
        with self.assertRaisesRegex(OSError, "mock metadata write failure"):
            with ResultPublication(path) as publication:
                publication.raw_path.write_bytes(b"new output")
                with patch("src.results.open", side_effect=fail_write, create=True):
                    publication.publish({"predictions": []}, publication.raw_path)
        self.assertEqual(old_video.read_bytes(), b"old output")
        self.assertFalse(data_path.exists())

    def test_run_binding_and_lock_protect_reads_until_commit(self):
        path = self.folder / "sample_video.mp4"
        with ResultPublication(path) as publication:
            publication.raw_path.write_bytes(b"new output")
            publication.publish({"predictions": [], "performance": {"totalS": 1}}, publication.raw_path)
            with self.assertRaises(VideoBusy):
                read_result(path)
            run_id = publication.run_id
        data = read_result(path)
        self.assertEqual(data["result"]["runId"], run_id)
        self.assertEqual(len(data["result"]["inputSha256"]), 64)
        path.write_bytes(b"changed input outside API")
        with self.assertRaises(ResultNotReady):
            read_result(path)

    def test_input_changed_during_run_prevents_publication(self):
        path = self.folder / "sample_video.mp4"
        with ResultPublication(path) as publication:
            publication.raw_path.write_bytes(b"output")
            path.write_bytes(b"input changed outside API")
            with self.assertRaises(ResultNotReady):
                publication.publish({"predictions": []}, publication.raw_path)
        self.assertFalse(result_paths(path)[1].exists())


class EvaluationValidationRegressions(unittest.TestCase):
    setUp = api_fixtures.EvaluationArtifactTests.setUp
    load = api_fixtures.EvaluationArtifactTests.load

    def test_nan_infinity_and_overflow_exponents_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1e308):
            with self.subTest(value=value):
                artifact = copy.deepcopy(self.artifact)
                artifact["metrics"]["accuracy"] = value
                with self.assertRaises(ValueError):
                    self.load(artifact)
        text = json.dumps(self.artifact).replace('"accuracy": 0.5', '"accuracy": 1e999')
        self.path.write_text(text)
        with self.assertRaises(ValueError):
            load_evaluation(self.path, self.checkpoint, LABELS, 32)

    def test_large_integer_scores_and_counts_do_not_overflow(self):
        for field in ("accuracy", "sampleCount"):
            for value in (10**400, MAX_COUNT + 1):
                with self.subTest(field=field, value=value):
                    artifact = copy.deepcopy(self.artifact)
                    artifact["metrics"][field] = value
                    with self.assertRaises(ValueError):
                        self.load(artifact)
        metrics = copy.deepcopy(self.artifact["metrics"])
        metrics["accuracy"] = 10**400
        with self.assertRaises(ValueError):
            validate_metrics(metrics, LABELS)

    def test_unknown_fields_rejected_at_every_level(self):
        for level in ("artifact", "metrics", "perClass"):
            with self.subTest(level=level):
                artifact = copy.deepcopy(self.artifact)
                target = artifact if level == "artifact" else artifact["metrics"]
                if level == "perClass":
                    target = target["perClass"][0]
                target["unsupported"] = 1
                with self.assertRaises(ValueError):
                    self.load(artifact)

    def test_unknown_nonfinite_fields_cannot_escape_into_json(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["metrics"]["debug"] = float("nan")
        with self.assertRaises(ValueError):
            self.load(artifact)

    def test_validated_output_is_detached_and_standard_json(self):
        metrics = self.artifact["metrics"]
        result = validate_metrics(metrics, LABELS)
        metrics["perClass"][0]["precision"] = float("nan")
        metrics["confusionMatrix"][0][0] = 999
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["confusionMatrix"][0][0], 1)


class ModelFallbackRegressions(unittest.TestCase):
    setUp = api_fixtures.ApiMetricsTests.setUp

    def test_invalid_provider_results_and_failures_return_null_json(self):
        for supplied in ({"accuracy": float("nan")}, {"accuracy": 10**400}, [], "invalid"):
            self.services.model_metrics = lambda supplied=supplied: supplied
            response = self.client.get("/api/model")
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json["metrics"])
            json.loads(response.get_data(as_text=True), parse_constant=lambda value: self.fail(value))
        self.services.model_metrics = lambda: (_ for _ in ()).throw(OverflowError("private details"))
        self.assertIsNone(self.client.get("/api/model").json["metrics"])


if __name__ == "__main__":
    unittest.main()
