import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import torch
from flask import Flask

from src.api import create_api_blueprint
from src.live_analyzer import LivePoseClassifier
from src.live_errors import LiveSessionLimitError


def jpeg_frame():
    success, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    if not success:
        raise RuntimeError("Could not encode test frame")
    return encoded.tobytes()


class LivePoseClassifierTests(unittest.TestCase):
    def setUp(self):
        keypoints = torch.zeros((1, 17, 3))
        keypoints[0, :, 0] = torch.arange(17)
        keypoints[0, :, 1] = torch.arange(17) + 10
        self.detector = Mock(return_value={
            "instances": SimpleNamespace(
                scores=torch.tensor([0.99]),
                pred_keypoints=keypoints,
            ),
        })
        self.classifier = Mock(return_value=torch.tensor([[-2.0, 0.0, 4.0, 0.0, 0.0, 0.0]]))
        self.live = LivePoseClassifier(self.detector, self.classifier)
        self.session_id = self.live.start()

    def test_buffers_32_poses_then_returns_predictions(self):
        result = None
        for _ in range(32):
            result = self.live.process_frame(self.session_id, jpeg_frame())
        self.assertEqual(result["posesBuffered"], 32)
        self.assertIsNone(result["label"])

        result = self.live.process_frame(self.session_id, jpeg_frame())
        self.assertEqual(result["framesSeen"], 33)
        self.assertEqual(result["posesBuffered"], 32)
        self.assertEqual(result["label"], "BOXING")
        self.assertGreater(result["confidence"], 0.9)
        self.classifier.assert_called_once()
        model_input = self.classifier.call_args.args[0]
        self.assertEqual(tuple(model_input.shape), (1, 32, 34))
        self.assertEqual(model_input[0, 0, 0].item(), 0)

    def test_missing_person_does_not_add_a_pose(self):
        self.detector.return_value = {
            "instances": SimpleNamespace(
                scores=torch.tensor([]),
                pred_keypoints=torch.empty((0, 17, 3)),
            ),
        }
        result = self.live.process_frame(self.session_id, jpeg_frame())
        self.assertFalse(result["personDetected"])
        self.assertEqual(result["posesBuffered"], 0)
        self.classifier.assert_not_called()

    def test_invalid_frame_and_unknown_session_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.live.process_frame(self.session_id, b"not a jpeg")
        with self.assertRaises(KeyError):
            self.live.process_frame("f" * 32, jpeg_frame())

    def test_session_limit_and_stop(self):
        other_sessions = [self.live.start() for _ in range(3)]
        with self.assertRaises(LiveSessionLimitError):
            self.live.start()
        self.assertTrue(self.live.stop(self.session_id))
        self.assertFalse(self.live.stop(self.session_id))
        self.assertEqual(len(other_sessions), 3)


class LiveClassificationApiTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.session_id = "a" * 32
        self.services = SimpleNamespace(
            start_live_session=Mock(return_value=self.session_id),
            classify_live_frame=Mock(return_value={
                "framesSeen": 33,
                "posesBuffered": 32,
                "personDetected": True,
                "label": "BOXING",
                "confidence": 0.95,
            }),
            stop_live_session=Mock(return_value=True),
        )
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_api_blueprint(self.services, str(self.folder)))
        self.client = app.test_client()

    def test_start_classify_and_stop_live_session(self):
        response = self.client.post("/api/live/sessions")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json, {"sessionId": self.session_id, "windowSize": 32})

        frame = jpeg_frame()
        response = self.client.post(
            f"/api/live/sessions/{self.session_id}/frames",
            data=frame,
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["label"], "BOXING")
        self.services.classify_live_frame.assert_called_once_with(self.session_id, frame)

        response = self.client.delete(f"/api/live/sessions/{self.session_id}")
        self.assertEqual(response.status_code, 204)
        self.services.stop_live_session.assert_called_once_with(self.session_id)

    def test_frame_validation_and_expired_session(self):
        invalid_type = self.client.post(
            f"/api/live/sessions/{self.session_id}/frames",
            data=b"not jpeg",
            content_type="application/octet-stream",
        )
        self.assertEqual(invalid_type.status_code, 415)
        self.assertIn("error", invalid_type.json)

        invalid_id = self.client.post(
            "/api/live/sessions/not-a-session/frames",
            data=io.BytesIO(jpeg_frame()),
            content_type="image/jpeg",
        )
        self.assertEqual(invalid_id.status_code, 404)

        self.services.classify_live_frame.side_effect = KeyError("expired")
        expired = self.client.post(
            f"/api/live/sessions/{self.session_id}/frames",
            data=jpeg_frame(),
            content_type="image/jpeg",
        )
        self.assertEqual(expired.status_code, 404)
        self.assertIn("error", expired.json)

    def test_busy_server_returns_too_many_requests(self):
        self.services.start_live_session.side_effect = LiveSessionLimitError("busy")
        response = self.client.post("/api/live/sessions")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json, {"error": "busy"})
