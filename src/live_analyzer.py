"""Stateful live pose classification from browser-captured JPEG frames."""

from collections import deque
import threading
import time
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .lstm import WINDOW_SIZE
from .live_errors import LiveSessionLimitError
from .utils import filter_persons
from .video_analyzer import LABELS

MAX_LIVE_SESSIONS = 4
LIVE_SESSION_TTL_S = 10 * 60
MAX_FRAME_DIMENSION = 1920


class LivePoseClassifier:
    def __init__(self, pose_detector, lstm_classifier):
        self.pose_detector = pose_detector
        self.lstm_classifier = lstm_classifier
        self.sessions = {}
        self.lock = threading.RLock()

    def start(self):
        now = time.monotonic()
        with self.lock:
            expired = [
                session_id
                for session_id, session in self.sessions.items()
                if now - session["last_active"] > LIVE_SESSION_TTL_S
            ]
            for session_id in expired:
                del self.sessions[session_id]
            if len(self.sessions) >= MAX_LIVE_SESSIONS:
                raise LiveSessionLimitError("The live classifier is busy; try again shortly.")
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = {
                "poses": deque(maxlen=WINDOW_SIZE),
                "framesSeen": 0,
                "last_active": now,
            }
        return session_id

    def process_frame(self, session_id, jpeg_bytes):
        encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame is not a valid JPEG image.")
        height, width = frame.shape[:2]
        if width > MAX_FRAME_DIMENSION or height > MAX_FRAME_DIMENSION:
            raise ValueError(f"Frame dimensions must not exceed {MAX_FRAME_DIMENSION} pixels.")

        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError("Live session not found or expired.")
            session["last_active"] = time.monotonic()
            session["framesSeen"] += 1

            outputs = self.pose_detector(frame)
            persons, _ = filter_persons(outputs)
            if not persons:
                return {
                    "framesSeen": session["framesSeen"],
                    "posesBuffered": len(session["poses"]),
                    "personDetected": False,
                    "label": None,
                    "confidence": None,
                }

            person = next(iter(persons.values()))
            features = [
                coordinate
                for row in person
                for coordinate in (float(row[0]), float(row[1]))
            ]
            label = None
            confidence = None
            if len(session["poses"]) == WINDOW_SIZE:
                model_input = torch.tensor(
                    np.asarray(session["poses"], dtype=np.float32),
                    dtype=torch.float32,
                ).unsqueeze(0)
                with torch.no_grad():
                    probabilities = F.softmax(self.lstm_classifier(model_input), dim=1)[0]
                confidence, index = torch.max(probabilities, dim=0)
                label = LABELS[int(index)]
                confidence = float(confidence)
            session["poses"].append(features)

            return {
                "framesSeen": session["framesSeen"],
                "posesBuffered": len(session["poses"]),
                "personDetected": True,
                "label": label,
                "confidence": confidence,
            }

    def stop(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                return False
            del self.sessions[session_id]
        return True
