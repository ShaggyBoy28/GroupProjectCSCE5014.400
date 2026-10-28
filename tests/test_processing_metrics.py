"""Deterministic synthetic timings; no model, checkpoint or video files loaded."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.performance import ProcessingPerformance, _memory_reader


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class PerformanceTests(unittest.TestCase):
    def test_averages_use_actual_stage_counts_and_pause_excludes_wait(self):
        clock = Clock()
        memory = iter([100, 120, None, 110])
        tracker = ProcessingPerformance(clock=clock, memory_reader=lambda: next(memory))
        for run_lstm in (False, True):
            start = tracker.begin_frame()
            with tracker.measure("pose"):
                clock.advance(.002)
            if run_lstm:
                with tracker.measure("lstm"):
                    clock.advance(.003)
            clock.advance(.005)
            live = tracker.end_frame(start)
            tracker.pause()
            clock.advance(10)
            tracker.resume()
        tracker.sample_memory()
        tracker.pause()
        result = tracker.summary()
        self.assertAlmostEqual(result["totalS"], .017)
        self.assertAlmostEqual(result["avgFrameMs"], 8.5)
        self.assertAlmostEqual(result["avgPoseMs"], 2)
        self.assertAlmostEqual(result["avgLstmMs"], 3)
        self.assertAlmostEqual(result["avgFps"], 2 / .017)
        self.assertEqual(result["predictionCount"], 1)
        self.assertEqual(result["framesProcessed"], 2)
        self.assertEqual(result["peakMemoryMb"], 120)
        self.assertNotIn("memoryMb", live)
        json.dumps(result, allow_nan=False)

    def test_empty_run_omits_unavailable_averages_and_zero_time_fps(self):
        tracker = ProcessingPerformance(clock=Clock())
        tracker.pause()
        self.assertEqual(tracker.summary(), {
            "totalS": 0.0, "framesProcessed": 0, "predictionCount": 0,
        })

    def test_unmeasured_stage_is_not_reused_from_previous_frame(self):
        clock = Clock()
        tracker = ProcessingPerformance(clock=clock)
        start = tracker.begin_frame()
        with tracker.measure("lstm"):
            clock.advance(.01)
        self.assertIn("lstmMs", tracker.end_frame(start))
        start = tracker.begin_frame()
        self.assertNotIn("lstmMs", tracker.end_frame(start))

    def test_synchronization_includes_completed_stage_work(self):
        clock = Clock()
        tracker = ProcessingPerformance(clock=clock)
        sync = Mock(side_effect=lambda: clock.advance(.01))
        with tracker.measure("pose", sync):
            clock.advance(.02)
        self.assertEqual(sync.call_count, 2)
        self.assertAlmostEqual(tracker.summary()["avgPoseMs"], 30)

    def test_failed_stage_does_not_count_as_prediction(self):
        tracker = ProcessingPerformance(clock=Clock())
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with tracker.measure("lstm"):
                raise RuntimeError("failed")
        self.assertEqual(tracker.summary()["predictionCount"], 0)
        self.assertNotIn("avgLstmMs", tracker.summary())

    def test_memory_is_optional_and_uses_decimal_mb(self):
        with patch.dict("sys.modules", {"psutil": None}):
            self.assertIsNone(_memory_reader())
        fake = SimpleNamespace(Process=Mock(), Error=RuntimeError)
        fake.Process.return_value.memory_info.return_value.rss = 123_000_000
        with patch.dict("sys.modules", {"psutil": fake}):
            reader = _memory_reader()
            self.assertEqual(reader(), 123)
            fake.Process.return_value.memory_info.side_effect = RuntimeError("unavailable")
            self.assertIsNone(reader())
            fake.Process.side_effect = RuntimeError("unavailable")
            self.assertIsNone(_memory_reader())


def load_mocked_analyzer():
    """Load the actual analyzer source with heavy dependencies replaced."""
    cv2 = ModuleType("cv2")
    for key, value in {
        "CAP_PROP_FRAME_WIDTH": 1, "CAP_PROP_FRAME_HEIGHT": 2,
        "CAP_PROP_FRAME_COUNT": 3, "FONT_HERSHEY_COMPLEX": 4,
        "VideoWriter_fourcc": Mock(return_value=123), "putText": Mock(),
    }.items():
        setattr(cv2, key, value)
    utils = ModuleType("src.utils")
    utils.draw_keypoints = Mock()
    utils.filter_persons = Mock()
    lstm = ModuleType("src.lstm")
    lstm.WINDOW_SIZE = 32
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.Tensor = np.asarray
    torch.unsqueeze = lambda value, dim: np.expand_dims(value, axis=dim)
    torch.cuda = SimpleNamespace(synchronize=Mock())
    nn = ModuleType("torch.nn")
    nn.__path__ = []
    functional = ModuleType("torch.nn.functional")
    best = SimpleNamespace(numpy=lambda: np.array([2]))
    probabilities = SimpleNamespace(data=SimpleNamespace(max=lambda dim: (None, best)))
    functional.softmax = Mock(return_value=probabilities)
    nn.functional = functional
    torch.nn = nn
    modules = {"cv2": cv2, "src.utils": utils, "src.lstm": lstm,
               "torch": torch, "torch.nn": nn, "torch.nn.functional": functional}
    path = Path(__file__).resolve().parents[1] / "src" / "video_analyzer.py"
    spec = importlib.util.spec_from_file_location("src._metrics_test_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module, cv2, utils


class AnalyzerInstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer, self.cv2, self.utils = load_mocked_analyzer()
        print_patch = patch.object(self.analyzer, "print", create=True)
        print_patch.start()
        self.addCleanup(print_patch.stop)
        self.clock = Clock()
        self.cap = Mock()
        self.writer = Mock()
        self.detector = Mock()
        self.detector.model = None
        self.classifier = Mock()
        self.classifier.parameters.return_value = iter(())
        self.inputs = []

        def classify(window):
            self.inputs.append(window.copy())
            self.clock.advance(.003)
            return "logits"

        self.classifier.side_effect = classify
        self.writer.write.side_effect = lambda img: self.clock.advance(.004)
        self.cap.release.side_effect = lambda: self.clock.advance(.005)
        self.writer.release.side_effect = lambda: self.clock.advance(.006)

        def capture(path):
            self.clock.advance(.01)
            return self.cap

        def writer(*args):
            self.clock.advance(.02)
            return self.writer

        self.cv2.VideoCapture = Mock(side_effect=capture)
        self.cv2.VideoWriter = Mock(side_effect=writer)

        def detect(frame):
            self.clock.advance(.002)
            return int(frame[0, 0])

        self.detector.side_effect = detect
        self.utils.filter_persons = lambda frame: ({0: [[frame, frame, 1]] * 17}, [0])
        # The source imported the function before this assignment.
        self.analyzer.filter_persons = self.utils.filter_persons
        self.progress = []
        self.completed = []

    def run_video(self, frames, *, reported_frames=None, no_person=False, waits=False):
        self.cap.get.side_effect = lambda key: {
            1: 640, 2: 480, 3: frames if reported_frames is None else reported_frames,
        }[key]
        images = iter([np.full((2, 2), i) for i in range(frames)])

        def read():
            self.clock.advance(.001)
            frame = next(images, None)
            return frame is not None, frame

        self.cap.read.side_effect = read
        if no_person:
            self.analyzer.filter_persons = lambda frame: ({}, [])
        tracker = ProcessingPerformance(clock=self.clock, memory_reader=lambda: 100)

        def progress(metrics):
            self.progress.append(metrics)
            if waits:
                self.clock.advance(.5)

        def complete(metrics):
            self.completed.append(metrics)
            if waits:
                self.clock.advance(.5)

        events = []
        with patch.object(self.analyzer.ProcessingPerformance, "for_process", return_value=tracker):
            stream = self.analyzer.analyse_video(
                self.detector, self.classifier, "clip.mp4",
                on_progress=progress, on_complete=complete,
            )
            while True:
                try:
                    events.append(next(stream))
                    if waits:
                        self.clock.advance(.7)
                except StopIteration as done:
                    self.assertEqual(done.value, self.completed[0])
                    break
        return events, self.completed[0]

    def test_full_pipeline_timings_and_classification_behavior(self):
        events, result = self.run_video(34, waits=True)
        self.assertEqual(events, [f"data:{int(i * 100 / 34)}\n\n" for i in range(1, 35)])
        self.assertEqual(result["framesProcessed"], 34)
        self.assertEqual(result["predictionCount"], 2)
        self.assertAlmostEqual(result["totalS"], .286)
        self.assertAlmostEqual(result["avgFrameMs"], 244 / 34)
        self.assertAlmostEqual(result["avgPoseMs"], 2)
        self.assertAlmostEqual(result["avgLstmMs"], 3)
        self.assertAlmostEqual(result["avgFps"], 34 / .286)
        self.assertEqual(result["peakMemoryMb"], 100)
        self.assertTrue(all("lstmMs" not in p for p in self.progress[:32]))
        self.assertAlmostEqual(self.progress[32]["lstmMs"], 3)
        self.assertEqual(self.inputs[0].shape, (1, 32, 34))
        np.testing.assert_array_equal(self.inputs[0][0, :, 0], np.arange(32))
        np.testing.assert_array_equal(self.inputs[1][0, :, 0], np.arange(1, 33))
        self.assertEqual(self.cv2.putText.call_args.args[1], "Action: BOXING")
        self.cv2.VideoWriter.assert_called_once_with("res_clip.mp4", 123, 30, (640, 480))
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_short_video_has_no_lstm_average(self):
        _, result = self.run_video(32)
        self.assertEqual(result["predictionCount"], 0)
        self.assertNotIn("avgLstmMs", result)
        self.classifier.assert_not_called()

    def test_no_person_generates_no_predictions(self):
        _, result = self.run_video(40, no_person=True)
        self.assertEqual(result["framesProcessed"], 40)
        self.assertEqual(result["predictionCount"], 0)
        self.assertNotIn("avgLstmMs", result)

    def test_frame_skipping_uses_pose_call_count(self):
        self.analyzer.SKIP_FRAME_COUNT = 1
        _, result = self.run_video(65)
        self.assertEqual(result["framesProcessed"], 65)
        self.assertEqual(self.detector.call_count, 33)
        self.assertEqual(result["predictionCount"], 1)
        self.assertAlmostEqual(result["avgPoseMs"], 2)
        self.assertNotIn("poseMs", self.progress[1])
        np.testing.assert_array_equal(self.inputs[0][0, :, 0], np.arange(0, 64, 2))

    def test_empty_video_and_unknown_frame_count(self):
        events, result = self.run_video(0)
        self.assertEqual(events, [])
        self.assertEqual(result["framesProcessed"], 0)
        self.assertNotIn("avgFrameMs", result)
        self.assertNotIn("avgPoseMs", result)
        self.assertEqual(result["avgFps"], 0)

    def test_unknown_frame_count_does_not_divide_by_zero(self):
        events, result = self.run_video(2, reported_frames=0)
        self.assertEqual(events, ["data:0\n\n", "data:0\n\n"])
        self.assertEqual(result["framesProcessed"], 2)

    def test_cuda_synchronizes_only_model_device(self):
        parameter = SimpleNamespace(is_cuda=True, device="cuda:1")
        self.analyzer._synchronize_model(SimpleNamespace(parameters=lambda: iter([parameter])))
        self.analyzer.torch.cuda.synchronize.assert_called_once_with("cuda:1")

    def test_exception_releases_resources_without_success_callback(self):
        self.cap.get.return_value = 10
        self.cap.read.side_effect = RuntimeError("decode failed")
        complete = Mock()
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            list(self.analyzer.analyse_video(self.detector, self.classifier, "x.mp4", on_complete=complete))
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()
        complete.assert_not_called()

    def test_default_progress_wire_format_and_early_close_cleanup(self):
        self.cap.get.return_value = 10
        self.cap.read.return_value = (True, np.zeros((2, 2)))
        stream = self.analyzer.analyse_video(self.detector, self.classifier, "x.mp4")
        self.assertEqual(next(stream), "data:10\n\n")
        stream.close()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
