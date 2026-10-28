"""Exercise real React inference generator with simulated video/model stages."""

from contextlib import nullcontext
import json
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import builtins
import unittest

import numpy as np

from src.performance import ProcessingPerformance
import test_processing_metrics as fixtures


class ReactProcessingTests(unittest.TestCase):
    # Reuse only fixture construction, without duplicating its legacy tests.
    def setUp(self):
        fixtures.AnalyzerInstrumentationTests.setUp(self)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.input_path = Path(temporary.name) / "clip.mp4"
        self.input_path.write_bytes(b"synthetic input video")

    def run_react(self, frames, *, memory=True, callback_error=False, write_error=False):
        info = {"fps": 30, "frameCount": frames, "width": 640, "height": 480, "durationS": frames / 30}
        images = iter([np.full((2, 2), i) for i in range(frames)])
        def read():
            self.clock.advance(.001)
            frame = next(images, None)
            return frame is not None, frame
        self.cap.read.side_effect = read
        self.cap.isOpened.return_value = True
        self.writer.isOpened.return_value = True
        tracker = ProcessingPerformance(clock=self.clock, memory_reader=(lambda: 123) if memory else None)
        self.analyzer.torch.no_grad = lambda: nullcontext()
        self.analyzer.F.softmax.return_value = np.array([[.05, .05, .7, .05, .05, .1]])
        events = []
        complete = Mock(side_effect=lambda summary: self.completed.append(summary))
        def progress(metrics):
            if callback_error:
                raise RuntimeError("mock callback failure")
            self.progress.append(metrics)
            self.clock.advance(.5)
        def transcode(*args):
            self.clock.advance(.008)
            Path(args[0]).write_bytes(b"synthetic output video")
            return False
        def fail_metadata(path, mode="r", *args, **kwargs):
            if "w" in mode:
                raise OSError("mock persistence failure")
            return builtins.open(path, mode, *args, **kwargs)
        with patch.object(self.analyzer, "probe_video", return_value=info), \
                patch.object(self.analyzer.ProcessingPerformance, "for_process", return_value=tracker), \
                patch.object(self.analyzer, "transcode_for_browser", side_effect=transcode), \
                patch("src.results.os.replace", wraps=os.replace) as replace, \
                (patch("src.results.open", side_effect=fail_metadata, create=True) if write_error else nullcontext()):
            self.replacements = replace
            self.completion_callback = complete
            stream = self.analyzer.analyse_video_events(
                self.detector, self.classifier, str(self.input_path), on_progress=progress, on_complete=complete,
            )
            for event in stream:
                if event["type"] == "done":
                    self.assertEqual(replace.call_count, 2)
                    complete.assert_called_once()
                    self.writer.release.assert_called_once()
                events.append(event)
                self.clock.advance(.7)
        data = json.loads(self.input_path.with_name("res_clip.json").read_text(encoding="utf-8"))
        return events, data

    def test_live_and_final_metrics_match_actual_work_and_exclude_waits(self):
        events, data = self.run_react(34)
        self.assertEqual(events[-1], {"type": "done"})
        self.assertEqual(len(events), 35)
        self.assertEqual(events[31]["label"], None)
        self.assertEqual(events[32]["label"], "BOXING")
        self.assertIsNone(events[31]["metrics"]["lstmMs"])
        self.assertAlmostEqual(events[32]["metrics"]["lstmMs"], 3)
        result = data["performance"]
        self.assertEqual(result, self.completed[0])
        self.assertEqual(result["framesProcessed"], 34)
        self.assertEqual(result["predictionCount"], 2)
        self.assertAlmostEqual(result["totalS"], .294)
        self.assertAlmostEqual(result["avgFrameMs"], 244 / 34)
        self.assertAlmostEqual(result["avgPoseMs"], 2)
        self.assertAlmostEqual(result["avgLstmMs"], 3)
        self.assertAlmostEqual(result["avgFps"], 34 / .294)
        self.assertEqual(result["peakMemoryMb"], 123)
        self.assertEqual([p["frame"] for p in data["predictions"]], [32, 33])
        self.assertEqual(data["predictions"][0]["probabilities"][2], .7)
        np.testing.assert_array_equal(self.inputs[0][0, :, 0], np.arange(32))
        self.cap.release.assert_called_once()

    def test_short_video_and_unavailable_memory_return_nulls(self):
        events, data = self.run_react(2, memory=False)
        self.assertEqual(data["predictions"], [])
        self.assertIsNone(data["performance"]["avgLstmMs"])
        self.assertIsNone(data["performance"]["peakMemoryMb"])
        self.assertIsNone(events[0]["metrics"]["memoryMb"])
        self.assertEqual(data["performance"]["predictionCount"], 0)

    def test_empty_video_has_no_invented_stage_averages(self):
        events, data = self.run_react(0)
        self.assertEqual(events, [{"type": "done"}])
        self.assertIsNone(data["performance"]["avgFrameMs"])
        self.assertIsNone(data["performance"]["avgPoseMs"])
        self.assertIsNone(data["performance"]["avgLstmMs"])

    def test_progress_callback_failure_releases_resources(self):
        with self.assertRaisesRegex(RuntimeError, "mock callback failure"):
            self.run_react(2, callback_error=True)
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.completion_callback.assert_not_called()

    def test_persistence_failure_does_not_announce_completion(self):
        with self.assertRaisesRegex(OSError, "mock persistence failure"):
            self.run_react(2, write_error=True)
        self.completion_callback.assert_not_called()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_closed_stream_cleans_up_without_publishing(self):
        info = {"fps": 30, "frameCount": 1, "width": 640, "height": 480, "durationS": 1 / 30}
        self.cap.read.return_value = (True, np.zeros((2, 2)))
        with patch.object(self.analyzer, "probe_video", return_value=info), \
                patch.object(self.analyzer, "transcode_for_browser") as transcode:
            stream = self.analyzer.analyse_video_events(self.detector, self.classifier, str(self.input_path))
            self.assertEqual(next(stream)["type"], "progress")
            stream.close()
            transcode.assert_not_called()
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_unopened_video_is_rejected_and_released(self):
        self.cap.isOpened.return_value = False
        with self.assertRaisesRegex(ValueError, "Could not open input video"):
            self.analyzer.probe_video("invalid.mp4")
        self.cap.release.assert_called_once()

    def test_unopened_writer_is_rejected_and_all_resources_released(self):
        info = {"fps": 30, "frameCount": 1, "width": 640, "height": 480, "durationS": 1 / 30}
        self.writer.isOpened.return_value = False
        with patch.object(self.analyzer, "probe_video", return_value=info):
            with self.assertRaisesRegex(ValueError, "Could not open output video writer"):
                list(self.analyzer.analyse_video_events(self.detector, self.classifier, str(self.input_path)))
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
