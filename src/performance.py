"""Processing instrumentation independent of OpenCV, torch and model logic.

Memory is sampled process RSS in decimal MB (1 MB = 1,000,000 bytes), not
GPU memory or an OS lifetime high-water mark. psutil is optional.
"""

from contextlib import contextmanager
import math
import time


def performance_payload(values, *, final=False):
    """Normalize API measurements: unavailable/nonfinite values become null."""
    fields = ("totalS", "avgFrameMs", "avgPoseMs", "avgLstmMs", "avgFps", "peakMemoryMb") if final else (
        "frameMs", "poseMs", "lstmMs", "fps", "memoryMb"
    )
    values = values if isinstance(values, dict) else {}
    result = {}
    for field in fields:
        value = values.get(field)
        result[field] = value if (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
        ) else None
    if final:
        for field in ("framesProcessed", "predictionCount"):
            value = values.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[field] = value
    return result


def _memory_reader():
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process()
    except (psutil.Error, OSError):
        return None

    def read():
        try:
            return process.memory_info().rss / 1_000_000
        except (psutil.Error, OSError):
            return None

    return read


class ProcessingPerformance:
    """Per-run timings; exclude consumer/callback delays with pause/resume.

    Stage averages divide by actual stage calls; frame averages divide by
    successfully processed frames. Missing averages are omitted, not zeroed.
    Live FPS is cumulative processed frames / active elapsed time.
    """

    def __init__(self, *, clock=None, memory_reader=None):
        self.clock = clock or time.perf_counter
        self.memory_reader = memory_reader
        self.elapsed = 0.0
        self.started = self.clock()
        self.frames = 0
        self.stage_seconds = {"pose": 0.0, "lstm": 0.0}
        self.stage_counts = {"pose": 0, "lstm": 0}
        self.frame_seconds = 0.0
        self.latest = {}
        self.peak_memory = None
        self.sample_memory()

    @classmethod
    def for_process(cls):
        return cls(memory_reader=_memory_reader())

    def sample_memory(self):
        value = self.memory_reader() if self.memory_reader else None
        if value is not None and math.isfinite(value) and value >= 0:
            self.peak_memory = max(self.peak_memory or 0.0, value)
            return value
        return None

    def pause(self):
        if self.started is not None:
            self.elapsed += self.clock() - self.started
            self.started = None

    def resume(self):
        if self.started is None:
            self.started = self.clock()

    def total_seconds(self):
        return self.elapsed + (self.clock() - self.started if self.started is not None else 0.0)

    def begin_frame(self):
        self.latest = {}
        return self.clock()

    @contextmanager
    def measure(self, stage, synchronize=None):
        if stage not in self.stage_counts:
            raise ValueError(f"Unknown processing stage: {stage}")
        if synchronize:
            synchronize()
        start = self.clock()
        yield
        if synchronize:
            synchronize()
        seconds = self.clock() - start
        self.stage_seconds[stage] += seconds
        self.stage_counts[stage] += 1
        self.latest[stage + "Ms"] = seconds * 1000

    def end_frame(self, start):
        seconds = self.clock() - start
        self.frames += 1
        self.frame_seconds += seconds
        result = {"frameMs": seconds * 1000, **self.latest}
        total = self.total_seconds()
        if total > 0:
            result["fps"] = self.frames / total
        memory = self.sample_memory()
        if memory is not None:
            result["memoryMb"] = memory
        return result

    def summary(self):
        total = self.total_seconds()
        result = {
            "totalS": total,
            "framesProcessed": self.frames,
            "predictionCount": self.stage_counts["lstm"],
        }
        if self.frames:
            result["avgFrameMs"] = self.frame_seconds * 1000 / self.frames
        if total > 0:
            result["avgFps"] = self.frames / total
        for stage, count in self.stage_counts.items():
            if count:
                result["avg" + stage.title() + "Ms"] = self.stage_seconds[stage] * 1000 / count
        if self.peak_memory is not None:
            result["peakMemoryMb"] = self.peak_memory
        return result
