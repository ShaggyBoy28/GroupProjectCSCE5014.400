"""File-based result publication. JSON is the last, atomic completion record."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


class ResultNotReady(Exception):
    """There is no complete result for the current input."""


class VideoBusy(Exception):
    """Another upload or analysis owns this video's lock."""


def result_paths(input_path):
    path = Path(input_path)
    return path.with_name("res_" + path.name), path.with_name("res_" + path.stem + ".json")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invalidate_result(input_path):
    """Remove only this input's completion record; retain old video files."""
    result_paths(input_path)[1].unlink(missing_ok=True)


@contextmanager
def video_operation(input_path):
    """Cross-process exclusive lock; never remove another operation's lock."""
    lock = result_paths(input_path)[1].with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise VideoBusy("Video operation already in progress") from error
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


class ResultPublication:
    """Own unique staging files and publish metadata only after its video.

    A failed run leaves no completion record, even if the video rename succeeded.
    No broad cleanup is performed. Only this run's scratch files are unlinked.
    """

    def __init__(self, input_path):
        self.input_path = Path(input_path).resolve()
        self.video_path, self.data_path = result_paths(self.input_path)
        self.run_id = uuid.uuid4().hex
        prefix = "res_stage_" + self.run_id
        self.raw_path = self.input_path.with_name(prefix + "_raw.mp4")
        self.encoded_path = self.input_path.with_name(prefix + ".mp4")
        self.metadata_path = self.input_path.with_name(prefix + ".json.tmp")

    def __enter__(self):
        self.operation = video_operation(self.input_path)
        self.operation.__enter__()
        try:
            invalidate_result(self.input_path)
            self.input_hash = file_sha256(self.input_path)
        except BaseException:
            self.operation.__exit__(None, None, None)
            raise
        return self

    def publish(self, data, video_path):
        if file_sha256(self.input_path) != self.input_hash:
            raise ResultNotReady("Input changed during analysis")
        document = {**data, "result": {
            "runId": self.run_id,
            "inputSha256": self.input_hash,
            "videoSha256": file_sha256(video_path),
        }}
        # Write/flush the full metadata before touching the published video.
        with open(self.metadata_path, "w", encoding="utf-8") as destination:
            json.dump(document, destination, allow_nan=False)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(video_path, self.video_path)
        os.replace(self.metadata_path, self.data_path)

    def __exit__(self, *exception):
        try:
            for path in (self.raw_path, self.encoded_path, self.metadata_path):
                path.unlink(missing_ok=True)
        finally:
            self.operation.__exit__(*exception)


def read_result(input_path, *, locked=False):
    """Read a completed result bound to this input, or reject it safely.

    Legacy sidecars are supported only when neither output predates its input.
    All newly generated results require content hashes and a run identity.
    """
    if not locked:
        with video_operation(input_path):
            return read_result(input_path, locked=True)
    input_path = Path(input_path)
    video_path, data_path = result_paths(input_path)
    if not data_path.is_file():
        raise ResultNotReady("Analysis results are not ready")
    with open(data_path, encoding="utf-8") as source:
        data = json.load(source)
    if not video_path.is_file():
        raise ResultNotReady("Analysis video is not ready")
    binding = data.get("result")
    if binding is None:
        if min(data_path.stat().st_mtime_ns, video_path.stat().st_mtime_ns) < input_path.stat().st_mtime_ns:
            raise ResultNotReady("Legacy results predate input")
    elif (not isinstance(binding, dict)
          or not isinstance(binding.get("runId"), str) or len(binding["runId"]) != 32
          or any(character not in "0123456789abcdef" for character in binding["runId"])
          or binding.get("inputSha256") != file_sha256(input_path)
          or binding.get("videoSha256") != file_sha256(video_path)):
        raise ResultNotReady("Analysis results do not match input/output")
    return data
