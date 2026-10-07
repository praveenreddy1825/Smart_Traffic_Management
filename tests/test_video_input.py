"""Tests for Module 1: VideoInput.

A small synthetic video (a white box moving across a grey background) is
generated for each test, so the tests do not depend on any downloaded file.
"""

import time

import cv2
import numpy as np
import pytest

from app.utils.config import load_config
from app.video.video_input import VideoFrame, VideoInput, VideoSourceError, parse_source

VIDEO_WIDTH, VIDEO_HEIGHT = 320, 240
VIDEO_FPS = 50
VIDEO_FRAMES = 20


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    assert writer.isOpened(), "OpenCV could not create the test video"
    for i in range(VIDEO_FRAMES):
        image = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), 80, dtype=np.uint8)
        x = 10 + i * 10
        cv2.rectangle(image, (x, 100), (x + 40, 140), (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    return path


def test_parse_source_webcam_and_file():
    assert parse_source(0) == 0
    assert parse_source("1") == 1
    assert parse_source("data/videos/x.mp4").is_absolute()


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(VideoSourceError, match="not found"):
        VideoInput(tmp_path / "nope.mp4").open()


def test_reads_every_frame_with_correct_index_and_timestamp(synthetic_video):
    video = VideoInput(synthetic_video, max_frame_size=None, realtime_playback=False)
    frames = list(video.frames())

    assert len(frames) == VIDEO_FRAMES
    assert all(isinstance(frame, VideoFrame) for frame in frames)
    assert [frame.index for frame in frames] == list(range(VIDEO_FRAMES))
    assert frames[10].timestamp_sec == pytest.approx(10 / VIDEO_FPS)
    assert video.fps == pytest.approx(VIDEO_FPS)
    assert video.frame_count == VIDEO_FRAMES
    assert not video.is_opened  # released automatically at the end


def test_resize_keeps_aspect_ratio(synthetic_video):
    video = VideoInput(synthetic_video, max_frame_size=160, realtime_playback=False)
    first_frame = next(video.frames())
    assert first_frame.image.shape == (120, 160, 3)
    assert video.original_size == (VIDEO_WIDTH, VIDEO_HEIGHT)
    assert video.output_size == (160, 120)
    video.release()


def test_portrait_frames_limited_by_longer_side(tmp_path):
    path = tmp_path / "portrait.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 320))
    for _ in range(3):
        writer.write(np.full((320, 240, 3), 100, dtype=np.uint8))
    writer.release()

    video = VideoInput(path, max_frame_size=160, realtime_playback=False)
    first_frame = next(video.frames())
    assert first_frame.image.shape == (160, 120, 3)  # height is the longer side
    video.release()


def test_small_frames_are_never_enlarged(synthetic_video):
    video = VideoInput(synthetic_video, max_frame_size=960, realtime_playback=False)
    first_frame = next(video.frames())
    assert first_frame.image.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    video.release()


def test_realtime_playback_respects_source_fps(synthetic_video):
    video = VideoInput(synthetic_video, realtime_playback=True)
    start = time.perf_counter()
    count = sum(1 for _ in video.frames())
    elapsed = time.perf_counter() - start
    # 20 frames at 50 FPS: the last frame is due at 19/50 = 0.38 s
    assert count == VIDEO_FRAMES
    assert elapsed >= (VIDEO_FRAMES - 1) / VIDEO_FPS * 0.95


def test_loop_video_restarts_and_keeps_counting(synthetic_video):
    video = VideoInput(synthetic_video, realtime_playback=False, loop_video=True)
    video.open()
    frames = [video.read() for _ in range(VIDEO_FRAMES * 2 + 5)]
    video.release()
    assert all(frame is not None for frame in frames)
    assert frames[-1].index == VIDEO_FRAMES * 2 + 4


def test_read_before_open_raises(synthetic_video):
    with pytest.raises(VideoSourceError, match="not open"):
        VideoInput(synthetic_video).read()


def test_from_config_uses_video_section():
    config = load_config()
    video = VideoInput.from_config(config, source="0")
    assert video.is_webcam
    assert video.max_frame_size == config["video"]["max_frame_size"]
