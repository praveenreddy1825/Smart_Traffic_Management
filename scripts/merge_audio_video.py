"""
Utility script to merge any traffic/emergency video file with a siren audio file
into a single presentation-ready MP4 container using PyAV.

Usage:
    python scripts/merge_audio_video.py --video data/videos/emergency/01_ambulance_corridor.mp4 --audio data/audio/scenarios/3_continuous_siren.wav --output data/videos/emergency/01_ambulance_merged.mp4
"""

import argparse
import sys
from pathlib import Path
import av


def merge_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" Multimodal Audio-Video Merger (PyAV Remuxer)")
    print("=" * 65)
    print(f" Input Video: {video_path}")
    print(f" Input Audio: {audio_path}")
    print(f" Target MP4:  {output_path}")
    print("-" * 65)

    in_v = av.open(str(video_path))
    in_a = av.open(str(audio_path))
    out_c = av.open(str(output_path), mode="w", format="mp4")

    v_in = in_v.streams.video[0]
    a_in = in_a.streams.audio[0]

    # Video stream passthrough (no re-encoding needed -> fast and lossless)
    v_out = out_c.add_stream_from_template(v_in)

    # Audio stream (standard AAC 16kHz mono, ideal for AST siren classifier)
    a_out = out_c.add_stream("aac", rate=16000)
    a_out.layout = "mono"
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

    print("[1/2] Remuxing video stream packets...")
    for packet in in_v.demux(v_in):
        if packet.dts is None:
            continue
        packet.stream = v_out
        out_c.mux(packet)

    print("[2/2] Transcoding and synchronizing audio stream...")
    for frame in in_a.decode(a_in):
        frame.pts = None
        for rf in resampler.resample(frame):
            for pkt in a_out.encode(rf):
                out_c.mux(pkt)

    for pkt in a_out.encode():
        out_c.mux(pkt)

    out_c.close()
    in_v.close()
    in_a.close()

    print(f"[+] DONE: Successfully created merged clip at:\n    {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Merge video with audio for SmartCity Sentinel presentation")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4, .mov, etc.)")
    parser.add_argument("--audio", required=True, help="Path to input audio (.wav, .mp3, etc.)")
    parser.add_argument("--output", required=True, help="Path for output merged .mp4 file")
    args = parser.parse_args()

    merge_video_audio(Path(args.video), Path(args.audio), Path(args.output))


if __name__ == "__main__":
    main()
