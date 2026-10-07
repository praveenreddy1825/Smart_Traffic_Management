# SmartCity Sentinel

**A Multimodal Edge-Based Intelligent Traffic Management and Emergency Vehicle Prioritization System**

> Status: **Modules 0–10 complete** (foundation, video input, vehicle detection, tracking & counting,
> traffic analysis, lane analysis, audio input, siren classification, temporal smoothing,
> multimodal decision engine, emergency priority logic). Other modules are under development.
> Sections marked *(to be completed)* are filled in as modules are finished.

## 1. Project Overview

SmartCity Sentinel is an edge-based prototype that watches traffic through a camera/video and
listens for emergency sirens through a microphone. It combines both to recommend when an emergency
route or lane should be prioritized, and shows everything on a Streamlit dashboard.
Everything runs locally on a laptop acting as the edge device. No video or audio is sent to the cloud.

## 2. Problem Statement

Emergency vehicles lose critical time in congested traffic. Conventional traffic monitoring uses
video alone and has no awareness of an approaching siren. This project explores combining
**video-based traffic analysis** with **audio-based siren detection** to raise an explainable,
real-time emergency-priority recommendation.

## 3. Objectives

1. Monitor traffic from recorded video or a webcam.
2. Detect vehicles with a pretrained YOLO model.
3. Track and count unique vehicles.
4. Estimate traffic congestion (low / medium / high).
5. Analyse traffic per lane using configurable lane regions.
6. Continuously capture microphone audio.
7. Detect emergency sirens with a pretrained Hugging Face audio model.
8. Fuse audio and video information in a rule-based decision engine.
9. Activate an emergency-priority mode when a siren is confirmed.
10. Generate an explainable recommendation (recommendation only; no physical signal control).
11. Display everything on a simulated smart-city dashboard.

## 4. Architecture

```
TRAFFIC VIDEO / WEBCAM                MICROPHONE / AUDIO FILE
        |                                      |
  Video Input (OpenCV)                  Audio Input (windows)
        |                                      |
  Vehicle Detection (YOLO)              HF Audio Classification
        |                                      |
  Tracking & Counting                   Siren Detection
        |                                      |
  Traffic Analysis + Lane Analysis      Temporal Smoothing
        \                                     /
         +------> MULTIMODAL DECISION <------+
                         |
              Emergency Priority Logic
                         |
                  Recommendation
                         |
                Streamlit Dashboard
```

## 5. Project Structure

```
smart_traffic_management/
├── app/
│   ├── video/       # Module 1  - video input (OpenCV)
│   ├── detection/   # Module 2  - YOLO vehicle detection
│   ├── tracking/    # Module 3  - tracking + unique counting
│   ├── traffic/     # Module 4  - traffic density / level
│   ├── lane/        # Module 5  - lane / region analysis
│   ├── audio/       # Modules 6-8 - audio input, siren model, smoothing
│   ├── decision/    # Modules 9-10 - decision engine + priority state machine
│   ├── dashboard/   # Module 11 - Streamlit dashboard
│   └── utils/       # config loader, logging, device selection
├── config/config.yaml   # ALL tunable settings
├── data/{videos,audio,test}/
├── models/              # downloaded model weights + Hugging Face cache
├── scripts/             # helper scripts (environment check, demos)
├── tests/               # pytest tests
├── logs/                # created automatically
├── main.py
├── requirements.txt
├── README.md
└── PROJECT_NOTES.md
```

## 6. Technologies

| Purpose | Technology |
|---|---|
| Language | Python 3.12 |
| Video I/O | OpenCV |
| Vehicle detection | Ultralytics YOLO *(model chosen in Module 2)* |
| Deep learning runtime | PyTorch (CUDA when an NVIDIA GPU is present) |
| Audio I/O | sounddevice, soundfile, SciPy |
| Siren detection | Hugging Face Transformers *(model chosen in Module 7)* |
| Dashboard | Streamlit |
| Config | YAML |
| Testing | pytest |

## 7. Model Information

### Vehicle detection: YOLO11n (Ultralytics)
| | |
|---|---|
| Weights | `yolo11n.pt` (official Ultralytics release, ~5.4 MB, auto-downloaded to `models/`) |
| Pretrained on | COCO dataset, 80 classes |
| Classes used | car, motorcycle, bus, truck (person optional) |
| Input / output | BGR frame (letterboxed to 640×640) → boxes, class labels, confidence scores |
| Measured speed (this laptop) | 14.2 ms/frame on RTX 3050 GPU, 33.3 ms/frame on i5-12500H CPU |
| Limitations | No ambulance or auto-rickshaw class; small/occluded vehicles may be missed |

### Vehicle tracking: ByteTrack (via Ultralytics)
Tracking-by-detection (Zhang et al., ECCV 2022): a Kalman filter predicts each vehicle's next box, new
YOLO boxes are matched to predictions by IoU (Hungarian algorithm), first confident boxes then weak
ones. No extra neural network and no training. Settings: `config/bytetrack.yaml`.

Counting: each track ID is counted once after 10 frames (class = majority vote); optionally vehicles
crossing a virtual line are counted once with direction.

### Siren detection: Audio Spectrogram Transformer (AudioSet)
| | |
|---|---|
| Model | [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (Gong et al., Interspeech 2021, BSD-3) |
| Labels used | Siren, Civil defense siren, Emergency vehicle, Police car (siren), Ambulance (siren), Fire engine (siren) — from the model's own 527-label AudioSet list |
| Input | 16 kHz mono, 2 s windows → 128-band log-mel spectrogram |
| Output | Sigmoid probability per label; siren score = max over siren labels; threshold 0.15 |
| Measured (ESC-50 folds 3–5, 2 s windows) | accuracy 0.985, siren recall 0.958, 0 false positives in 177 non-siren windows |
| Measured latency | 96 ms/window (RTX 3050 GPU), 844 ms/window (CPU) |
| Limitations | Trained on 10 s clips; mostly Western sirens; not yet tested on real street audio through a laptop mic |

Evaluation data: ESC-50 (K. J. Piczak, ACM MM 2015, CC BY-NC 3.0), non-commercial academic use.

## 8. Installation

Requirements: Windows/Linux/macOS, Python 3.10–3.12 (developed on 3.12), about 5 GB free disk space.

```powershell
# 1. Create and activate a virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch FIRST
#    NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#    No GPU (CPU only):
#    pip install torch torchvision

# 3. Install the remaining dependencies
pip install -r requirements.txt

# 4. Verify the setup
python scripts/check_environment.py
```

Tip: if your system drive is low on space, add `--no-cache-dir` to the pip commands.
Hugging Face models are stored in `models/hf_cache` (set in `config/config.yaml`).

## 9. How to Run

```powershell
python scripts/run_video_input.py --source data/videos/traffic.mp4   # Module 1: play a video
python scripts/run_video_input.py --source 0                         # Module 1: webcam
python scripts/run_detection.py --source data/videos/traffic.mp4     # Module 2: YOLO detection
python scripts/run_detection.py --source some_photo.jpg              # Module 2: single image
python scripts/run_tracking.py                                       # Module 3: tracking + unique counting
python scripts/run_tracking.py --save                                # ... and save annotated MP4
python scripts/run_traffic_analysis.py                               # Module 4: density + LOW/MEDIUM/HIGH
python scripts/define_lanes.py --source <clip>                       # Module 5: click to draw lane regions
python scripts/define_lanes.py --source <clip> --preview             # Module 5: save image of configured lanes
python scripts/run_lane_analysis.py                                  # Module 5: per-lane counts and levels
python scripts/run_audio_input.py --list-devices                     # Module 6: list microphones
python scripts/run_audio_input.py --source mic                       # Module 6: live mic loudness meter
python scripts/make_test_audio.py                                    # Module 6: generate synthetic test sounds
python scripts/download_esc50_subset.py                              # Module 7: real siren / non-siren test clips
python scripts/run_siren_detection.py --source mic                   # Module 7: live siren classification
python scripts/evaluate_siren_classifier.py                          # Module 7: accuracy / FP / FN / latency
python scripts/make_smoothing_scenarios.py                           # Module 8: build 4 real-audio test scenarios
python scripts/run_siren_smoothing.py --evaluate                     # Module 8: score smoothing on the scenarios
python scripts/run_siren_smoothing.py --source mic                   # Module 8: live raw vs smoothed state
python scripts/run_decision_engine.py --audio data/audio/scenarios/3_continuous_siren.wav   # Module 9: video + audio decisions
python scripts/run_priority_logic.py --audio data/audio/scenarios/3_continuous_siren.wav    # Module 10: priority state machine
python main.py                       # foundation check (full pipeline in Module 12)
python -m pytest                     # run all tests
python scripts/check_environment.py  # verify libraries, GPU, webcam, microphone
```

## 10. Test Scenarios

*(to be completed in Module 13)*

## 11. Results

*(to be completed in Module 14. Only measured values will be reported.)*

## 12. Limitations

*(to be completed; see PROJECT_NOTES.md for the running list)*

## 13. Future Scope

*(to be completed in Module 15)*
