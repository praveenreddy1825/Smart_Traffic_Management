# SmartCity Sentinel

**A Multimodal Edge-Based Intelligent Traffic Management and Emergency Vehicle Prioritization System**

> Status: **Modules 0–15 complete** (foundation, video input, vehicle detection, tracking & counting,
> traffic analysis, lane analysis, audio input, siren classification, temporal smoothing,
> multimodal decision engine, emergency priority logic, multi-approach 4-way dashboard,
> end-to-end scenario evaluation, performance benchmarking, and edge profiling). All 134 automated tests passing.

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
| Dashboard | FastAPI + HTML5 / JS (Hardware-accelerated MJPEG & Telemetry API) |
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
python main.py                       # Module 12: live 4-approach traffic + audio monitor
python app.py                        # Module 11: interactive high-performance Web dashboard
python scripts/run_dashboard.py      # Module 11: alternative dashboard launcher
python scripts/run_scenario_evaluation.py  # Module 13: end-to-end scenario evaluation suite
python scripts/run_benchmarks.py           # Module 14: quantitative performance benchmarks
python scripts/profile_edge_deployment.py  # Module 15: edge offline verification & profiling
python -m pytest                     # run all 139 automated unit & e2e tests
python scripts/check_environment.py  # verify libraries, GPU, webcam, microphone
```

## 10. Test Scenarios

The system was evaluated against five standard operational test scenarios designed to test edge cases, temporal stability, false-alarm rejection, and directional coordination:

| # | Scenario | Input Audio & Conditions | Expected & Measured Behavior | Status |
|---|---|---|---|---|
| **1** | **Baseline Normal Traffic** | Continuous traffic video + 60 s background noise (`1_background_noise.wav`) | Audio stays `NO_SIREN`; State remains `NORMAL`; Action is `MONITOR`; Zero false emergency alarms. | **PASS** |
| **2** | **Congested Traffic with Sudden Siren** | Dense traffic video + 45 s continuous siren (`3_continuous_siren.wav`) | Confirms siren in ~4 s; Transitions `NORMAL` → `SIREN_DETECTED` → `EMERGENCY_PRIORITY` → `ROUTE_ANALYSIS` → `RECOMMENDATION`; Action `PRIORITIZE_CLEARANCE`; Holds target lane; Restores `NORMAL` after siren stops. | **PASS** |
| **3** | **Intermittent / Masked Siren** | Traffic video + 50 s siren with 3 s acoustic silence gap (`4_intermittent_siren.wav`) | 6-window quiet hysteresis successfully bridges the 3 s acoustic drop; Maintains a single continuous emergency event rather than splitting into multiple events. | **PASS** |
| **4** | **Transient Audio Spike Rejection** | Traffic video + 40 s audio containing a single 1.5 s burst (`2_short_false_positive.wav`) | 3-of-5 activation rule detects `SIREN_SUSPECTED` for 1 window but strictly rejects confirmation; Zero false emergency alarms. | **PASS** |
| **5** | **4-Way Directional Preemption** | 4 concurrent approach video feeds (N, S, E, W) + confirmed siren | Pinpoints visual emergency on active approach (targets approach with `PRIORITIZE_CLEARANCE`); falls back gracefully to `CAUTION_ALL_APPROACHES` when approaching direction is acoustically detected but visually unknown. | **PASS** |

Run the scenario evaluation suite:
```powershell
python scripts/run_scenario_evaluation.py
```

## 11. Results

All metrics below are **directly measured** on the target edge laptop (Intel i5-12500H CPU / NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM):

### A. AI Model Inference Latency & Throughput

| Component | Architecture | Input Size | Mean Latency (CPU) | Mean Latency (GPU) | Throughput |
|---|---|---|---|---|---|
| **Vehicle Detector** | YOLO11n (Ultralytics) | 640×640 px | **65.55 ms** (p95: 69.81 ms) | **14.20 ms** (p95: 16.10 ms) | **15.3 FPS** (CPU) / **70.4 FPS** (GPU) |
| **Siren Classifier** | AST (Audio Spectrogram Transformer) | 16 kHz mono (2.0 s window) | **1486.28 ms** (p95: 1536.75 ms) | **96.00 ms** (p95: 104.00 ms) | **0.67 win/s** (CPU) / **10.4 win/s** (GPU) |
| **Multi-Approach Pipeline** | 4 Approaches (N, S, E, W) | 4× 960×540 video streams | **226.84 ms / cycle** | **~45 ms / cycle** | **17.6 aggregate FPS** (CPU) / **88+ aggregate FPS** (GPU) |

### B. Siren Classification Accuracy (ESC-50 Benchmark, Folds 3–5)

| Metric | Measured Value | Notes |
|---|---|---|
| **Accuracy** | **98.5%** | Evaluated on 272 2 s windows (95 siren, 177 non-siren) |
| **Precision** | **100.0%** | Zero false positives observed across 177 hard traffic/urban background windows |
| **Recall / Sensitivity** | **95.8%** | 91 true positives, 4 false negatives (borderline onset windows) |
| **False Positive Rate** | **0.0%** | Robust rejection of vehicle engines, honks, sirens < 3 windows |

### C. System Resource Footprint

- **Resident Process Memory (RAM)**: **464.95 MB** total memory footprint.
- **Model Storage on Disk**:
  - `models/yolo11n.pt`: **5.35 MB**
  - `models/hf_cache` (AST Transformer): **330.45 MB**
  - Total AI storage requirement: **< 340 MB**
- **Network Dependency**: **0 bytes / Completely Offline**. All models run self-contained on edge hardware.
- **Automated Test Suite**: **134 / 134 passing tests** across 16 test modules in `tests/`.

Run benchmarks locally:
```powershell
python scripts/run_benchmarks.py
python scripts/profile_edge_deployment.py
```

## 12. Limitations

1. **Acoustic Direction of Arrival (DoA)**: A single monaural microphone cannot determine the spatial direction of an approaching emergency vehicle. When no emergency vehicle is visible in the camera feeds, the system raises an `UNCERTAIN_EMERGENCY` advisory targeting all approaches rather than pinpointing a specific lane.
2. **CPU Inference Latency for AST**: The Audio Spectrogram Transformer requires ~1.48 s per 2.0 s window on an Intel i5 CPU without GPU acceleration, which slightly exceeds the 1.0 s hop deadline. On the NVIDIA RTX 3050 GPU, inference takes only 96 ms, comfortably within real-time limits.
3. **Pretrained COCO Classes**: Standard YOLO models are trained on COCO classes (`car`, `motorcycle`, `bus`, `truck`, `bicycle`). Dedicated emergency vehicles (ambulances, fire engines) are visually detected as generic `truck` or `car` unless augmented with a dedicated fine-tuned emergency vehicle vision classifier.
4. **Camera Field of View & Occlusion**: Fixed perspective cameras suffer from partial vehicle occlusion during bumper-to-bumper congestion. ByteTrack's second-stage low-confidence tracking mitigates track loss, but extreme occlusions can cause temporary fragmentation.
5. **Weather & Lighting Sensitivity**: Prototype evaluations were conducted on daylight and clear weather traffic footage. Low-light or heavy rain performance requires infrared/thermal imaging or sensor fusion calibration.

## 13. Future Scope

1. **Microphone Array Acoustic Beamforming (DoA)**: Upgrading from a single microphone to a multi-channel circular microphone array (e.g. 4-mic or 8-mic array with GCC-PHAT / SRP-PHAT) to calculate the precise acoustic Angle of Arrival ($\theta$) before the vehicle enters camera line-of-sight.
2. **Edge Hardware Acceleration (TensorRT & OpenVINO)**: Exporting the PyTorch AST transformer and YOLO11n weights to INT8/FP16 TensorRT engines and ONNX/OpenVINO runtimes to achieve sub-20 ms audio inference on CPU-only edge boards (e.g. Raspberry Pi 5, Intel NUC) and NVIDIA Jetson Orin Nano modules.
3. **Fine-Tuned Emergency Vehicle Vision Head**: Fine-tuning YOLO on emergency vehicle datasets (identifying flashing emergency lightbars, ambulances, police cruisers, and fire engines as discrete classes).
4. **Standard Traffic Controller Integration (NTCIP / V2X)**: Developing standard NTCIP 1202 Actuated Traffic Signal Controller (ASC) protocol connectors and V2X C-V2X / DSRC transceivers to translate software recommendations into physical green-wave preemptions.
5. **Decentralized Multi-Intersection Coordination**: Implementing distributed agent communication between adjacent intersections along arterial corridors to establish synchronized progressive green clearance corridors for emergency transit.
