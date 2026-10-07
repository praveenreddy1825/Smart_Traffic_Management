# PROJECT NOTES — SmartCity Sentinel

Running record of decisions, assumptions and pending work. Updated after every module.

## Decisions Made

| # | Module | Decision | Reason |
|---|---|---|---|
| 1 | 0 | Project lives directly in `E:\smart_traffic_management` (no extra sub-folder) | User's choice |
| 2 | 0 | Python 3.12 virtual environment in `.venv` | PyTorch/Ultralytics support for 3.13 is less mature; 3.12 is the safest choice |
| 3 | 0 | PyTorch CUDA build (cu126) | Laptop has an NVIDIA RTX 3050 Laptop GPU (4 GB); speeds up YOLO and the audio model |
| 4 | 0 | Hugging Face cache redirected to `models/hf_cache` | C: drive had only ~3.5 GB free; also keeps the project self-contained |
| 5 | 0 | Single YAML config (`config/config.yaml`) loaded into a dict | Simple, human-editable, satisfies the centralized-config rule; no extra config framework needed |
| 6 | 0 | Python `logging` with console + rotating file (`logs/sentinel.log`) | Standard library, no extra dependency |
| 7 | 0 | No `.env` file | No secrets/API keys are needed; everything runs locally |
| 8 | 0 | Added `logs/` folder and `pytest.ini` to the suggested structure | Keeps log files out of the root; lets tests import `app` from any folder |
| 9 | 1 | One `VideoInput` class for both files and webcams, yielding `VideoFrame(image, index, timestamp_sec)` | Later modules stay independent of the source type |
| 10 | 1 | Frames are shrunk so the longer side is at most `video.max_frame_size` (960); smaller frames are never enlarged (changed from width-based after portrait clips gave 960x1707 frames taller than the screen) | Less work for YOLO; enlarging adds no detail |
| 11 | 1 | `realtime_playback: true` by default for recorded video | Video must run at real speed to stay in sync with live microphone audio |
| 12 | 1 | If processing is slower than real time, frames are NOT skipped (video plays slower) | Simpler and counting stays correct; revisit in Module 12 if needed |
| 13 | 1 | Webcam opened with DirectShow backend on Windows | Opens noticeably faster than the default backend |
| 14 | 2 | YOLO11n (Ultralytics, COCO-pretrained) as default detector | Smallest standard YOLO with car/motorcycle/bus/truck classes; mature and well documented; measured 14.2 ms/frame GPU, 33.3 ms CPU |
| 15 | 2 | Confidence threshold 0.35, IoU 0.5 (prototype values) | Common starting values; to be tuned on the user's real traffic clips |
| 16 | 2 | Person detection available but OFF by default | Prompt says not to make it central; traffic counts only use vehicles |
| 17 | 2 | Weights stored in `models/` (auto-downloaded) | Keeps project self-contained; not in git |
| 18 | 2 | Main demo clip = Pexels 13997539 (fixed camera, intersection); heavy clip = 16922729 (1080p copy of the 4K original) | Steady camera, good detections; 4K decoding alone took 15.7 ms/frame vs 3.4 ms for the 1080p copy |
| 19 | 2 | Clips 15252488 (pedestrian street, 0 vehicles) and 19452742 (moving top-down drone, 0 detections) and 14552312 (panning camera) not used | No vehicles / COCO models fail on bird's-eye view / moving camera breaks fixed lane regions |
| 20 | 3 | ByteTrack via Ultralytics (`model.track(persist=True)`), settings in `config/bytetrack.yaml` | Reliable in dense traffic, no extra neural network, no training; centroid tracker swaps IDs too easily, DeepSORT adds unnecessary complexity |
| 21 | 3 | Tracking uses detection confidence 0.1 (ByteTrack's low threshold); new IDs still need 0.25 | Lets ByteTrack's second stage keep partly hidden vehicles on the same ID |
| 22 | 3 | Count a track once after `min_frames_to_count` = 10 frames; class by majority vote | 48% of raw IDs lived < 10 frames (fragmentation); majority vote removes car/truck flicker |
| 23 | 3 | Line-crossing count with line at 75% frame height | Measured: crossings stable (33-34) across all min_frames settings while unique count varied 121-232; at 75% all crossings matched traffic direction |
| 24 | 3 | Text panel drawn on a semi-transparent dark box with auto font scaling | Old thick-outline text looked doubled; portrait frames are only 540 px wide |
| 25 | 4 | Traffic level from **density** = size-weighted vehicles in view, averaged over 3 s of video time | Congestion = how full the road is NOW (needed by the emergency logic); weighting reflects that a bus occupies more road than a motorcycle; averaging stops one missed frame flipping the level |
| 26 | 4 | Class weights car 1.0 / motorcycle 0.5 / bus 2.5 / truck 2.5 | PROTOTYPE values inspired by the PCU idea, not from an official standard |
| 27 | 4 | Thresholds medium = 8, high = 15 | Initially placeholders; checked against measured densities: intersection p10-p90 7.7-14.2 (never HIGH), Dhaka 16.2-27.6 and motorcycle clip 16.3-21.4 (always HIGH). Kept. Camera-specific |
| 28 | 4 | Flow rate (line crossings/min) reported but NOT used for the level | Informational; flow is low both on empty roads and in jams, so it is ambiguous for congestion |
| 29 | 5 | Lanes = hand-drawn polygons (0-1 fractions), one profile per video file name | Brief forbids complex lane detection; polygons follow perspective; per-file profiles let each camera keep its own layout |
| 30 | 5 | Vehicle -> lane by bounding-box **bottom-centre** (configurable to centre) | Wheels touch the road; a tall vehicle's box centre can float over the neighbouring lane in angled views (covered by a unit test) |
| 31 | 5 | Lane thresholds medium = 3, high = 6 (prototype), same class weights and 3 s smoothing as Module 4 | Placeholders checked against measured lane densities: intersection lanes p90 3.8-5.0 (never HIGH); Dhaka lanes 1-2 mostly HIGH, lane 3 MEDIUM |
| 32 | 5 | Added `scripts/define_lanes.py` (click to draw, `--preview` to check) | Makes lane configuration easy to modify, as the brief requires |
| 33 | 5 | Intersection regions are approach ZONES (left / centre / right of the near road), not painted lanes | The junction has turning markings, not simple parallel lanes; near road = large, reliably detected vehicles |
| 34 | — | **Single-camera system** (lanes = regions within one camera's view). Multi-camera junction (one camera + mic per approach, signal-phase comparison) moved to Future Scope; optional simulated 4-approach demo only if time allows after Module 12 | User decision; matches the brief, keeps scope realistic |
| 35 | 6 | Audio standardised to mono, 16 kHz, float32 -1..1 (SciPy polyphase resampling) | Usual input format of Hugging Face audio models; to be confirmed for the model chosen in Module 7 |
| 36 | 6 | Windows of 2.0 s every 1.0 s (50% overlap) - prototype values | A siren sweep lasts ~1-3 s; overlap gives a decision every second |
| 37 | 6 | Microphone via sounddevice callback + queue; tries 16 kHz first, falls back to device rate + resampling | Background capture never drops audio while the main loop is busy; works with any mic |
| 38 | 6 | Recorded files release each window when its END time is reached (real-time mode) | Behaves exactly like the microphone, so file-based tests reflect live latency |
| 39 | 6 | Synthetic test sounds (`scripts/make_test_audio.py`), clearly labelled | Tests plumbing and Module 8 smoothing scenarios; NOT valid for judging siren-model accuracy |
| 40 | 7 | Searched the HF Hub (API) and read real `id2label` lists before choosing | Brief requirement; NO dedicated maintained siren model exists ("siren" hits were LLMs) |
| 41 | 7 | Model: `MIT/ast-finetuned-audioset-10-10-0.4593` (AudioSet AST) | Labels really include Siren, Civil defense siren, Emergency vehicle, Police car/Ambulance/Fire engine (siren); official authors' model, BSD-3, most downloaded. ESC-50/UrbanSound8K models only have a generic "siren" |
| 42 | 7 | Sigmoid per label (not softmax / not the HF `pipeline`) | AudioSet is multi-label and config has no problem_type, so the default pipeline would apply softmax |
| 43 | 7 | Siren score = max probability over 6 siren labels; type = best of ambulance/police/fire | Simple and explainable; type shown only when a siren is detected |
| 44 | 7 | Threshold 0.15, chosen on ESC-50 folds 1-2, reported on folds 3-5 | Avoids tuning on test data; placeholder 0.30 measured worse recall (0.895 vs 0.958) |
| 45 | 7 | Keep 2 s windows (not 4 s) | 4 s only slightly better on ESC-50 (99.3% vs 98.5%, small test set) but doubles the delay to first decision; slow-wail alternation handled in Module 8 |
| 46 | 7 | Load model with `local_files_only` first, download only if missing | Edge-first: no internet needed after setup |
| 47 | 7 | Evaluation data: ESC-50 subset (40 siren + 80 hard/traffic negatives, 1 per class per fold) | Real recordings, public academic benchmark; AST was trained on AudioSet, not ESC-50, so it is an external test |
| 48 | 8 | Siren smoothing = small state machine NO_SIREN -> SIREN_SUSPECTED -> SIREN_CONFIRMED | Rule-based, explainable; feeds the Module 10 priority state machine |
| 49 | 8 | Activate on **3 of the last 5** windows (not "n in a row", not 3 of 4) | Unit test showed 3-of-4 NEVER activates on a strictly alternating slow wail (only 2 of any 4); 3-of-5 does, and a continuous siren still confirms after 3 windows |
| 50 | 8 | Release after **6** quiet windows in a row (hysteresis) | Measured: 4 split a real siren with a 3 s gap into two events (one model miss + gap); 6 bridges it. Ending late is the safe direction for emergency priority |
| 51 | 8 | Smoothed confidence = mean of the real model scores in the history | No invented confidence |
| 52 | 8 | Test scenarios built from REAL ESC-50 audio (folds 3-5 only) with exact ground truth (`scripts/make_smoothing_scenarios.py`) | Realistic sounds; folds 1-2 were used for the Module 7 threshold |
| 53 | 9 | Decision engine = stateless explainable rules over (TrafficReport, LaneReports, AudioEmergencyState) | Brief: rule-based, no extra ML; the state machine is Module 10 |
| 54 | 9 | Actions: target HIGH -> PRIORITIZE_CLEARANCE, MEDIUM -> PREPARE_CLEARANCE, LOW -> MAINTAIN_ROUTE; no siren -> MONITOR | Matches the brief's example recommendations; MEDIUM case added so moderate traffic is not treated as clear |
| 55 | 9 | Target lane = option C (designated lane if configured and present, else most congested; whole road if no lanes) | User decision; one microphone gives no direction |
| 56 | 9 | Every emergency decision carries `reasons` + notes "siren direction unknown" and "recommendation only - no signal control" | Explainability + brief rules on honesty and no physical control |
| 57 | 10 | Priority state machine NORMAL -> SIREN_DETECTED -> EMERGENCY_PRIORITY -> ROUTE_ANALYSIS -> RECOMMENDATION -> RETURN_TO_NORMAL -> NORMAL | States exactly as in the brief; driven by Module 8's confirmed/ended events |
| 58 | 10 | No extra "confidence threshold" in Module 10 | The brief's threshold + temporal confirmation are already applied by Modules 7 (0.15, measured) and 8 (3 of 5); a second unmeasured threshold would be arbitrary |
| 59 | 10 | Route analysis averages lane densities for 2 s before choosing the target | Snapshot choice flipped between near-equal lanes (Module 9); test shows the average picks the truly worse lane |
| 60 | 10 | Target lane held; switch only if another lane is >= 2.0 denser for >= 3 s; designated lane never switched; every switch logged with reason | Stable, explainable recommendation; measured: intersection held Lane 2 for all 19 s, Dhaka switched once (15.3 vs 6.9) |
| 61 | 10 | Event confidence = mean / peak of model scores of SIREN windows during the event (each window once) | Fixes the misleading low "smoothed confidence" (which averaged in quiet windows) |
| 62 | 10 | Module 9 `decide()` got optional `target_lane` / `target_rule` (backward compatible, Module 9 tests unchanged) | Lets the state machine hold its chosen lane |

### Emergency-lane decision (user choice "C", before Module 9)
A single microphone cannot tell which lane/direction the emergency vehicle is in. Therefore:
- if `decision.emergency_lane` is set in config -> that designated lane is analysed and recommended;
- otherwise -> the MOST CONGESTED lane (highest lane density) is targeted;
- every recommendation states which rule was used and that the siren direction is unknown.

## Model Choices

### Vehicle detection: YOLO11n
- Source: Ultralytics official weights (`yolo11n.pt`, ~5.4 MB), auto-downloaded to `models/`
- Pretrained on COCO (80 classes). Classes used: car (2), motorcycle (3), bus (5), truck (7); person (0) optional
- Documented by Ultralytics: ~2.6 M parameters, COCO mAP50-95 39.5
- Measured on this laptop (960-px frame, imgsz 640, 50 runs after warm-up):
  RTX 3050 GPU 14.2 ms/frame (~70 FPS), i5-12500H CPU 33.3 ms/frame (~30 FPS); ~60 MB GPU memory
- Measured detection: bus in Ultralytics `bus.jpg` sample photo, confidence 0.94
- Limitations: no ambulance / auto-rickshaw / emergency classes in COCO; small or occluded vehicles may be missed

### Siren detection: AST fine-tuned on AudioSet
- Repository: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 (BSD-3-Clause)
- Paper: Gong, Chung & Glass, "AST: Audio Spectrogram Transformer", Interspeech 2021
- 527 AudioSet labels; siren labels used (id): Emergency vehicle (322), Police car (siren) (323),
  Ambulance (siren) (324), Fire engine, fire truck (siren) (325), Siren (396), Civil defense siren (397)
- Preprocessing (from preprocessor_config.json): 16 kHz mono -> 128-bin log-mel filterbank,
  1024 frames (~10.24 s; shorter input padded), normalised with mean -4.268 / std 4.569
- Output: sigmoid probability per label; siren score = max over siren labels; threshold 0.15
- Weights ~346 MB (safetensors), stored in models/hf_cache
- Measured latency per 2 s window: GPU 96 ms mean; CPU 844 ms mean (subset of 90 windows)
- Measured on ESC-50 folds 3-5 (2 s windows, 272 windows: 95 siren, 177 other):
  accuracy 0.985, precision 1.000, recall 0.958, false-positive rate 0.000 (TP 91, FN 4, TN 177, FP 0)
- Live laptop mic, quiet room, 9 windows: 0 false siren detections (scores 0.00)
- Limitations: trained on 10 s clips (2 s windows are mostly padding); AudioSet sirens are mostly
  Western; slow "wail" sweeps can alternate siren / "sine wave" between 2 s windows (synthetic test);
  real street audio through the laptop mic not yet tested; small test set (24 siren clips)

ESC-50 attribution: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification",
ACM MM 2015, CC BY-NC 3.0 - used for non-commercial evaluation only.

## Assumptions

- The laptop is the edge device; no cloud services are used.
- Recorded traffic videos (fixed overhead/angled camera) will be provided by the user in `data/videos/`.
- Siren and non-siren audio clips will be collected for testing in `data/audio/`.

## Configuration

All settings are in `config/config.yaml`. Current sections:

- `project`: name and version
- `paths`: data, model, cache and log folders (relative to project root)
- `logging`: level, file logging, rotation
- `device`: `auto` / `cuda` / `cpu`
- `video`: source (file path or webcam index), max_frame_size, realtime_playback, loop_video,
  webcam_fallback_fps, max_consecutive_read_failures, window_name
- `detection`: model_name, confidence_threshold, iou_threshold, image_size, vehicle_classes, include_person
- `tracking`: tracker_config, detection_confidence, min_frames_to_count, forget_after_frames, counting_line
- `config/bytetrack.yaml`: ByteTrack thresholds (Ultralytics defaults)
- `traffic`: class_weights, density_window_seconds, level_thresholds (medium, high), flow_window_seconds
- `lanes`: assignment_point, density_window_seconds, level_thresholds, default_profile,
  profiles (per video file: list of {name, polygon})
- `audio`: source ("mic" or file), sample_rate, window_seconds, hop_seconds, microphone_device,
  realtime_playback, loop_audio, silence_threshold_dbfs
- `siren`: model_name, siren_labels, type_labels, detection_threshold
- `smoothing`: history_windows (5), windows_to_activate (3), quiet_windows_to_end (6)
- `decision`: emergency_lane (null = most congested lane)
- `priority`: route_analysis_seconds (2), target_switch_margin (2.0), target_switch_hold_seconds (3), return_to_normal_seconds (5)

### How to calibrate the traffic thresholds for a new camera
1. Record that camera during clearly free-flowing and clearly congested periods.
2. Run `python scripts/run_traffic_analysis.py --source <clip> --no-display --no-realtime` on each.
3. Set `medium` above the free-flow p90 density and `high` near the congested p10-p25 density.
4. Ideally confirm with a traffic engineer's or manual LOW/MEDIUM/HIGH labels on sample periods.

## Known Limitations

- Module 1: the built-in webcam does not report its FPS; the configured fallback (30) is assumed.
- Module 1: a corrupted frame in the middle of a video file is treated as end-of-video.
- Module 1: no real traffic video tested yet; tests use synthetic generated videos.
- Module 2: ambulances are detected as truck/car (no dedicated class); auto-rickshaws may be missed or mislabelled.
- Module 2: detection quality on real traffic footage not yet measured (no traffic video yet).
- Module 2: per-frame counts shown by the demo are NOT unique vehicle counts (Module 3).
- Module 3: ID fragmentation (vehicle loses its ID when hidden/far away and gets a new one) inflates the
  UNIQUE count; it is sensitive to `min_frames_to_count`. Line-crossing count is more robust.
- Module 3: counting accuracy NOT yet measured - needs a manual ground-truth count (Module 14).
  Suggested: manually count vehicles crossing the 75% line in clip 13997539 (tool measured 34).
- Module 3: `min_frames_to_count` is in frames, so it means less time on 60 FPS clips.
- Module 4: thresholds are camera-specific (field of view changes how many vehicles are visible);
  the three clips come from DIFFERENT cameras, so comparing them is only indicative.
- Module 4: flow per minute is noisy in the first seconds of a clip (few crossings over a short time).
- Module 4: density counts only COCO vehicle classes; auto-rickshaws/cycle-rickshaws (Dhaka clip) are
  under-represented, so real congestion there is higher than measured.
- Module 5: lane counts inherit detector misses (e.g. a white pickup and two motorcyclists in
  intersection Lane 1 were not detected in a checked frame) -> lane load can be underestimated.
- Module 5: regions are drawn by eye per camera; if the camera moves, regions must be redrawn.
- Module 5: a vehicle changing lanes is counted in each lane it spent >= 10 frames in.
- Module 6: a decision can only be made after the first full window (2 s) - built-in latency.
- Module 6: the default Windows mic is a laptop array mic with noise suppression; real street sirens
  through it are untested. Quiet-room noise floor measured at about -57 to -62 dBFS.
- Module 6: no real siren recordings of our own yet (synthetic + ESC-50 downloads only).
- Module 7: a single window can flip between siren / not siren on slow wail sirens -> Module 8 must
  smooth over several windows ("k of last n", not "n in a row").
- Module 7: needs ~0.85 s per window on CPU only; with YOLO also on CPU it would not keep up.
- Module 7: siren TYPE (ambulance/police/fire) scores are much lower than the generic "Siren" score
  and were not evaluated - treat the type as indicative only (it switched between Fire truck and
  Police within one scenario).
- Module 8: confirmation takes ~4 s after a siren starts (2 s window + 3 windows) and the emergency
  ends ~6-7 s after the siren stops (measured on real-audio scenarios).
- Module 8: gaps longer than ~5 s create a new event; scenarios are mixes of 5 s ESC-50 clips, not
  continuous real street recordings.
- Module 9: the snapshot decision can switch target lane when lane densities are close
  (measured: Lane 2 -> Lane 1 at 4.5 vs 4.4) - target stability is handled in Module 10.
- Module 9: smoothed confidence includes the quiet windows before the siren (0.17 at confirmation
  in a measured run) - looks low on a display; Module 10/11 should show siren-window scores.
- Module 9: video and audio are combined by time only; their real-time synchronisation in one
  application is Module 12.
- Module 10: the held lane's ACTION follows its live level, so it can alternate (e.g. PREPARE /
  MAINTAIN) when the lane density sits on a threshold (measured: 5 s of MAINTAIN in a 19 s event).
- Module 10: total time from siren start to a lane recommendation = ~4 s confirmation + ~1 s +
  2 s route analysis = ~7 s (measured: siren at 10 s -> recommendation at 17 s).
- Module 10: a siren re-confirmed during RETURN_TO_NORMAL is logged as a NEW event.

## Measured Results So Far

| Clip | Frames | FPS (tracking) | Unique count | Line crossings | Naive per-frame sum |
|---|---|---|---|---|---|
| 13997539 intersection | 1528 (50.9 s) | 29.4 | 159 (117 car, 32 motorcycle, 2 bus, 8 truck) | 34 (all down) | 18570 |
| 16922729 Dhaka 1080p | 443 (14.8 s) | 30.0 | 53 (40 car, 13 bus) | line off | 6115 |

Settings: YOLO11n, imgsz 640, GPU, min_frames_to_count 10. Counts are tool outputs, NOT verified against ground truth.

Traffic density (Module 4, measured, thresholds 8 / 15):

| Clip | Density p10 / p50 / p90 / max | Level share | Processing FPS (no realtime) |
|---|---|---|---|
| 13997539 intersection | 7.7 / 13.0 / 14.2 / 16.5 | LOW 13%, MEDIUM 86%, HIGH 0% | 48.7 |
| 16922729 Dhaka 1080p | 16.2 / 20.5 / 27.6 / 28.4 | HIGH 100% | 41.5 |
| 13275193 motorcycles 60 fps | 16.3 / 19.9 / 21.4 / 21.7 | HIGH 100% | 42.5 |

Lane density (Module 5, measured, lane thresholds 3 / 6):

| Clip / lane | Density p10 / p50 / p90 / max | Level share | Different vehicles passed |
|---|---|---|---|
| Intersection Lane 1 | 1.6 / 3.0 / 4.3 / 4.8 | LOW 50%, MEDIUM 50% | 58 |
| Intersection Lane 2 | 1.4 / 3.0 / 5.0 / 6.0 | LOW 51%, MEDIUM 49% | 57 |
| Intersection Lane 3 | 0.8 / 1.9 / 3.8 / 4.8 | LOW 81%, MEDIUM 19% | 52 |
| Dhaka Lane 1 | 4.2 / 8.7 / 15.7 / 16.7 | MEDIUM 27%, HIGH 73% | 35 |
| Dhaka Lane 2 | 6.8 / 8.0 / 10.5 / 10.8 | HIGH 100% | 21 |
| Dhaka Lane 3 | 1.6 / 3.9 / 4.7 / 4.9 | LOW 31%, MEDIUM 69% | 7 |

Siren smoothing (Module 8, measured on real-audio scenarios, 3-of-5 activation, release after 6):

| Scenario | Siren intervals | Raw siren windows | Events | False activations | Activation delay | Release delay |
|---|---|---|---|---|---|---|
| 1 Background noise (60 s) | none | 0/59 | 0 | 0 | - | - |
| 2 Short false positive (40 s) | 20-21.5 s | 1/39 | 0 | 0 | - | - |
| 3 Continuous siren (45 s) | 10-30 s | 19/44 | 1 | 0 | 4.0 s | 6.0 s |
| 4 Intermittent siren (50 s) | 10-20, 23-33 s | 17/49 | 1 | 0 | 4.0 s | 7.0 s |

With release after 4 quiet windows, scenario 4 split into 2 events (14-23 s, 26-38 s) - see decision 50.

Decision engine (Module 9, offline video + audio runs, measured):

| Video + audio | Result |
|---|---|
| Intersection + 1_background_noise | NORMAL / MONITOR for all 49 s |
| Dhaka + 3_continuous_siren (offset 8 s, siren from t = 2 s) | confirmed at t = 6 s -> PRIORITIZE_CLEARANCE (Lane 2, later Lane 1, both HIGH); 4 s MONITOR, 9 s PRIORITIZE |
| Intersection + 3_continuous_siren (siren from t = 10 s) | confirmed at t = 14 s -> PREPARE_CLEARANCE (Lane 2, MEDIUM); 22 s MONITOR, 22 s PREPARE |

Priority state machine (Module 10, offline video + audio, measured):

| Video + audio | Timeline | Event log |
|---|---|---|
| Intersection + 3_continuous_siren | SIREN_DETECTED 12 s, EMERGENCY_PRIORITY 14 s, ROUTE_ANALYSIS 15 s, RECOMMENDATION 17 s (Lane 2 held all 19 s), RETURN_TO_NORMAL 36 s, NORMAL 41 s | 22 s event, confidence mean 0.47 / peak 0.83, PREPARE 14 / MAINTAIN 5 updates |
| Dhaka + 3_continuous_siren (offset 8 s) | SIREN_DETECTED 4 s, EMERGENCY_PRIORITY 6 s, RECOMMENDATION 9 s (Lane 2), target switch 13 s Lane 2 -> Lane 1 (15.3 vs 6.9 for >= 3 s) | confidence mean 0.31 / peak 0.56, PRIORITIZE 6 updates (clip ended during the event) |

## Pending Work / Open Questions

- **Emergency-lane question (decide before Module 9):** a single microphone cannot tell which lane
  or direction the emergency vehicle is in. Options: (a) a fixed lane set in the config,
  (b) default to the most congested lane, with the limitation clearly stated.
- Obtain traffic videos (light and heavy traffic).
- Obtain siren and non-siren audio clips.
- Manually count vehicles in one clip for Module 14 counting-accuracy evaluation.
- Modules 1–15.
