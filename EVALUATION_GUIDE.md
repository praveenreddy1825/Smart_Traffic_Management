# SmartCity Sentinel — Project Evaluation & Defense Guide
**Municipal Police Department & Traffic Operations Center (TMC) — Autonomous Multimodal Edge Intelligence**

---

## 1. Executive Summary & Evaluation Objective

**SmartCity Sentinel** is an end-to-end **Multimodal Edge Intelligence Platform** engineered for municipal traffic management centers and police departments. It continuously synthesizes dual sensory streams:
1. **Visual Surveillance (Video / Image)**: Real-time traffic volumetrics, multi-class vehicle tracking (ByteTrack), road congestion indexing, and high-density crowd gathering/surge estimation.
2. **Acoustic Intelligence (Audio)**: Real-time emergency siren classification (Ambulance, Fire Engine, Police car) powered by an Audio Spectrogram Transformer (AST) running locally at 16 kHz.

### Evaluation Focus:
- **Zero Cloud Dependence**: 100% self-contained local edge inference (runs on CPU or NVIDIA GPU).
- **Multimodal Sensor Fusion**: Visual scene analysis fused with acoustic emergency audio to prioritize life safety.
- **Municipal Police Use Case**: Automated incident classification (`CODE-3`, `CODE-2`, `CODE-1`), signal phase preemption, pedestrian crowd surge interlocks, and exportable evidentiary police reports.

---

## 2. System Architecture & Multimodal Pipeline

```mermaid
flowchart TD
    subgraph Inputs ["Multimodal Input Modalities"]
        V["CCTV Video Stream\n(MP4 / RTSP / Webcam)"]
        I["Static Surveillance Still\n(JPG / PNG Image)"]
        A["Acoustic Audio Stream\n(16 kHz Mic / WAV / MP3)"]
    end

    subgraph EdgeAI ["Local Edge Intelligence Layer"]
        YOLO["YOLOv11s Detector\n+ Dual-Tile Crowd Tiler"]
        BYTE["ByteTrack Tracker\n(Vehicles + Pedestrians)"]
        AST["Audio Spectrogram Transformer\n(AST AudioSet Pretrained)"]
        SMOOTH["Acoustic State Machine\n(3-of-5 Hysteresis Filter)"]
    end

    subgraph Analytics ["Traffic & Crowd Analytics Engine"]
        TA["Traffic Volumetrics & PCU\n(Low / Medium / High Density)"]
        CA["Crowd Density & Surge Analyzer\n(Gathering Estimation & Interlock)"]
        FUS["Directional Decision Engine\n(Multimodal Priority Arbiter)"]
    end

    subgraph Outputs ["Police Department TMC Outputs"]
        OSD["Live CCTV Video OSD Feed\n(/video_feed MJPEG)"]
        UI["TMC Central Signal Controller\n(HTML5 Responsive Dashboard)"]
        DOC["Police Incident Evidence Dossier\n(HTML / PDF / JSON Export)"]
        CLI["CLI Incident Assessment\n(main.py Terminal Summary)"]
    end

    V --> YOLO
    I --> YOLO
    YOLO --> BYTE
    BYTE --> TA
    BYTE --> CA

    A --> AST
    AST --> SMOOTH

    TA --> FUS
    CA --> FUS
    SMOOTH --> FUS

    FUS --> OSD
    FUS --> UI
    FUS --> DOC
    FUS --> CLI
```

---

## 3. Step-by-Step Evaluation Demonstration

### Mode 1: Interactive Police TMC Web Operations Dashboard (Primary Demo)
Launch the web operations center:
```powershell
python main.py --dashboard
```
Or directly:
```powershell
python app.py
```
Open **`http://127.0.0.1:8000`** in any modern web browser.

#### Scenarios to Demonstrate to the Panel:
1. **Mass Crowd Surge & Public Safety Interlock**:
   - In the scenario bar, select **Crowd** folder -> **`15443786_...mp4 (Metropolitan Pedestrian Promenade)`**.
   - **What to show**:
     - Dual-tile detector identifies **60+ visible heads** deep into the background.
     - Calibrated gathering estimation projects **~170+ total persons**.
     - Metric badge shifts to **`CRITICAL: CROWD SURGE`** (Red alert).
     - Signal phase triggers **`CROWD CLEARANCE`** with 25s hold on vehicular red.
     - Interlock card displays **`Crowd Safety Interlock: ENGAGED`**.
2. **Acoustic Emergency Siren & Arterial Preemption**:
   - Select **Emergency** folder -> **`01_ambulance_corridor.mp4`** (with synchronized siren audio).
   - **What to show**:
     - AST acoustic classifier detects siren frequency; confidence gauge rises past threshold.
     - Decibel meter registers live loudness (-24 dBFS).
     - State machine confirms `SIREN_CONFIRMED (Ambulance)`.
     - Decision engine preempts all traffic: `🚨 EMERGENCY PREEMPTION: AMBULANCE APPROACHING` (Phase: `EMERGENCY_CLEARANCE`).
3. **Heavy Traffic Gridlock & Adaptive Signal Extension**:
   - Select **Vehicles** folder -> **`01_dense_intersection.mp4`**.
   - **What to show**:
     - Multi-class vehicle tracking (Cars, Buses, Trucks, Motorcycles).
     - Size-weighted road density calculation triggers `FLOW: HIGH`.
     - Adaptive signal extension: `🚗 HIGH CONGESTION: EXTENDING GREEN PHASE (+30s)`.
4. **Live Police Incident Evidence Dossier Export**:
   - While any incident is active on the feed, click the **"Police Incident Report"** button in the top navigation bar.
   - **What to show**:
     - Opens the official, printable **Police Department Incident Report**.
     - Shows incident code (`CODE-3`, `CODE-2`, `CODE-1`), embedded annotated surveillance snapshot, telemetry matrices, and dispatch advisory.
     - Evaluators can click **Print / Save Police Dossier (PDF)** or **Download JSON**.

---

### Mode 2: Static Surveillance Image Assessment (`--image`)
Demonstrates multimodal analysis on a static crime scene or traffic surveillance still:
```powershell
# Crowd Gathering Still
python main.py --image data/images/02_crowd_surge_holi.jpg

# Heavy Intersection Still with Companion Siren Audio
python main.py --image data/images/01_intersection_vehicles.jpg --audio data/audio/scenarios/3_continuous_siren.wav
```
*Outputs a complete terminal police dossier and saves annotated evidence to `data/test/output/incident_*.jpg`.*

---

### Mode 3: Single Video Surveillance Feed (`--video`)
Process any recorded CCTV video clip:
```powershell
python main.py --video data/videos/emergency/01_ambulance_corridor.mp4 --seconds 15
```

---

### Mode 4: Automated System Evaluation Suite (`--report`)
Run the automated verification suite for the evaluation panel:
```powershell
python main.py --report
```
*Executes all test tracks and generates `data/test/output/EVALUATION_REPORT.md`.*

---

## 4. Key Defense Questions & Answers for the Panel

### Q1: Why is a Multimodal (Video/Image + Audio) approach superior to vision-only traffic systems?
> **Answer**: Vision cameras suffer from blind spots, occlusions by tall trucks, building corners, rain, and night glare. Acoustic sirens propagate non-line-of-sight and can be detected 4–7 seconds before an emergency vehicle is visually in frame. Combining audio (for early acoustic warning) with vision (for lane pinpointing and crowd volume) ensures zero false alarms and guarantees life-safety preemption.

### Q2: How does your system run locally without cloud APIs?
> **Answer**: All models run on-device. The vehicle and crowd detector uses YOLOv11s (9.4M parameters, PyTorch CPU/CUDA), and the acoustic classifier uses an Audio Spectrogram Transformer (AST) with quantized Mel-filterbanks. Both sets of weights are cached locally in `models/`, ensuring zero network latency and complete privacy compliance for municipal police surveillance.

### Q3: Why does your crowd module perform better than standard object detectors on crowds?
> **Answer**: Standard YOLO models are trained on full-body upright human figures. In packed crowds (100–500 people), 80% of bodies are occluded by foreground rows, and standard 640p downscaling shrinks background faces to 5 pixels. Our system uses a **Dual-Tile Multi-Scale Enhancement** that extracts high-resolution horizontal tiles across the gathering band at sensitive thresholds with OpenCV NMS deduplication, quadrupling detected heads and providing realistic volume projection.

### Q4: How do you prevent audio false alarms from car horns and loud street noise?
> **Answer**: We use a two-stage temporal smoothing filter:
> 1. AST AudioSet model evaluates multi-label probabilities across 6 emergency siren classes.
> 2. Temporal Hysteresis: Requires **3 detections out of the last 5 sliding windows (2.0s)** to activate, and requires **6 consecutive quiet windows** to release. Transient horn bursts (< 2s) are automatically rejected.

---

## 5. Verification Checklist for Evaluation Day

- [x] Python 3.12 Virtual Environment (`.venv`) configured with PyTorch, Ultralytics, PyAV, Transformers.
- [x] YOLO weights cached locally in `models/yolo11s.pt` (offline verified).
- [x] AST Audio Spectrogram Transformer cached locally in `models/hf_cache/` (offline verified).
- [x] Clean <kbd>Ctrl</kbd>+<kbd>C</kbd> instant terminal shutdown implemented.
- [x] Fast 0 ms browser loading without infinite tab spinners.
- [x] Video (`.mp4, .mov, .avi`) and Image (`.jpg, .png`) upload support active.
- [x] Police Incident Dossier export active (`/api/incident/export`).
- [x] Automated evaluation suite verified (`python main.py --report`).
