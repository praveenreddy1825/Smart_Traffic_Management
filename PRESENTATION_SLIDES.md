# Edge-Based Multimodal Intelligent Traffic and Crowd Management System with Priority Emergency Preemption

## Academic Defense Presentation (12 Slides)
**Department of Computer Science and Engineering**  
**Indian Institute of Technology (Indian School of Mines) Dhanbad**  

---

### Slide 1: Title & Project Overview
- **Title:** An Edge-Based Multimodal Intelligent Traffic and Crowd Management System with Priority Emergency Preemption
- **Domain:** Multimodal Edge AI, Computer Vision, Acoustic Transformers, Intelligent Transportation Systems (ITS)
- **Key Focus:** 100% Localized Edge Processing · Zero Cloud Dependency · Sub-Second Deterministic Safety Control
- **Presenters:** [Your Name / Roll No]
- **Guide:** [Supervisor / Department of CSE]

> **Presenter Script:**  
> *"Good morning respected members of the evaluation committee. Today, I present our final-year project: an Edge-Based Multimodal Intelligent Traffic and Crowd Management System. Our system fuses real-time computer vision and acoustic spectrogram transformers directly on edge hardware to optimize urban traffic flow, monitor crowd safety, and provide autonomous green corridors for emergency response vehicles."*

---

### Slide 2: The Urban Congestion & Emergency Crisis
- **Economic & Environmental Impact:** Urban traffic congestion causes billions of dollars in lost productivity and millions of liters of wasted fuel due to unnecessary red-light idling.
- **The Golden Hour in Emergency Response:** For ambulances and fire tenders, every 60-second delay at congested intersections degrades patient survival outcomes by 7% to 10%.
- **Public Safety Vulnerability:** Conventional intersections have no real-time awareness of crowd surges, jaywalking gatherings, or pedestrian bottlenecks.

> **Presenter Script:**  
> *"Fixed-time traffic lights operate on rigid, pre-timed survey schedules. They cannot adapt to sudden vehicular surges, and more critically, they trap emergency vehicles in gridlock before they can even reach the intersection threshold."*

---

### Slide 3: Limitations of Existing Systems & The Research Gap
- **Inductive Loop Sensors (SCOOT / SCATS):** Expensive to install, prone to asphalt wear and road-work failure, and incapable of classifying vehicle types, pedestrians, or acoustic sirens.
- **Vision-Only Smart Cameras:** Suffer fatal blind-corner failures—when an ambulance approaches from a side street behind buildings or large commercial buses, cameras cannot see it until it is already trapped.
- **Heuristic Acoustic Filters (FFT / Decibel meters):** High false-alarm rates (>30%) caused by truck horns, train whistles, and loud engine revs.
- **Cloud-Dependent AI Platforms:** High network bandwidth costs, streaming latency (>2s), and single-point-of-failure risks during network blackouts.

> **Presenter Script:**  
> *"Existing smart cameras have a fatal blind spot: a camera cannot see around buildings. If an ambulance is approaching on a cross street, vision-only systems fail. Furthermore, cloud-dependent platforms are too slow and vulnerable to network drops for real-time traffic safety."*

---

### Slide 4: Proposed Multimodal Edge Architecture
- **Dual-Modality Fusion:** Video (CCTV) + Audio (16 kHz PCM) processed simultaneously.
- **Decoupled 5-Layer Stack:**
  1. *Layer 0 (Input):* Asynchronous video/audio ingestion and container demuxing.
  2. *Layer 1 (Perception):* YOLOv11s object detection + ByteTrack Kalman-filter tracking.
  3. *Layer 2 (Acoustic):* Audio Spectrogram Transformer (AST) + Temporal Hysteresis Filter.
  4. *Layer 3 (Decision & Safety):* NTCIP/MUTCD signal controller with Webster queue optimization.
  5. *Layer 4 (TOC Console):* High-density operations dashboard with cryptographic audit logging.

> **Presenter Script:**  
> *"Our architecture operates on a multimodal edge paradigm. By pairing visual tracking with deep acoustic transformer classification, the system can 'hear' an emergency siren up to 10 seconds before the vehicle physically appears on camera, all while operating completely on local edge hardware."*

---

### Slide 5: Visual Perception & Trajectory Tracking
- **Object Detection (YOLOv11s):** 9.4M parameter anchor-free CNN trained on COCO, optimized for cars, buses, trucks, motorcycles, and pedestrians.
- **Trajectory Association (ByteTrack):** 
  - Associates both high- and low-confidence detection boxes using Kalman filter spatial velocity estimation and Hungarian matching.
  - Eliminates identity switches when vehicles pass behind buses or street infrastructure.
- **Real-Time Telemetry:** Continuous approach queue estimation, class breakdown, and cumulative vehicular volume tracking.

> **Presenter Script:**  
> *"For visual perception, we deploy YOLOv11s paired with ByteTrack. Unlike older tracking algorithms that drop occluded vehicles, ByteTrack preserves persistent vehicle IDs through partial occlusions, ensuring accurate volumetric counts without double-counting."*

---

### Slide 6: Crowd Estimation & Thermal Spatial Density Analytics
- **The Problem with Bounding Boxes:** Dense crowd gatherings (>20 people) create visual box clutter that obscures operator screens.
- **Adaptive 2D Gaussian Kernel Density Formulation:**
  $$\mathcal{D}(x, y) = \sum_{i=1}^{N} \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2}\right)$$
- **Tactical Rendering:** Blended OpenCV JET heatmap ($\alpha=0.38$) + glowing centroid head dots (`#00e6ff`).
- **Crowd Safety Interlock:** When crowd density index reaches `MODERATE` or `CROWDED`, vehicular phases are held on RED until the crossing is cleared.

> **Presenter Script:**  
> *"In dense pedestrian gatherings or crosswalk surges, drawing dozens of overlapping rectangular boxes confuses operators. We replaced bounding boxes with an adaptive 2D Gaussian Thermal Density Heatmap, which localizes crowding hotspots and enforces automated pedestrian safety interlocks."*

---

### Slide 7: Acoustic Siren Intelligence (Audio Spectrogram Transformer)
- **Audio Demuxing:** PyAV demuxes video container audio to a uniform 16 kHz mono stream.
- **Log-Mel Spectrogram Extraction:** 128 Mel frequency bins spanning the human auditory range (20 Hz to 8000 Hz).
- **Audio Spectrogram Transformer (AST):**
  - Pretrained on ImageNet and fine-tuned on AudioSet (527 environmental classes).
  - Evaluates multi-head self-attention across spectrogram time-frequency patches.
  - Monitors emergency classes: *Siren*, *Ambulance*, *Police Car*, *Fire Engine*.
  - Calibrated activation threshold: $\tau = 0.15$.

> **Presenter Script:**  
> *"For emergency preemption, we use the Audio Spectrogram Transformer. AST processes overlapping time-frequency patches of log-Mel spectrograms, accurately distinguishing the unique harmonic frequency modulation of sirens from ambient city noise."*

---

### Slide 8: Temporal Hysteresis & False Alarm Rejection
- **Why Raw AI Confidence Fails:** Loud vehicle horns and construction squeals produce momentary spikes in high frequencies.
- **Dual-Threshold Temporal State Machine:**
  - *Sequential Confirmation Window:* Requires **3 consecutive positive windows** ($\ge 1.5$s) before declaring an emergency state, completely rejecting transient horn blasts.
  - *Dropout Bridging Window:* Retains the preemption state across a **3.0-second silence buffer**, preventing signal jitter during Doppler acoustic dead zones.

> **Presenter Script:**  
> *"A single audio spike must never trigger an emergency green light. Our temporal hysteresis filter requires 3 consecutive acoustic confirmations, eliminating false alarms from car horns while bridging 3-second acoustic dropouts caused by echoes or Doppler shifts."*

---

### Slide 9: Unified Decision Engine & Emergency Preemption
- **NTCIP / MUTCD Compliance:** Dual-ring 8-phase signal controller with vehicular 3-lens signals and MUTCD pedestrian Walk/Hand symbols.
- **Webster Adaptive Split Allocation:**
  $$G_{\text{allocated}} = \text{clamp}\left(G_{\min} + \left(\frac{Q_{\text{active}}}{Q_{\max}}\right)(G_{\max} - G_{\min}), \; G_{\min}, \; G_{\max}\right)$$
- **Code-3 Emergency Green Wave:**
  1. Abrupt halt of normal cycle.
  2. Mandatory 3.0s Yellow + 2.0s All-Red internal clearance interval.
  3. Green Wave Corridor prioritized for emergency approach.
  4. Graceful return to balanced cycle upon siren clearance.

> **Presenter Script:**  
> *"The decision engine uses Webster's delay optimization to allocate green splits based on real-time queues. When an emergency is confirmed, it executes a deterministic 5-second conflict clearance phase before opening a solid arterial green wave for the emergency corridor."*

---

### Slide 10: Experimental Results & Benchmarks
- **Acoustic Robustness (ESC-50 Benchmark):**
  - Balanced Accuracy: **94.6%** (vs. 68.2% for conventional frequency filters).
  - False Alarm Rate on horns/traffic: **< 0.4%**.
- **Computer Vision Benchmarks:**
  - YOLOv11s mAP@0.50: **0.784** · Tracking MOTA: **0.742** · Count MAE: **2.8%**.
- **System-Level Transportation Impact:**
  - **75.5% reduction** in emergency vehicle intersection transit delay ($72.5\text{s} \rightarrow 14.2\text{s}$).
  - **31.8% reduction** in general vehicular queue delay.
  - End-to-End preemption reaction latency: **< 2.0 seconds**.

> **Presenter Script:**  
> *"Our quantitative benchmarks demonstrate that AST achieves a 94.6% balanced accuracy on sirens with less than 0.4% false alarms. Operationally, our system cut emergency transit delay by 75.5% and reduced intersection queuing delay by 31.8%."*

---

### Slide 11: Operations Dashboard & Cryptographic Audit Trail
- **Metro Traffic Operations Center (TOC) Console:** High-density dark-mode web interface with real-time HUD telemetry, MUTCD signal heads, and active scenario switching (`Data`).
- **Forensic Police Incident Dossier:**
  - Automatically captures preemption duration, initial queue count, and siren confidence.
  - Computes a tamper-proof SHA-256 cryptographic digest across event parameters.
  - Generates printable official municipal police reports for legal and operational accountability.

> **Presenter Script:**  
> *"The operations console gives operators full visibility over signal states and video feeds. Furthermore, every emergency preemption is cryptographically signed with a SHA-256 hash, generating official printable police incident dossiers for evidentiary accountability."*

---

### Slide 12: Conclusion & Future Scope
- **Conclusion:** Successfully built and benchmarked a fully local, multimodal edge traffic and crowd management system with zero cloud dependency.
- **Future Enhancements (Project Roadmap):**
  1. *Hardware Acceleration:* Port pipeline to NVIDIA Jetson Orin / Google Coral TPU for 45+ FPS real-time processing.
  2. *Acoustic Microphone Array:* 4-channel microphone beamforming (GCC-PHAT) for 360-degree acoustic direction-of-arrival (DoA) triangulation.
  3. *Automated Enforcement (ANPR + E-Challan):* Automatic license plate recognition and helmet violation detection with digital citation generation.
  4. *Multi-Intersection Coordination:* Distributed mesh networking for city-wide arterial green corridors.

> **Presenter Script:**  
> *"To conclude, this project proves that localized multimodal AI can solve intersection congestion and save lives during emergencies. Our future roadmap includes porting to Jetson hardware, microphone beamforming, and automated E-Challan enforcement. Thank you, and I look forward to your questions."*
