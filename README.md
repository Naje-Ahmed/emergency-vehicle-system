# 🚨 AegisFlow: Emergency Vehicle Priority Routing & Adaptive Signal Preemption

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-red.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Operational-brightgreen.svg)]()

> **AegisFlow** is an end-to-end multi-modal emergency vehicle detection and adaptive traffic signal preemption system. Built with **YOLOv8**, **PyTorch**, **OpenCV**, and **Supervision**, it detects approaching emergency vehicles (Ambulance, Fire Truck, Police Cruisers), computes spatial approach vectors and urgency indexes, and opens dynamic **Green Corridors** at urban intersections.

---

## 🌟 Key Features

1. **Dual-Factor Emergency Detection**:
   - High-speed real-time inference with **YOLOv8** for vehicles: `Ambulance`, `Fire Truck`, `Police Car`, `Bus`, `Truck`, `Car`, `Motorcycle`.
   - Heuristic emergency strobe light & livery spectral verification to prevent look-alike false triggers.
   - Dual-factor acoustic siren frequency analysis (FFT spectral peaks in 650Hz–1650Hz wail/yelp band).

2. **Dynamic Priority Engine (Urgency Matrix)**:
   - Calculates real-time Urgency Index ($U$):
     $$U = w_{\text{type}} \cdot C_{\text{det}} \cdot \left(1 + 0.4 \cdot \frac{v_{\text{approach}}}{v_{\text{max}}} + 0.6 \cdot \frac{A_{\text{bbox}}}{A_{\text{frame}}}\right) + \gamma_{\text{siren}}$$
   - Automatically executes a **Green Wave Override** when $U \ge \tau_{\text{preempt}}$.

3. **Adaptive Signal State Machine**:
   - Multi-phase state machine (`NORMAL_CYCLE` $\rightarrow$ `PREEMPTION_PENDING` $\rightarrow$ `GREEN_WAVE_OVERRIDE` $\rightarrow$ `CLEARANCE_HOLD` $\rightarrow$ `RESTORE_CYCLE`).
   - Resolves multi-lane emergency vehicle conflicts smoothly.

4. **Incident Database & Telemetry Analytics**:
   - SQLite persistence (`data/events.db`) logging timestamps, preemption duration, vehicle speed, trigger snapshots, and clearance time.
   - Interactive Plotly telemetry charts and CSV exports.

5. **Integrated Synthetic CCTV Simulator**:
   - Built-in traffic stream generator rendering perspective CCTV feeds with approaching ambulances, flashing emergency beacons, and civilian traffic.

---

## 📁 Repository Structure

```
emergency_vehicle_detection_system/
├── app.py                      # Main Streamlit command center dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── samples/                # CCTV video clips (cctv_ambulance_approach.mp4)
│   ├── audio/                  # Acoustic siren test audio files (.wav)
│   ├── snapshots/              # Automated incident snapshot captures
│   └── events.db               # SQLite incident database
├── models/
│   ├── weights/                # YOLOv8 weights (yolov8n.pt, best.pt)
│   └── train.py                # Transfer learning & fine-tuning script
└── src/
    ├── __init__.py
    ├── detector.py             # YOLOv8 detection engine & heuristic ROI analysis
    ├── tracker.py              # ByteTrack / vector velocity & proximity tracker
    ├── signal_controller.py    # Urgency scoring & adaptive signal state machine
    ├── audio_analyzer.py       # Dual-factor siren acoustic FFT analyzer
    ├── database.py             # SQLite persistence & telemetry statistics
    ├── stream_generator.py     # High-fidelity synthetic CCTV stream generator
    └── create_samples.py       # Sample asset generation utility
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository and navigate to workspace
cd "emergency vehicle detection system"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch AegisFlow Command Center

```bash
# Run Streamlit dashboard
streamlit run app.py
```

The application will be accessible at: `http://localhost:8501`

---

## 🧪 Simulation Modes

- **🚨 Synthetic CCTV Simulator**: Real-time 25 FPS stream with approaching emergency vehicles, flashing strobe lights, and civilian traffic.
- **📁 Custom Video Upload**: Test on recorded real-world traffic surveillance footage (`.mp4`, `.avi`, `.mov`).
- **📹 Built-in CCTV Sample**: Pre-recorded emergency approach test sequence.
- **📷 Live Webcam**: Connect any local or USB CCTV camera feed for real-time edge testing.

---

## 🔬 Research & Experimental Benchmarks

Fine-tuned on urban vehicle datasets (Ambulance, Bus, Car, Motorcycle, Truck), AegisFlow achieves:
- **Ambulance Class mAP@0.5**: `0.780` (Precision: `0.755`, Recall: `0.688`)
- **Overall Model mAP@0.5**: `0.496` (+105% improvement over pre-trained COCO baseline)
- **Optimal Operational Threshold**: $F_1$-optimal at confidence cutoff `0.323`

---

## 📜 Authors & Academic Citation
*Developed for Advanced Intelligent Transportation Systems (ITS) & Smart Urban Mobility Research.*
