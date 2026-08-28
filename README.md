# 🛡️ Smart Vision

> **Making boring CCTV cameras smart.**

Smart Vision is an AI-powered home safety platform designed to transform conventional CCTV cameras into intelligent systems capable of understanding what is happening inside a home instead of simply recording video.

The long-term goal is to build a modular, scalable, and production-ready computer vision system that can detect safety-related situations, understand activities over time, and notify users only when an event actually matters.

---

## Overview

Traditional CCTV systems primarily act as passive observers. They continuously capture and record video, but they generally do not understand what is happening in the scene.

Smart Vision aims to change that.

Instead of simply asking:

> **What objects are visible?**

Smart Vision aims to answer:

> **What is happening, and does it require attention?**

The system is being developed incrementally so that each new capability becomes part of the same software product rather than a collection of unrelated experiments.

---

## Vision

Smart Vision is intended to evolve from a basic computer vision system into an intelligent home safety assistant.

The long-term system will combine visual detection, tracking, temporal reasoning, event processing, and notifications to identify potentially important situations inside a home.

The focus is not only on detecting objects, but on understanding their context and behavior.

For example:

```text
Person detected
      ↓
Person enters restricted area
      ↓
Activity continues over time
      ↓
Potential security event
      ↓
Notify user
```

The same architecture can eventually support safety scenarios such as fires, smoke, falls, unusual activity, and unknown people.

---

## Current Release: v0.1.0

Smart Vision v0.1.0 is the first working vertical slice of the project.

The current release captures live video from a webcam, processes each frame using a fire and smoke detection model, and displays detected regions with bounding boxes and confidence scores.

### Current Pipeline

```text
Webcam
   │
   ▼
WebcamCamera
   │
   ▼
Frame
   │
   ▼
YOLODetector
   │
   ▼
Fire / Smoke Detection
   │
   ▼
Detection Objects
   │
   ▼
Bounding Boxes + Confidence
   │
   ▼
Live Visualization
```

### Current Capabilities

* Live webcam video capture
* Camera abstraction
* Camera lifecycle management
* Camera connection status
* Frame representation with timestamps and camera identifiers
* Fire detection
* Smoke detection
* Bounding-box visualization
* Confidence score visualization
* Configurable YOLO model path
* Configurable webcam device index
* Configurable detection confidence threshold
* Structured application logging
* Unit tests for detector behavior
* Camera integration tests
* Fake camera/model components for controlled testing

---

## Current Fire and Smoke Detection

The current baseline uses a pretrained YOLO11s fire and smoke detection model.

```text
Model:
firedetect-11s.pt

Architecture:
YOLO11s

Classes:
0 → Fire
1 → Smoke
```

The model is stored locally at:

```text
models/firedetect-11s.pt
```

Model weights are intentionally excluded from the Git repository.

The current implementation treats this model as a baseline. Future versions may experiment with temporal reasoning, anomaly detection, model optimization, and alternative architectures.

---

## How the Current System Works

At the current stage, the system follows a simple real-time pipeline.

### 1. Camera Input

`WebcamCamera` manages the connection to the webcam and provides frames to the rest of the application.

```text
Webcam
   ↓
WebcamCamera
   ↓
Frame
```

### 2. Frame Representation

Each captured image is represented by a Smart Vision `Frame` containing:

```text
image
timestamp
camera_id
```

This gives the rest of the system a consistent representation independent of the underlying camera implementation.

### 3. Detection

`YOLODetector` receives a `Frame`, performs inference using the configured YOLO model, and converts the model output into Smart Vision `Detection` objects.

```text
Frame
   ↓
YOLODetector
   ↓
Detection[]
```

Each detection contains:

```text
class_id
class_name
confidence
bounding_box
```

### 4. Visualization

The current live demo draws bounding boxes and confidence scores on the captured frame and displays the result in an OpenCV window.

Visualization is currently kept in the demo layer so that the camera and detector modules remain independent of presentation logic.

---

## Project Architecture

Smart Vision is designed around modular components so that individual parts can evolve without requiring major changes to unrelated modules.

The long-term architecture is:

```text
Camera Input
     │
     ▼
Detection Engine
     │
     ▼
Tracking Engine
     │
     ▼
Home Intelligence
     │
     ▼
Event Engine
     │
     ├──────────────┐
     ▼              ▼
Database          Alerts
     │
     ▼
Dashboard
```

Additional components such as anomaly detection, activity understanding, and multi-camera management will eventually integrate into this pipeline.

### Why modularity matters

A detection model should not control the camera.

A camera should not know how alerts are sent.

The dashboard should not need to know which AI model produced a detection.

Each module should have a clear responsibility and communicate with other modules through well-defined interfaces.

This allows future implementations to replace individual components without redesigning the entire system.

For example:

```text
YOLODetector
      ↓
RTDETRDetector
      ↓
CustomDetector
```

The camera and other downstream components should not need to change simply because the detection model changes.

---

## Technology Stack

### Programming Language

**Python**

Python is used as the primary language because it provides a strong ecosystem for:

* Computer vision
* Machine learning
* Deep learning
* Backend development
* Automation
* Rapid experimentation

### Computer Vision

**OpenCV**

Used for:

* Camera input
* Image handling
* Video processing
* Visualization

**PyTorch**

Used as the underlying deep learning framework for model inference and future custom machine learning development.

**Ultralytics YOLO**

Used for the current fire and smoke detection baseline and future experimentation with object detection models.

### Backend

**FastAPI**

Planned for the backend API that will eventually expose camera state, detections, events, alerts, and analytics to the frontend and other clients.

### Database

**PostgreSQL**

Planned as the primary database for persistent application data such as:

* Cameras
* Users
* Events
* Detections
* Alerts
* System configuration

**SQLAlchemy**

Planned as the database abstraction and ORM layer.

### Frontend

**React**

Planned for the Smart Vision dashboard.

The dashboard will eventually provide:

* Live camera views
* Detection information
* Event history
* Analytics
* Camera status
* Alert history

### Deployment

**Docker**

Planned for consistent development and deployment environments.

**Raspberry Pi**

Planned as an edge deployment target for lightweight Smart Vision configurations.

### Version Control

**Git**

Used for source-code version control and release management.

**GitHub**

Used for repository hosting and collaboration.

---

## Project Structure

```text
Smart_Vision/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── requirements.txt
│
├── app/
│   ├── main.py
│   ├── __init__.py
│   │
│   ├── alerts/
│   ├── api/
│   ├── camera/
│   ├── core/
│   ├── database/
│   ├── detection/
│   ├── events/
│   ├── intelligence/
│   ├── schemas/
│   ├── services/
│   ├── tracking/
│   └── utils/
│
├── config/
├── data/
├── docker/
├── docs/
├── frontend/
├── logs/
├── models/
├── requirements/
├── scripts/
└── tests/
```

### `app/`

Contains the main Smart Vision application code.

### `app/camera/`

Contains camera abstractions and concrete camera implementations.

The current implementation includes:

```text
base.py
frame.py
status.py
webcam.py
```

The camera package is responsible for providing frames and managing camera state without exposing implementation details to other modules.

### `app/core/`

Contains application-wide infrastructure such as logging and other core functionality.

### `app/detection/`

Contains detection abstractions and implementations.

The current implementation includes:

```text
detection.py
detector.py
yolo.py
```

This layer converts model-specific results into Smart Vision detection objects.

### `app/tracking/`

Reserved for multi-object tracking and persistent identities across frames.

### `app/intelligence/`

Reserved for higher-level scene and activity understanding.

### `app/events/`

Reserved for converting raw detections and temporal information into meaningful events.

### `app/database/`

Reserved for persistence and database access.

### `app/api/`

Reserved for the FastAPI backend.

### `app/alerts/`

Reserved for notification and alert delivery.

### `app/services/`

Reserved for application-level services and orchestration.

### `app/schemas/`

Reserved for structured data schemas used between application boundaries, APIs, and persistence layers.

### `config/`

Reserved for application configuration as the project grows.

### `data/`

Contains development data and other non-source assets.

### `models/`

Contains locally stored model weights.

Large model files are excluded from Git.

### `scripts/`

Contains development and demonstration scripts.

Current examples include:

```text
live_fire_detection.py
test_fire_model.py
```

### `tests/`

Contains automated tests for Smart Vision components.

Tests are organized by application module.

---

## Getting Started

### Prerequisites

The current development environment uses:

```text
Windows 11
Python 3.13.5
Git
```

A compatible webcam is required to run the live detection demo.

### 1. Clone the repository

```powershell
git clone https://github.com/1abhishekpandey2/Smart_Vision.git
cd Smart_Vision
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install runtime dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install development dependencies

```powershell
python -m pip install -r requirements-dev.txt
```

### 5. Install Smart Vision in editable mode

```powershell
python -m pip install -e .
```

### 6. Add the model

Place:

```text
models/firedetect-11s.pt
```

in the `models/` directory.

The model weights are not included in the Git repository.

### 7. Run the live demo

```powershell
python scripts/live_fire_detection.py
```

Available command-line options include:

```powershell
python scripts/live_fire_detection.py --help
```

Example:

```powershell
python scripts/live_fire_detection.py --device 0 --conf 0.4
```

Where:

```text
--device
    Webcam device index

--conf
    Detection confidence threshold

--model
    Path to the detection model
```

Press `q` in the video window to exit.

---

## Testing

Smart Vision uses `pytest` for automated testing.

Run the complete test suite from the project root:

```powershell
pytest
```

The current test suite covers:

* Camera lifecycle behavior
* Frame handling
* Consecutive camera read failures
* Preservation of the last successful frame
* Detector output conversion
* Class-name normalization

Testing is treated as part of feature development rather than something added after implementation.

---

## Development Workflow

Smart Vision follows:

> **Learn → Build → Improve → Integrate**

Each feature follows an incremental development process:

```text
Understand the problem
        ↓
Design the component
        ↓
Implement
        ↓
Test
        ↓
Refactor
        ↓
Document
        ↓
Commit
        ↓
Release
```

The goal is to build a software product while simultaneously developing strong software engineering and machine learning practices.

---

## Design Principles

The project follows several principles throughout development.

### Separation of Concerns

Each component should have a clearly defined responsibility.

```text
Camera
    → captures frames

Detector
    → detects objects

Tracker
    → maintains identities

Event Engine
    → determines meaningful events

Alerts
    → notifies users
```

### Encapsulation

Modules should own and control their internal state.

For example, camera status is controlled by the camera implementation rather than being freely writable by other modules.

### Abstraction

Other components should depend on Smart Vision interfaces rather than implementation-specific libraries wherever practical.

### Testability

Components should be designed so that external dependencies such as cameras and machine-learning models can be replaced with test doubles.

### Incremental Development

New capabilities should extend the existing architecture instead of replacing it with a separate implementation.

---

## Roadmap

### Phase 0 – Foundation

✅ Project foundation
✅ Repository structure
✅ Development environment
✅ Git workflow
✅ Logging
✅ Testing infrastructure

### Phase 1 – Vision Engine

✅ Camera abstraction
✅ Webcam input
✅ Frame representation
✅ Detection abstraction
✅ YOLO integration
✅ Fire and smoke detection baseline
✅ Live visualization

### Phase 2 – Tracking Engine

⬜ Multi-object tracking
⬜ Persistent object identities
⬜ Track lifecycle management

### Phase 3 – Home Intelligence

⬜ Scene understanding
⬜ Restricted zones
⬜ Basic activity reasoning

### Phase 4 – Event Engine

⬜ Detection-to-event conversion
⬜ Temporal consistency
⬜ Event severity
⬜ Event lifecycle

### Phase 5 – Database

⬜ PostgreSQL integration
⬜ Event persistence
⬜ Detection persistence
⬜ Historical queries

### Phase 6 – Backend API

⬜ FastAPI application
⬜ Camera endpoints
⬜ Detection endpoints
⬜ Event endpoints
⬜ System status endpoints

### Phase 7 – Dashboard

⬜ React dashboard
⬜ Live camera feed
⬜ Detection overlays
⬜ Event timeline
⬜ System monitoring

### Phase 8 – Alerts

⬜ Telegram alerts
⬜ Email alerts
⬜ Alert throttling
⬜ Alert history

### Phase 9 – Fall Detection

⬜ Human pose/activity representation
⬜ Fall detection model
⬜ Temporal fall confirmation
⬜ Fall event generation

### Phase 10 – Family Recognition

⬜ Face detection
⬜ Face embeddings
⬜ Family member recognition
⬜ Unknown-person detection

### Phase 11 – Advanced Fire & Smoke Intelligence

⬜ Temporal fire/smoke reasoning
⬜ False-positive reduction
⬜ Anomaly-aware fire detection
⬜ Model comparison
⬜ Edge optimization

### Phase 12 – Activity Understanding

⬜ Activity classification
⬜ Temporal activity modeling
⬜ Context-aware interpretation

### Phase 13 – Rule Engine

⬜ User-defined safety rules
⬜ Event conditions
⬜ Rule priorities
⬜ Automated responses

### Phase 14 – Edge Deployment

⬜ Raspberry Pi deployment
⬜ Resource optimization
⬜ Camera management
⬜ Lightweight inference

### Phase 15 – Smart Vision v1.0

⬜ Multi-camera support
⬜ Production deployment
⬜ Complete event pipeline
⬜ Dashboard
⬜ Alerts
⬜ Home analytics
⬜ End-to-end integration

---

## Current Development Status

Smart Vision is under active development.

The current release, **v0.1.0**, demonstrates a complete working path from webcam input to AI-based fire and smoke detection.

The system is intentionally being developed in small, testable increments. Each milestone should leave the project in a more useful state than before.

Future releases will extend the same architecture with tracking, temporal reasoning, event detection, databases, APIs, dashboards, alerts, and additional safety capabilities.

---

## Versioning

Smart Vision uses semantic versioning for releases:

```text
MAJOR.MINOR.PATCH
```

For example:

```text
v0.1.0
```

represents the first functional development release.

Future releases will use version numbers to communicate the scope and stability of changes.

---

## License

Smart Vision does not currently declare a project license.

Third-party libraries, pretrained models, datasets, and other external assets may have their own licenses and usage restrictions.

Licensing information for third-party components will be documented as the project matures and before a stable public release.
