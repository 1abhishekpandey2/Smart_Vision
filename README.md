
# Smart Vision

![Version](https://img.shields.io/badge/version-v0.2.0-blue)
![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![Tests](<https://img.shields.io/badge/tests-24%20passed-brightgreen>)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

**Intelligent Video Monitoring, Tracking, Spatial Reasoning, and Event Analysis Platform**

Smart Vision is a modular intelligent video-monitoring system built with Python, OpenCV, Ultralytics YOLO, and PyQt5.

The project began as a live fire and smoke detection system and has evolved into a structured video-intelligence platform capable of working with live and recorded video, detecting and tracking people, monitoring restricted areas, reasoning across multiple frames, generating intrusion events, and presenting results through a desktop monitoring interface.

---

# Version

Current release:

```text
v0.2.0
```

Smart Vision v0.2.0 represents the second major development milestone of the project.

The original v0.1.0 release focused primarily on:

```text
Webcam
   ↓
Frame Capture
   ↓
YOLO Fire / Smoke Detection
   ↓
Live Visualization
```

Version 0.2.0 expands the system into:

```text
Video Sources
      ↓
Detection
      ↓
Tracking
      ↓
Spatial Reasoning
      ↓
Temporal Reasoning
      ↓
Event Generation
      ↓
Desktop Monitoring Interface
```

The project is therefore moving beyond simple per-frame object detection toward a modular video-intelligence architecture.

---

# v0.2.0 Highlights

Smart Vision v0.2.0 introduces:

- Fire and smoke detection
- Temporal fire confirmation
- Person detection and tracking
- Persistent tracking IDs
- Smoothed track history
- Polygon restricted zones
- Intrusion entry and exit detection
- Intrusion trajectory visualization
- Recorded-video analysis
- Live webcam monitoring
- Playback timeline and seeking
- Playback speed controls
- Event history
- Live monitoring metrics
- Configurable detection confidence
- Asynchronous vision processing
- Latest-frame inference handling
- Modular camera, detection, tracking, reasoning, event, zone, and UI layers

---

# Core Features

## Fire and Smoke Detection

Smart Vision uses an Ultralytics YOLO detector for fire and smoke detection.

The current fire/smoke model is expected at:

```text
models/firedetect-11s.pt
```

The detector produces structured detection objects containing information such as:

```text
Class ID
Class Name
Confidence
Bounding Box
```

The detection implementation is separated from the camera and user-interface layers.

This allows the rest of Smart Vision to work with application-level detection objects instead of depending directly on Ultralytics result structures.

---

# Temporal Fire Reasoning

A single frame containing a possible fire detection should not automatically be treated as a confirmed fire event.

Smart Vision therefore contains a temporal persistence layer.

The current default temporal rule uses:

```text
Observation window: 10
Required positive fire observations: 7
```

Conceptually:

```text
Frame
  ↓
YOLO detection
  ↓
Fire present?
  ↓
Store observation
  ↓
Recent temporal window
  ↓
Enough positive observations?
  ↓
FIRE CONFIRMED
```

For example:

```text
Window size       = 10
Positive required = 7
```

A confirmation requires sufficient repeated evidence across the recent observation window.

The temporal component is implemented independently from the detector so that temporal reasoning can evolve without tightly coupling it to YOLO.

---

# Person Detection and Tracking

Smart Vision supports person tracking using an Ultralytics YOLO tracking pipeline.

Each tracked person receives a persistent track ID.

Example:

```text
Person #1
Person #2
Person #7
```

A tracked object contains information including:

```text
Track ID
Confidence
Bounding Box
Ground Point
```

The ground point is calculated from the bottom-center of the bounding box.

This is useful because zone reasoning should usually depend on where a person's feet or physical ground position are located rather than simply checking whether any part of the bounding box overlaps a zone.

---

# Track History

Smart Vision maintains movement history for tracked people.

The track-history layer provides:

```text
Bounded history
Movement filtering
Trajectory smoothing
Per-track storage
```

An exponential moving average is used to reduce visual jitter in movement trajectories.

Track history is stored internally even when the path is not currently visible.

This becomes useful for intrusion analysis.

---

# Restricted Zones

Operators can define polygon-shaped restricted areas directly over the video.

Current zone controls include:

```text
Create Zone
Edit Zone
Clear Zone
```

Zones are stored using normalized coordinates:

```text
0.0 <= x <= 1.0
0.0 <= y <= 1.0
```

Using normalized coordinates means the zone remains aligned correctly even if the video display is resized.

A person's ground point is checked against the polygon to determine whether the tracked person is:

```text
OUTSIDE
```

or:

```text
INSIDE
```

the restricted zone.

---

# Intrusion Monitoring

Smart Vision monitors transitions between outside and inside states.

The primary intrusion transition is:

```text
OUTSIDE
   ↓
Boundary crossing
   ↓
INSIDE
   ↓
INTRUSION EVENT
```

When the person leaves the area:

```text
INSIDE
   ↓
OUTSIDE
   ↓
EXIT EVENT
```

These transitions are recorded as events and displayed in the desktop application.

---

# Intrusion Trajectory Visualization

Smart Vision does not continuously display movement trails for every person.

Instead:

```text
Person is tracked
      ↓
Movement history stored silently
      ↓
Person remains outside zone
      ↓
No trajectory displayed
```

When the tracked person enters a restricted zone:

```text
OUTSIDE
   ↓
INSIDE
   ↓
INTRUSION
   ↓
Stored approach path becomes visible
```

Example:

```text
●
 \
  ●
   \
    ●
     \
      ● ─────────────▶ Person #4
                       Restricted Zone
```

This allows the operator to understand where the intruder approached from without cluttering the display with paths for every tracked person.

The trajectory is hidden again when the person exits the restricted zone.

The operator can also enable or disable intrusion-path visualization from the interface.

---

# Desktop Monitoring Interface

Smart Vision v0.2.0 introduces a PyQt5 desktop monitoring application.

Run it using:

```bash
python -m app
```

The interface is structured as a CCTV/video-management style application.

Conceptually:

```text
┌────────────────┬──────────────────────────────┬────────────────────┐
│                │                              │                    │
│    SOURCES     │          VIDEO VIEW          │    INTELLIGENCE    │
│                │                              │                    │
│  Webcam 0      │   Video                     │  Fire / Smoke      │
│                │   Bounding Boxes            │  Person Tracking   │
│  Open Video    │   Restricted Zones          │  Intrusion         │
│                │   Intrusion Paths           │  Confidence        │
│                │                              │  Zone Controls     │
│                │                              │                    │
├────────────────┴──────────────────────────────┴────────────────────┤
│                         PLAYBACK CONTROLS                           │
│                  Restart / Play / Pause / Timeline                 │
├────────────────────────────────────────────────────────────────────┤
│                          RECENT EVENTS                              │
└────────────────────────────────────────────────────────────────────┘
```

---

# Video Sources

## Recorded Video

Smart Vision supports analysis of recorded video files.

The video-file camera abstraction provides:

```text
FPS
Frame Count
Duration
Current Frame
Current Playback Time
Seeking
Restart
```

The desktop interface provides:

```text
Play
Pause
Restart
Timeline
Click-to-seek
Playback speed
```

Local MP4 recordings under:

```text
data/videos/
```

are ignored by Git so that large local video files are not accidentally committed.

---

## Live Webcam

Smart Vision v0.2.0 also supports live webcam monitoring.

The current interface exposes:

```text
Webcam 0
```

as a live source.

The same downstream intelligence pipeline is shared between recorded video and webcam input.

```text
Recorded Video ─────┐
                    │
                    ├────▶ Frame
                    │        │
Live Webcam ────────┘        ▼
                         VisionWorker
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Fire / Smoke                  Person Tracking
                │                           │
                ▼                           ▼
       Temporal Reasoning             Zone Reasoning
                │                           │
                └─────────────┬─────────────┘
                              ▼
                            Events
                              │
                              ▼
                              UI
```

The webcam therefore supports the same intelligence functions as recorded video:

```text
Fire / Smoke Detection
Person Tracking
Restricted Zones
Intrusion Monitoring
Track History
Intrusion Paths
Events
Metrics
```

---

# Vision Processing Architecture

Smart Vision is intentionally designed as a modular system.

The current architecture can be represented as:

```text
                       VIDEO SOURCES
                            │
                ┌───────────┴───────────┐
                │                       │
           WebcamCamera           VideoFileCamera
                │                       │
                └───────────┬───────────┘
                            │
                          Frame
                            │
                            ▼
                       UI Controller
                            │
                            ▼
                       VisionWorker
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      Fire/Smoke Detector           Person Tracker
              │                           │
              ▼                           ▼
     TemporalPersistence          TrackHistoryStore
              │                           │
              │                           ▼
              │                      PolygonZone
              │                           │
              │                           ▼
              │                    Intrusion Logic
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                       VisionResult
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
            VideoView     Events      Metrics
```

---

# Latest-Frame Processing Strategy

Video capture can operate faster than AI inference.

For example:

```text
Camera: 25-30 FPS
AI inference: slower than camera
```

If every captured frame were queued for AI processing, the system could gradually fall behind real time.

Smart Vision therefore uses a latest-frame strategy.

Example:

```text
Frame 1
   ↓
AI processing starts

Frame 2 arrives
Frame 3 arrives
Frame 4 arrives
Frame 5 arrives
```

Instead of creating a large queue:

```text
1 → 2 → 3 → 4 → 5
```

Smart Vision keeps the latest pending request:

```text
Frame 1 processing
        ↓
Frame 5 pending
        ↓
Frame 5 processed next
```

This helps maintain responsiveness for live video.

---

# Separation of Responsibilities

Smart Vision follows clear responsibility boundaries.

```text
Camera
    Captures frames

Detector
    Detects objects

Tracker
    Maintains identities

Track History
    Stores movement information

Zone
    Provides spatial context

Temporal Reasoning
    Evaluates observations over time

Event System
    Represents meaningful events

Vision Worker
    Coordinates AI processing

UI
    Presents information to the operator
```

The goal is to avoid building the entire system inside one large GUI or detection script.

---

# Project Structure

```text
Smart_Vision/
│
├── app/
│   │
│   ├── __main__.py
│   │
│   ├── camera/
│   │   ├── base.py
│   │   ├── status.py
│   │   ├── webcam.py
│   │   └── video_file.py
│   │
│   ├── core/
│   │   └── logger.py
│   │
│   ├── detection/
│   │   └── yolo.py
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── history.py
│   │   └── intrusion.py
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── temporal.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── track.py
│   │   ├── history.py
│   │   └── yolo_person.py
│   │
│   ├── zones/
│   │
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── seek_slider.py
│       ├── styles.py
│       ├── video_player.py
│       ├── video_view.py
│       ├── vision_types.py
│       ├── webcam_controller.py
│       │
│       └── workers/
│           ├── __init__.py
│           └── vision_worker.py
│
├── scripts/
│   ├── live_fire_detection.py
│   ├── live_person_tracking.py
│   ├── live_intrusion_monitor.py
│   └── smart_vision_monitor.py
│
├── tests/
│   ├── camera/
│   ├── detection/
│   └── reasoning/
│
├── models/
│
├── data/
│   ├── samples/
│   └── videos/
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

The exact structure will continue evolving as additional Smart Vision subsystems are introduced.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/1abhishekpandey2/Smart_Vision.git
```

Enter the project:

```bash
cd Smart_Vision
```

---

## 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The current application uses technologies including:

```text
Python
OpenCV
NumPy
Ultralytics
PyQt5
PyTorch / Ultralytics inference stack
pytest
Ruff
Black
```

---

# AI Models

Model weights are intentionally excluded from Git.

Place required models inside:

```text
models/
```

The current project expects models such as:

```text
models/firedetect-11s.pt
models/yolo26s.pt
```

The fire/smoke model is used for:

```text
Fire
Smoke
```

The person model is used for:

```text
Person detection
Tracking
Persistent track IDs
```

---

# Running Smart Vision

## Desktop Application

Start the main Smart Vision application using:

```bash
python -m app
```

The application will open the PyQt5 monitoring interface.

You can then choose:

```text
Webcam 0
```

or open a recorded video using:

```text
Open Video...
```

---

# Diagnostic Scripts

The project also retains standalone scripts for subsystem testing and development.

## Live Fire Detection

```bash
python scripts/live_fire_detection.py
```

View available options:

```bash
python scripts/live_fire_detection.py --help
```

---

## Person Tracking

```bash
python scripts/live_person_tracking.py
```

---

## Intrusion Monitoring

```bash
python scripts/live_intrusion_monitor.py
```

---

## Combined OpenCV Monitor

```bash
python scripts/smart_vision_monitor.py
```

These scripts are useful for testing individual capabilities independently from the PyQt5 desktop application.

---

# Testing

Smart Vision uses `pytest` for automated testing.

Run:

```bash
pytest
```

Current verified test result for v0.2.0 development:

```text
24 passed
```

Current automated tests cover areas including:

```text
Camera behavior
YOLO detector behavior
Temporal reasoning
```

Future versions will expand automated coverage for:

```text
Tracking
Zones
Events
Video playback
Webcam controller
Vision worker
UI integration
```

---

# Code Quality

Smart Vision uses Ruff and Black as part of the development workflow.

## Ruff

Run:

```bash
ruff check .
```

Ruff is used for linting and identifying code-quality problems.

---

## Black

Check formatting:

```bash
black --check .
```

Format the codebase:

```bash
black .
```

---

# Development Workflow

Meaningful Smart Vision changes follow a deliberate development and Git workflow.

```text
Requirement
    ↓
Design
    ↓
Implementation
    ↓
Testing
    ↓
Ruff
    ↓
Black
    ↓
Git Status
    ↓
Diff Review
    ↓
Deliberate Staging
    ↓
Staged Diff Review
    ↓
Commit
    ↓
Push
```

Changes are intentionally staged by logical feature rather than blindly grouping unrelated modifications into a single commit.

---

# Current Limitations

## Loitering / Dwell Detection

Loitering and dwell-time monitoring are planned features.

The current desktop interface contains disabled placeholders for dwell configuration.

Loitering detection is **not implemented in v0.2.0**.

---

## Camera Sources

The current desktop application supports:

```text
Recorded Video
Webcam 0
```

Features planned for future releases include:

```text
Automatic camera discovery
Multiple webcams
RTSP streams
IP CCTV cameras
Multi-camera monitoring
```

---

## Multiple Restricted Zones

The current system is focused on a single polygon restricted zone.

Future versions may support multiple independently configured zones.

---

## Configuration Persistence

Zone configuration and intelligence settings are currently runtime-oriented.

Future versions will persist application configuration between sessions.

---

## Automated Test Coverage

The project currently contains automated camera, detection, and temporal reasoning tests.

Additional testing is required for:

```text
Person tracking
Track history
Polygon zones
Intrusion events
Video playback
Webcam controller
Vision worker
Qt integration
```

---

# Roadmap

Future Smart Vision development is expected to include:

- Loitering and dwell-time reasoning
- Multiple restricted zones
- Camera discovery
- RTSP/IP CCTV integration
- Multi-camera monitoring
- Persistent configuration
- Improved event engine
- Event severity and lifecycle management
- Alert management
- Event-linked video playback
- Video recording
- Additional automated tests
- GPU inference support
- Performance profiling
- Better camera health monitoring
- Advanced spatial-temporal reasoning
- Research experiment tracking
- Improved event analytics

---

# Research Direction

Smart Vision is also being developed as an AI and software-engineering research platform.

Traditional object detection primarily answers:

```text
What object exists in this frame?
```

Smart Vision aims to progressively address a broader question:

```text
What meaningful event is occurring across this video sequence?
```

The longer-term direction therefore combines:

```text
Object Detection
       +
Tracking
       +
Spatial Context
       +
Temporal Reasoning
       +
Event Understanding
```

For example:

```text
Person detected
      ↓
Person tracked
      ↓
Movement history maintained
      ↓
Restricted-zone relationship evaluated
      ↓
Boundary crossing detected
      ↓
Intrusion event generated
```

Similarly:

```text
Fire detected in one frame
      ↓
Repeated evidence collected
      ↓
Temporal persistence evaluated
      ↓
Fire confirmed
```

This architecture provides a foundation for future research into spatial-temporal video understanding.

---

# Version History

## v0.2.0

Smart Vision v0.2.0 expands the original fire-detection system into a modular desktop video-intelligence platform.

### Added

- Temporal fire reasoning
- Configurable temporal fire window
- Person detection and tracking
- Persistent tracking IDs
- Ground-point reasoning
- Track-history storage
- Trajectory smoothing
- Polygon restricted zones
- Zone creation
- Zone editing
- Zone clearing
- Intrusion entry detection
- Intrusion exit detection
- Intrusion event generation
- Intrusion trajectory visualization
- Recorded-video camera abstraction
- Video playback
- Restart support
- Clickable playback timeline
- Video seeking
- Playback speed controls
- PyQt5 desktop monitoring interface
- Live Webcam 0 support
- Shared video/live AI processing pipeline
- Asynchronous VisionWorker
- Latest-frame inference strategy
- Event history
- Live monitoring metrics
- Configurable detection confidence

### Architecture

v0.2.0 introduces dedicated layers for:

```text
Camera
Detection
Tracking
Track History
Zones
Temporal Reasoning
Events
UI
```

The project no longer treats object detection as the complete application.

Instead, detection is one component within a larger video-intelligence pipeline.

### Known Limitations

v0.2.0 does not yet provide:

```text
Loitering / Dwell Detection
Multiple Simultaneous Zones
Persistent Zone Configuration
Automatic Camera Discovery
RTSP / IP CCTV Integration
Multi-Camera Monitoring
Comprehensive UI Tests
```

---

## v0.1.0

The first meaningful Smart Vision release.

### Added

- Initial project foundation
- Camera abstraction
- Camera lifecycle states
- Frame domain object
- Webcam implementation
- YOLO detector abstraction
- Fire and smoke detection
- Live webcam fire/smoke detection
- Initial automated tests
- Logging infrastructure

---

# Semantic Versioning

Smart Vision uses semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Current version:

```text
0.2.0
```

Because Smart Vision is still below version `1.0.0`, the architecture and APIs may continue evolving significantly as the project matures.

---

# Repository

```text
https://github.com/1abhishekpandey2/Smart_Vision
```

---

# Project Status

**Current Version: v0.2.0**

Smart Vision is under active development.

Version 0.2.0 establishes the project's first integrated desktop video-intelligence platform combining:

```text
Video Acquisition
       +
Detection
       +
Tracking
       +
Spatial Reasoning
       +
Temporal Reasoning
       +
Event Generation
       +
Operator Interface
```

The next development phase will focus on improving event intelligence, expanding camera-source support, increasing automated test coverage, introducing configuration persistence, and developing more advanced spatial-temporal reasoning capabilities.
