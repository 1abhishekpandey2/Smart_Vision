# Smart Vision

![Version](https://img.shields.io/badge/version-v0.2.0-blue)
![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

**Intelligent Video Monitoring, Tracking, Spatial Reasoning, and Event Analysis Platform**

Smart Vision is a modular AI-powered video monitoring platform designed to
transform conventional CCTV and webcam feeds into intelligent safety and
surveillance systems.

The project combines computer vision, object tracking, temporal reasoning,
restricted-zone analysis, and event generation to move beyond frame-by-frame
detection toward meaningful interpretation of video.

Current development includes:

- Fire and smoke detection
- Temporal fire reasoning
- Person detection and multi-object tracking
- Face registration and face-to-track recognition
- Track identity confirmation and retention
- Persistent tracking IDs
- Track-history reasoning
- Restricted-zone monitoring
- Intrusion entry and exit detection
- Dwell / loitering detection
- Live webcam monitoring
- Recorded-video analysis
- Event generation
- Live monitoring metrics
- PyQt5 desktop monitoring interface

## Quick start

Smart Vision currently targets Windows and Python 3.13+. Model weights and
personal face galleries are local files; they are not included in Git.

```powershell
git clone https://github.com/1abhishekpandey2/Smart_Vision.git
cd Smart_Vision
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app
```

Before enabling detections, place the required YOLO weights at the paths listed
in [AI models](#ai-models). In the app, open a video or select **Webcam 0**.
Face registration steps are in [Face registration and recognition](#face-registration-and-recognition).

---

# Version

Current released version:

```text
v0.2.0
```

Smart Vision `v0.2.0` represents the second major development milestone of the
project.

The original `v0.1.0` release focused primarily on:

```text
Webcam
   ↓
Frame Capture
   ↓
YOLO Fire / Smoke Detection
   ↓
Live Visualization
```

Version `v0.2.0` expanded the system into:

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

The project is therefore moving beyond simple per-frame object detection toward
a modular video-intelligence architecture.

---

# Current Development State

The `main` branch contains additional development completed after the
`v0.2.0` release.

Current post-v0.2.0 additions include:

- Zone dwell / loitering reasoning
- Configurable dwell-time threshold
- Per-track dwell-state management
- Continuous in-zone dwell timing
- `LOITERING` event generation
- Live loitering metrics
- Dwell-time information in person overlays
- Recorded-video timeline-aware dwell timing
- Monotonic dwell timing for live webcam monitoring
- Automated dwell reasoning tests
- Face-image registration into a persistent NPZ gallery
- Face-to-person association and track-level identity memory
- Configurable strong-match confirmation count

These features are currently part of ongoing development and have not yet been
assigned a new release tag.

---

# Current Capabilities

## Video Sources

Smart Vision currently supports:

- Live Webcam 0 input
- Recorded video files
- Play
- Pause
- Restart
- Timeline seeking
- Click-to-seek
- Playback speed control
- Shared AI processing for live and recorded sources

The vision pipeline uses a latest-frame processing strategy so that video
capture does not build an increasingly large inference queue when AI processing
is slower than frame acquisition.

---

# Fire and Smoke Detection

Smart Vision uses an Ultralytics YOLO detector for fire and smoke detection.

The current fire/smoke model is expected at:

```text
models/firedetect-11s.pt
```

The detector produces structured Smart Vision detection objects containing:

```text
Class ID
Class Name
Confidence
Bounding Box
```

The detection implementation is separated from the camera and user-interface
layers.

Conceptually:

```text
Frame
  ↓
YOLO
  ↓
Raw Model Output
  ↓
YOLODetector
  ↓
Smart Vision Detection Objects
```

This allows the rest of Smart Vision to operate on application-level domain
objects instead of depending directly on Ultralytics result structures.

The YOLO model currently serves as a practical baseline rather than being
assumed to be the final research architecture.

---

# Temporal Fire Reasoning

A possible fire detection in a single frame should not automatically be treated
as a confirmed fire event.

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
YOLO Detection
  ↓
Fire Present?
  ↓
Store Observation
  ↓
Recent Temporal Window
  ↓
Enough Positive Observations?
  ↓
FIRE CONFIRMED
```

For example:

```text
Window size       = 10
Positive required = 7
```

A confirmation requires sufficient repeated evidence across the recent
observation window.

The system can generate higher-level fire events when the temporal state
changes:

```text
Possible detections
      ↓
Temporal evidence accumulates
      ↓
FIRE confirmed
```

and:

```text
Confirmed fire
      ↓
Evidence decreases
      ↓
FIRE cleared
```

The temporal component is implemented independently from the detector so that
temporal reasoning can evolve without tightly coupling it to YOLO.

This deterministic temporal approach provides an interpretable baseline for
future comparison with more advanced learned temporal methods.

---

# Person Detection and Tracking

Smart Vision supports person detection and tracking using an Ultralytics-based
tracking pipeline.

Each tracked person receives a persistent track ID.

Example:

```text
Person #1
Person #2
Person #7
```

A tracked person contains information including:

```text
Track ID
Confidence
Bounding Box
Ground Point
```

The ground point is calculated using the bottom-center of the person's bounding
box.

Conceptually:

```text
        Bounding Box
     ┌───────────────┐
     │               │
     │    Person     │
     │               │
     └───────●───────┘
             ↑
        Ground Point
```

This is useful because spatial reasoning should normally evaluate where a
person physically meets the ground rather than determining zone membership
from any overlap between the person's bounding box and a polygon.

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

An exponential moving average is used to reduce visual jitter in movement
trajectories.

Conceptually:

```text
Raw Ground Point
       ↓
Movement Filtering
       ↓
Trajectory Smoothing
       ↓
Track History
```

Track history can be stored internally even when a path is not being displayed
to the operator.

This allows Smart Vision to reveal useful trajectory information when an
intrusion actually occurs without permanently cluttering the video with trails
for every tracked person.

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

Using normalized coordinates allows a polygon to remain aligned with the video
scene when the displayed video area changes size.

A tracked person's ground point is evaluated against the polygon.

Conceptually:

```text
Person Track
     ↓
Ground Point
     ↓
Polygon Test
     ↓
OUTSIDE / INSIDE
```

The same polygon relationship can currently support both:

```text
Intrusion Reasoning
        +
Dwell / Loitering Reasoning
```

---

# Intrusion Monitoring

Smart Vision monitors transitions between outside and inside states.

The primary intrusion transition is:

```text
OUTSIDE
   ↓
Boundary Crossing
   ↓
INSIDE
   ↓
INTRUSION EVENT
```

When the tracked person leaves the restricted area:

```text
INSIDE
   ↓
Boundary Crossing
   ↓
OUTSIDE
   ↓
EXIT EVENT
```

A person that is first observed already inside the zone is not automatically
treated as having crossed the boundary.

This distinction prevents the system from generating a false intrusion-entry
event without observing an actual outside-to-inside transition.

Intrusion and exit events are displayed in the desktop application's recent
event table.

---

# Dwell / Loitering Detection

Smart Vision can detect when a tracked person remains continuously inside a
restricted zone longer than a configurable period.

Dwell reasoning operates independently for each tracking ID.

Conceptually:

```text
Person Tracked
      ↓
Ground Point Evaluated
      ↓
Inside Restricted Zone
      ↓
Dwell Timer Starts
      ↓
Person Remains Inside
      ↓
Configured Threshold Reached
      ↓
LOITERING EVENT
```

For example, if the configured dwell threshold is:

```text
15 seconds
```

a tracked person that continuously remains inside the restricted zone for
15 seconds reaches the loitering state.

The current behavior is:

```text
Enter / First Observed Inside
      ↓
Start Dwell Timer
      ↓
Remain Inside
      ↓
Accumulate Dwell Time
      ↓
Threshold Reached
      ↓
Generate One LOITERING Event
      ↓
Continue Monitoring
      ↓
Exit Zone
      ↓
Reset Dwell State
```

A `LOITERING` event is generated once for a continuous dwell session rather
than once on every processed frame.

After the tracked person exits the zone, the dwell state for that session is
reset.

If the person later enters again, a new dwell session begins.

The desktop interface allows the operator to configure the allowed dwell time.

Example:

```text
Allowed dwell: 15 sec
```

---

## Dwell Time Sources

Smart Vision deliberately uses different timing sources for recorded and live
video.

### Recorded Video

For recorded footage:

```text
Video Timeline Time
```

is used for dwell reasoning.

This means that a 15-second dwell represents 15 seconds of source-video time
rather than 15 seconds of computer processing time.

Changing playback speed therefore does not redefine the semantic duration of
the event.

Conceptually:

```text
Recorded Video Timeline
         ↓
Frame Position
         ↓
Dwell Timing
```

### Live Webcam

For a live webcam:

```text
Monotonic Runtime Time
```

is used.

A monotonic clock is appropriate because live dwell duration represents actual
elapsed monitoring time and should not be affected by changes to the computer's
wall-clock time.

---

# Intrusion Trajectory Visualization

Smart Vision does not continuously display movement trails for every tracked
person.

Instead:

```text
Person Tracked
      ↓
Movement History Stored
      ↓
Person Remains Outside Zone
      ↓
No Intrusion Path Displayed
```

When the tracked person crosses into a restricted zone:

```text
OUTSIDE
   ↓
INSIDE
   ↓
INTRUSION
   ↓
Stored Approach Path Becomes Visible
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

This allows the operator to understand where an intruding person approached
from without permanently displaying movement trails for every tracked object.

The intrusion trajectory is hidden again when the person's active intrusion
state ends.

The operator can enable or disable path visualization through the desktop
interface.

---

# Event Types

Smart Vision converts lower-level detections and tracking state into
higher-level events.

Current event concepts include:

| Event         | Meaning                                                           |
| ------------- | ----------------------------------------------------------------- |
| `FIRE`      | Temporal fire state becomes confirmed or clears                   |
| `INTRUSION` | A tracked person crosses from outside to inside a restricted zone |
| `EXIT`      | A tracked person moves from inside to outside the restricted zone |
| `LOITERING` | A tracked person exceeds the configured continuous dwell duration |

Conceptually:

```text
Detection / Track State
          ↓
Reasoning
          ↓
Meaningful Event
```

This is an important architectural direction for Smart Vision.

The system should not treat every raw neural-network output as an operator-level
alert.

---

# Desktop Monitoring Interface

Smart Vision includes a PyQt5 desktop monitoring application.

Run it using:

```bash
python -m app
```

The interface follows a CCTV/video-management style layout.

Conceptually:

```text
┌────────────────┬──────────────────────────────┬────────────────────┐
│                │                              │                    │
│    SOURCES     │          VIDEO VIEW          │    INTELLIGENCE    │
│                │                              │                    │
│  Webcam 0      │   Video                     │  Fire / Smoke      │
│                │   Bounding Boxes            │  Person Tracking   │
│  Open Video    │   Restricted Zones          │  Intrusion         │
│                │   Intrusion Paths           │  Loitering / Dwell │
│                │   Dwell Information         │  Dwell Threshold   │
│                │                              │  Confidence        │
│                │                              │  Zone Controls     │
│                │                              │                    │
├────────────────┴──────────────────────────────┴────────────────────┤
│                         PLAYBACK CONTROLS                           │
│                  Restart / Play / Pause / Timeline                 │
├────────────────────────────────────────────────────────────────────┤
│                          RECENT EVENTS                              │
└────────────────────────────────────────────────────────────────────┘
```

Current desktop controls include:

- Source selection
- Webcam 0 monitoring
- Recorded-video loading
- Play
- Pause
- Restart
- Timeline seeking
- Playback speed adjustment
- Fire / smoke enable or disable
- Person tracking enable or disable
- Face recognition enable or disable
- Strong-match count for identity confirmation
- In-app face registration
- Intrusion monitoring
- Dwell / loitering monitoring
- Configurable dwell duration
- Track-path visibility
- Restricted-zone creation
- Restricted-zone editing
- Restricted-zone clearing
- Fire confidence threshold
- Person confidence threshold
- Event table
- Live metrics

Current live metrics include:

```text
Tracked People
Fire Evidence
Inside Zone
Loitering
```

---

# Video Sources

## Recorded Video

Smart Vision supports analysis of recorded video files.

The video-file subsystem provides information and operations including:

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
Click-to-Seek
Playback Speed
```

Local recordings can be placed under:

```text
data/videos/
```

Large local video files should not be committed to Git.

---

## Live Webcam

Smart Vision also supports live webcam monitoring.

The current desktop interface exposes:

```text
Webcam 0
```

as the current live camera source.

Recorded video and live webcam input feed the same downstream intelligence
pipeline.

Conceptually:

```text
Recorded Video ─────┐
                    │
                    ├────▶ Frame
                    │        │
Live Webcam ────────┘        ▼
                         VisionWorker
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Fire / Smoke               Person Tracking
                 │                         │
                 ▼                         ▼
        Temporal Reasoning           Spatial Reasoning
                 │                         │
                 │                 ┌───────┴───────┐
                 │                 │               │
                 │                 ▼               ▼
                 │            Intrusion          Dwell
                 │                 │               │
                 └─────────────────┴───────┬───────┘
                                           ▼
                                         Events
                                           │
                                           ▼
                                           UI
```

The live source can therefore use the same intelligence features as recorded
video:

```text
Fire / Smoke Detection
Temporal Fire Reasoning
Person Tracking
Restricted Zones
Intrusion Monitoring
Dwell / Loitering Monitoring
Track History
Intrusion Paths
Events
Live Metrics
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
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
      Fire / Smoke Detector              Person Tracker
              │                               │
              ▼                               ▼
     TemporalPersistence               TrackHistoryStore
              │                               │
              │                               ▼
              │                          Ground Point
              │                               │
              │                               ▼
              │                          PolygonZone
              │                               │
              │                      ┌────────┴────────┐
              │                      │                 │
              │                      ▼                 ▼
              │               Intrusion Logic      DwellTracker
              │                      │                 │
              │                INTRUSION/EXIT      LOITERING
              │                      │                 │
              └──────────────────────┴────────┬────────┘
                                             │
                                             ▼
                                        VisionResult
                                             │
                                ┌────────────┼────────────┐
                                │            │            │
                                ▼            ▼            ▼
                            VideoView      Events       Metrics
```

This separates:

```text
Per-Frame Detection
        +
Tracking
        +
Spatial Context
        +
Temporal Reasoning
        +
Event Generation
```

into distinct responsibilities.

---

# Latest-Frame Processing Strategy

Video capture can operate faster than AI inference.

For example:

```text
Camera:       25-30 FPS
AI inference: slower than camera
```

If every captured frame were queued for AI processing, the application could
gradually fall farther behind the actual video.

Smart Vision therefore uses a latest-frame processing strategy.

Example:

```text
Frame 1
   ↓
AI Processing Starts

Frame 2 Arrives
Frame 3 Arrives
Frame 4 Arrives
Frame 5 Arrives
```

Instead of creating a queue:

```text
1 → 2 → 3 → 4 → 5
```

Smart Vision retains the newest pending request:

```text
Frame 1 Processing
        ↓
Frame 5 Pending
        ↓
Frame 5 Processed Next
```

Intermediate pending frames can therefore be replaced by newer frames.

This helps keep live monitoring responsive instead of turning AI inference into
an increasingly delayed playback pipeline.

---

# Runtime State Reset

Stateful reasoning must be reset when the underlying video context changes.

Smart Vision resets relevant runtime state during operations such as:

```text
Video Seek
Restart
Zone Modification
Intelligence Configuration Change
Source Change
```

State that may be reset includes:

```text
Temporal Fire State
Tracking History
Intrusion State
Dwell State
```

This prevents reasoning from a previous scene position or configuration from
incorrectly carrying into a new context.

---

# Separation of Responsibilities

Smart Vision follows clear responsibility boundaries.

```text
Camera
    Captures frames

Detector
    Detects objects

Tracker
    Maintains tracked identities

Track History
    Stores movement information

Zone
    Provides spatial context

Temporal Reasoning
    Evaluates observations over time

Dwell Reasoning
    Measures continuous in-zone duration

Event System
    Represents meaningful events

Vision Worker
    Coordinates AI and reasoning processing

UI
    Presents information to the operator
```

The goal is to avoid building the entire system inside one large GUI class or
single detection script.

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
│   │   ├── detection.py
│   │   ├── detector.py
│   │   └── yolo.py
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── history.py
│   │   └── intrusion.py
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   └── dwell.py
│   │
│   ├── recognition/
│   │   ├── association.py
│   │   ├── identity_memory.py
│   │   ├── insightface_embedder.py
│   │   └── registry.py
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
│   ├── smart_vision_monitor.py
│   ├── build_face_registry.py
│   └── test_face_registry.py
│
├── tests/
│   ├── camera/
│   ├── detection/
│   ├── reasoning/
│   └── recognition/
│
├── models/
│   └── (local model weights; not tracked)
│
├── data/
│   └── (local videos, face photos, and gallery; not tracked)
│
├── docs/research/
│   └── experiment_log.md
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

The exact project structure will continue evolving as additional Smart Vision
subsystems are introduced.

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

Use Python 3.13. On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development tools, install the runtime dependencies first, then the
development requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The application currently uses technologies including:

```text
Python 3.13
OpenCV
NumPy
Ultralytics and PyTorch
PyQt5
InsightFace and ONNX Runtime
```

The development requirements add pytest, Ruff, and Black.

---

# AI Models

Model weights are intentionally excluded from Git.

Place the required models inside:

```text
models/
```

The current application expects models such as:

```text
models/firedetect-11s.pt
models/yolo26s.pt
```

The model files are not stored in this repository. Obtain the fire/smoke
weights from the project owner or their authorized model source, and obtain a
compatible person model from its model provider. Save both files under
`models/` using the exact names above. Smart Vision reports a missing-model
error when detection is enabled before the files are present.

The fire/smoke model is used for:

```text
Fire
Smoke
```

The person model is used for:

```text
Person Detection
Person Tracking
Persistent Track IDs
```

The current models are working baselines.

Smart Vision does not assume that the current YOLO-based approach is the final
research solution.

---

# Face Registration and Recognition

Face recognition is optional and requires person tracking to be enabled. The
gallery is private local data and is not committed to Git.

## Register a person in the desktop app

1. Run `python -m app` from the repository root.
2. In **Face Registration**, enter the person’s name.
3. Select **Add Images to Gallery** and choose one or more JPG, JPEG, or PNG
   photos. Each photo must contain exactly one detectable face.
4. Wait for the registration status message before closing the app.

Smart Vision computes all selected image embeddings before saving. If any
selected image cannot be read or does not contain exactly one face, the batch
fails without replacing the existing gallery. Successful registration adds
the samples to `data/face_registry.npz`. Registering `Alex` and then `alex`
adds both sets of samples to the same identity and preserves the first saved
spelling.

For a bulk initial gallery, organize photos into one directory per identity,
then run:

```powershell
python scripts/build_face_registry.py <photo-root> data/face_registry.npz
```

This command writes a new gallery file; use the desktop registration flow to
append images to an existing gallery.

The `.npz` gallery contains biometric embeddings. It is excluded by
`.gitignore`; back it up privately if you need to move it to another machine.
Do not commit face photos or the gallery.

## Show identities in video

1. Open a video or select **Webcam 0**.
2. Keep **Person Tracking** enabled and turn on **Face Recognition**.
3. The worker associates a face only when it unambiguously falls inside one
   person box. Ambiguous faces are left unassigned.
4. The default identity rule requires three strong matches within a two-second
   source-time window. The required count can be changed in the Intelligence
   panel. A confirmed name is retained while that person track stays active.

The gallery uses a candidate floor of 0.35 and a strong-match cutoff of 0.50.
These are provisional similarity cutoffs, not probabilities or validated
operating points. The similarity margin is diagnostic only.

InsightFace initializes its `buffalo_l` face-analysis model on first use. Allow
extra startup time and access to its model assets the first time you register
photos or enable face recognition.

---

# Running Smart Vision

## Desktop Application

Start the main application using:

```bash
python -m app
```

The PyQt5 monitoring interface will open.

You can then select:

```text
Webcam 0
```

or load recorded video using:

```text
Open Video...
```

---

# Diagnostic Scripts

Smart Vision also retains standalone scripts for subsystem testing and
development.

These scripts allow components to be evaluated separately from the integrated
desktop interface.

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

Diagnostic scripts are useful for isolating subsystem behavior while the main
application continues to evolve.

---

# Testing

Smart Vision uses `pytest` for automated testing.

Run:

```bash
pytest
```

Automated tests cover camera behavior, YOLO output conversion, face embedding
and association, face registration, track identity memory, temporal fire
reasoning, and dwell / loitering reasoning. A real-model integration test uses
the local fire model and sample image when both are present; otherwise pytest
skips it. The test suite does not require private face photos or a local face
gallery.

Run the full suite:

```bash
python -m pytest -q
```

Run tests that do not use model weights or sample media:

```bash
python -m pytest -q -m "not integration"
```

The dwell reasoning tests currently verify behavior including:

- dwell timer initialization
- dwell accumulation while a track remains inside
- threshold triggering
- one trigger per continuous dwell session
- reset after leaving the zone
- threshold validation

A successful application run and a successful automated test suite are treated
as different forms of validation.

---

# Code Quality

Smart Vision uses Ruff and Black as part of the development workflow.

## Ruff

Run:

```bash
ruff check .
```

Ruff is used for linting and detecting common code-quality problems.

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

## Recommended Quality Check

Before a meaningful commit:

```bash
black --check .
ruff check .
pytest
```

All three should be reviewed before committing feature work.

---

# Development Workflow

Meaningful Smart Vision changes follow a deliberate engineering and Git
workflow.

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
Whitespace Check
    ↓
Commit
    ↓
Push
```

Typical Git review commands include:

```bash
git status
git diff
git diff --cached
git diff --cached --check
```

Changes are intentionally staged by logical feature rather than blindly
grouping unrelated modifications into the same commit.

For example:

```bash
git add app/reasoning/dwell.py
git add tests/reasoning/test_dwell.py
git add app/ui/vision_types.py
git add app/ui/workers/vision_worker.py
git add app/ui/main_window.py
```

is preferable to staging unrelated project files simply because they happen to
exist in the working tree.

---

# Current Limitations

Smart Vision is under active development.

The following limitations are known and expected to evolve.

---

## Tracking and Dwell Robustness

Dwell reasoning is keyed by person tracking ID.

The current implementation does not yet include a dedicated missing-track
expiry or grace-period policy for dwell sessions.

Temporary tracking loss, long observation gaps, or track-ID reassignment can
therefore affect continuous dwell interpretation.

Future work may introduce:

```text
Track-Loss Tolerance
Track Re-Association
Dwell Session Expiry
Short Occlusion Handling
```

---

## Camera Sources

The current desktop application supports:

```text
Recorded Video
Webcam 0
```

Future camera-source support may include:

```text
Automatic Camera Discovery
Multiple Webcams
RTSP Streams
IP CCTV Cameras
Multi-Camera Monitoring
```

---

## Multiple Restricted Zones

The current system is focused on one polygon restricted zone for the active
scene.

Future versions may support:

```text
Multiple Zones
Zone Names
Zone-Specific Rules
Different Dwell Thresholds Per Zone
Zone-Specific Event Policies
```

---

## Configuration Persistence

Restricted zones and intelligence settings are currently primarily
runtime-oriented.

Future versions are expected to persist application configuration between
sessions.

Potential persistent settings include:

```text
Zones
Confidence Thresholds
Dwell Threshold
Enabled Intelligence Modules
Camera Configuration
```

---

## Automated Test Coverage

Automated unit tests cover camera behavior, detection conversion, temporal
fire reasoning, dwell reasoning, face embedding conversion, face association,
face registration, identity memory, and worker label behavior. The optional
real-model check is skipped when local model weights or sample media are not
available. These tests do not replace live-camera, full-window Qt, or
annotated-video evaluation.

---

## Formal Performance Evaluation

The current application has working functionality, but comprehensive formal
benchmarking has not yet been completed.

Future evaluation should measure metrics such as:

```text
Precision
Recall
F1 Score
mAP
False Alarm Rate
False Negative Rate
Inference Latency
FPS
CPU Usage
GPU Usage
Memory Consumption
```

Performance should eventually be evaluated on representative Smart Vision
scenarios rather than inferred from isolated demonstrations.

---

# Roadmap

Future Smart Vision development is expected to include:

- Improved dwell / loitering robustness
- Missing-track handling for dwell reasoning
- Track re-association
- Multiple restricted zones
- Zone-specific dwell thresholds
- Camera discovery
- Multiple webcam support
- RTSP / IP CCTV integration
- Multi-camera monitoring
- Persistent configuration
- Controlled evaluation of face association and identity memory
- Improved event engine
- Event lifecycle management
- Event severity
- Event cooldown and deduplication
- Alert management
- Telegram / email notifications
- Event-linked video playback
- Video recording
- Additional automated tests
- GPU inference support
- Performance profiling
- Better camera health monitoring
- Fall detection
- Activity understanding
- Anomaly detection
- Advanced spatial-temporal reasoning
- Expanded research evaluation and reproducibility
- Event analytics
- Edge deployment and optimization

---

# Research Direction

Smart Vision is also being developed as an AI and software-engineering research
platform.

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

---

## Intrusion Reasoning Example

```text
Person Detected
      ↓
Person Tracked
      ↓
Movement History Maintained
      ↓
Restricted-Zone Relationship Evaluated
      ↓
Boundary Crossing Detected
      ↓
INTRUSION Event Generated
```

---

## Fire Reasoning Example

```text
Fire Detected in One Frame
      ↓
Repeated Evidence Collected
      ↓
Temporal Persistence Evaluated
      ↓
Fire Confirmed
```

---

## Dwell Reasoning Example

Dwell reasoning combines identity, spatial context, and time:

```text
Person Detected
      ↓
Persistent Track ID
      ↓
Person Located Inside Restricted Zone
      ↓
Continuous Dwell Duration Measured
      ↓
Configured Duration Exceeded
      ↓
LOITERING Event Generated
```

This demonstrates why meaningful video understanding requires more than
per-frame object detection.

---

# Research Development Strategy

The intended research process is:

```text
Working Baseline
      ↓
Observe Failure Modes
      ↓
Identify Research Problem
      ↓
Form Hypothesis
      ↓
Develop Improved Method
      ↓
Experiment
      ↓
Ablation Study
      ↓
Compare Baselines
      ↓
Measure Accuracy + Efficiency
      ↓
Document Results
      ↓
Research Paper
```

The current system provides interpretable engineering baselines such as:

```text
YOLO Fire Detection
        +
Temporal Persistence

Person Tracking
        +
Polygon Reasoning

Tracking
        +
Dwell Timer
```

These can later be compared against more advanced learned spatial-temporal
approaches.

Potential future research directions include:

- learned temporal fire/smoke reasoning
- lightweight video models
- ConvLSTM-based temporal modeling
- temporal transformers
- video anomaly detection
- context-aware event reasoning
- multimodal sensing
- edge-oriented AI optimization

Novelty should be established experimentally rather than assumed.

---

# Design Principles

Smart Vision follows several development principles.

## Modularity

Major concerns should remain separate:

```text
Camera
Detection
Tracking
Spatial Reasoning
Temporal Reasoning
Events
UI
```

---

## Domain Objects

External library result formats should be converted into Smart Vision objects
at module boundaries.

Example:

```text
Ultralytics Result
       ↓
YOLODetector
       ↓
Detection
```

This reduces coupling between the rest of the application and third-party
library internals.

---

## Avoid Premature Abstraction

New managers, factories, service layers, and configuration systems should only
be introduced when the project has a concrete need for them.

The goal is:

```text
Simple
Clear
Testable
Extensible
```

rather than abstraction for its own sake.

---

## Explicit State

Stateful reasoning such as:

```text
Temporal Fire Evidence
Intrusion State
Tracking History
Dwell State
```

should have explicit lifecycle and reset behavior.

---

## Testability

Deterministic reasoning modules should be independently testable without
requiring physical cameras or running full neural-network inference.

The dwell tracker and temporal persistence layer are examples of reasoning that
can be unit tested independently.

---

# Version History

## Unreleased

Current development on `main`.

### Added

- Dwell / loitering reasoning
- Configurable dwell-time threshold
- Per-track dwell-state management
- Continuous in-zone timing
- `LOITERING` event generation
- Loitering count metric
- Dwell-time person overlays
- Recorded-video timeline-based dwell timing
- Monotonic live-webcam dwell timing
- Automated dwell-reasoning tests
- Face recognition with face-to-person association
- Per-track identity confirmation and retention
- In-app registration to a persistent local face gallery
- Case-insensitive identity names that preserve the original spelling
- Automated face recognition, registration, and identity-memory tests

### Changed

- Restricted-zone membership can now support both intrusion and dwell reasoning
- Zone reasoning is no longer exclusively dependent on intrusion monitoring
- Recorded-video source time is propagated into vision requests
- Runtime resets now also clear dwell reasoning state
- Model-dependent test skips when local model and sample assets are absent

### Current Known Limitation

Dwell reasoning does not yet include a dedicated missing-track expiry or
occlusion-grace policy.

---

## v0.2.0

Smart Vision `v0.2.0` expanded the original fire-detection application into a
modular desktop video-intelligence platform.

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
- Asynchronous `VisionWorker`
- Latest-frame inference strategy
- Event history
- Live monitoring metrics
- Configurable detection confidence

### Architecture

Version `v0.2.0` introduced dedicated layers for:

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

Instead:

```text
Detection
    ↓
Reasoning
    ↓
Event
    ↓
Operator Interface
```

became the direction of the system architecture.

### Known Limitations at v0.2.0 Release

The `v0.2.0` release did not yet provide:

```text
Loitering / Dwell Detection
Multiple Simultaneous Zones
Persistent Zone Configuration
Automatic Camera Discovery
RTSP / IP CCTV Integration
Multi-Camera Monitoring
Comprehensive UI Tests
```

Dwell / loitering reasoning was subsequently implemented on the `main` branch
after this release.

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

The primary `v0.1.0` pipeline was:

```text
Camera
  ↓
Frame
  ↓
YOLO Fire / Smoke Detection
  ↓
Detection Objects
  ↓
Live Visualization
```

---

# Semantic Versioning

Smart Vision uses semantic versioning:

```text
MAJOR.MINOR.PATCH
```

Current released version:

```text
0.2.0
```

The current `main` branch contains additional unreleased development.

Because Smart Vision remains below version `1.0.0`, architecture and APIs may
continue evolving significantly as the project matures.

A new version is created for meaningful release milestones rather than every
individual code change.

---

# Repository

```text
https://github.com/1abhishekpandey2/Smart_Vision
```

---

# Project Status

**Current Released Version:** `v0.2.0`

**Development Branch State:** Post-v0.2.0 development

Smart Vision is under active development.

The current system combines:

```text
Video Acquisition
       +
Detection
       +
Tracking
       +
Track History
       +
Spatial Reasoning
       +
Temporal Reasoning
       +
Dwell Reasoning
       +
Event Generation
       +
Operator Interface
```

The project has progressed from:

```text
Detect something in a frame
```

toward:

```text
Track what is happening
        ↓
Understand where it is happening
        ↓
Understand how long it is happening
        ↓
Generate a meaningful event
```

The next development phases will focus on improving event intelligence,
strengthening tracking and dwell robustness, expanding camera-source support,
increasing automated integration coverage, introducing configuration
persistence, and developing more advanced spatial-temporal reasoning.

---

# License

See the repository's `LICENSE` file for licensing information.
