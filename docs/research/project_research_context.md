# Smart Vision — project and research-paper context

**Prepared:** 2026-11-14

**Repository snapshot:** base commit `a477155` plus the timestamped notification changes described here
**Purpose:** keep the project’s research framing, architecture, development decisions, evidence boundaries, and reproducibility context in one place for later paper writing.

Read this file together with [the experiment log](experiment_log.md). This document is the project-context and methods companion; the log is the dated record of comparisons, runs, anomalies, and results. They describe the project-specific information currently preserved in the repository. They do not replace external literature citations, raw datasets, or missing run artifacts.

## 1. Executive summary

Smart Vision is a modular desktop prototype for video-based home and small-site safety monitoring. It combines fire and smoke object detection, person detection and tracking, polygon-zone reasoning, temporal fire confirmation, dwell/loitering rules, face-to-track association, and track-level identity memory. A PyQt5 application accepts Webcam 0 or a recorded video and displays detections, track labels, event messages, and metrics.

The project is best described at present as an **implemented and software-tested system prototype with preliminary, small-sample face-similarity observations**. It is not yet a benchmarked safety system or a validated face-recognition study. The repository has automated tests and a handful of local or user-reported checks, but no fully specified held-out evaluation dataset, annotated CCTV benchmark, live-camera performance study, or comprehensive detector evaluation.

Important distinctions for any paper:

- Temporal persistence is applied to **fire** detections. Smoke is detected and displayed, but the retrieved worker does not apply the fire persistence rule to smoke.
- Intrusion entry requires an observed tracked **outside-to-inside transition**. A person first observed already inside a zone does not provide that transition.
- Dwell/loitering logic is implemented separately from intrusion-entry logic.
- Face similarity cutoffs 0.35 and 0.50 are provisional settings motivated by a small reported image-pair evaluation. They are not probabilities or validated operating points.
- The similarity margin is diagnostic only; it is not an acceptance gate.
- Face-to-person association and track identity memory are implemented in the current repository, but the **NPZ face gallery is not live identity memory**. The gallery stores identity-labeled face embeddings; track memory stores temporary per-track evidence and a confirmed name at runtime.
- Live registration uses a manually captured frontal reference, then provisional five-landmark ratios to auto-capture four prompted poses after three consecutive matching checks. A manual override remains available; the ratios have not been camera-validated.

## 2. Project scope and research framing

The application was developed as a modular video-intelligence prototype. The intended engineering contribution is the integration of independent visual detectors with explicit, inspectable temporal and spatial rules in a desktop workflow. The prototype is designed to expose state and settings to an operator rather than hide all decisions inside a single opaque pipeline.

The repository does not establish a finalized thesis statement or a single adopted research question. Candidate paper framing should be selected only after evaluation data and a protocol are available. Possible questions, explicitly **proposed rather than established**, include:

1. How does an explicit temporal-persistence rule change fire-alert behavior relative to per-frame detections on annotated video?
2. How do face-association and track-identity confirmation policies affect false identity assignments, coverage, and confirmation delay?
3. What is the operational cost and behavior of integrating these modules in the desktop pipeline on representative hardware and video sources?

The face-registration user flow is a product capability, not itself evidence that recognition accuracy improved. In particular, five guided views have not yet been compared against one-view or photo-based registration on controlled test data.

## 3. System architecture and data flow

### 3.1 High-level flow

```text
Webcam 0 or recorded video
          ↓
Frame(image, timestamp, camera_id)
          ↓
PyQt application submits a VisionRequest
          ↓
VisionWorker in a QThread
   ┌──────┼───────────────┐
   │      │               │
 YOLO   YOLO person     InsightFace
 fire/  tracking        face detection
 smoke  + track IDs     + embeddings
   │      │               │
 fire    zone, dwell,    face-to-track
 temporal intrusion      association
 logic   and history         ↓
   │      │             NPZ gallery match
   │      │             + track identity memory
   └──────┴───────────────┘
          ↓
VisionResult: overlays, paths, counts, evidence, events
          ↓
Video view and desktop metrics/event list
```

`Frame` is the source boundary. A `VisionRequest` carries the frame, current settings, normalized polygon points, source time in seconds, a generation number, and an optional recording path. A `VisionResult` carries overlays, paths, counts, fire evidence, confirmation status, zone state, and timestamped events. This keeps model-specific result objects out of most UI and reasoning code.

The desktop application runs `VisionWorker` on a `QThread`. Model loading is lazy: fire, person, and face components are initialized only when their feature is first requested. The UI uses a latest-frame policy: when inference is busy, the pending request is replaced with the newest frame instead of accumulating a long queue. This favors a current preview over processing every captured frame. It also means processed observations may be skipped when inference is slower than capture; this matters when interpreting observation-count and latency behavior.

Video source time is used for recorded-video reasoning. Live webcam reasoning uses a monotonic clock. Seeking, changing analysis settings, or changing the zone resets relevant runtime state and stale generations are ignored. These choices are in `app/ui/main_window.py` and `app/ui/workers/vision_worker.py`.

### 3.2 Main implementation modules

| Concern | Main source locations | Responsibility |
|---|---|---|
| Camera/frame abstraction | `app/camera/` | Webcam and video frame acquisition; frame metadata and status. |
| Fire/smoke detection | `app/detection/` | Convert YOLO predictions into application `Detection` objects. |
| Person tracking | `app/tracking/yolo_person.py`, `app/tracking/track.py` | Person-only YOLO tracking, persistent tracker IDs, bounding boxes, ground points. |
| Temporal fire rule | `app/reasoning/temporal.py` | Bounded boolean observation window and positive-count confirmation. |
| Zone geometry and entry | `app/zones/polygon.py`, `app/ui/workers/vision_worker.py` | Point-in-polygon check and per-track transition reasoning. |
| Dwell / loitering | `app/reasoning/dwell.py` | Per-track continuous in-zone elapsed time and one trigger per dwell session. |
| Face detector and embedding | `app/recognition/insightface_embedder.py` | InsightFace `buffalo_l` analysis, normalized embeddings, face boxes. |
| Face-to-track association | `app/recognition/association.py` | Unique face-center containment rule. |
| Saved face gallery | `app/recognition/registry.py` | NPZ persistence, sample append, cosine-similarity matching, match states. |
| Runtime identity state | `app/recognition/identity_memory.py` | Strong-evidence quorum, per-track name retention and expiry. |
| Desktop orchestration | `app/ui/main_window.py`, `app/ui/workers/vision_worker.py` | Settings, asynchronous requests, UI state, result flow. |
| Timestamped notifications | `app/ui/vision_types.py`, `app/ui/workers/vision_worker.py` | Trigger events with source-time offsets and optional recording paths. |
| Live registration dialog | `app/ui/face_registration_dialog.py` | Circular preview, five user prompts, explicit capture action. |
| Research evidence | `docs/research/experiment_log.md` | Dated user reports, implementation checks, proposed protocols, and results. |

## 4. Implemented methods and their boundaries

### 4.1 Fire and smoke

The fire/smoke detector is an Ultralytics YOLO model loaded from `models/firedetect-11s.pt`. Model weights are not included in Git, and their provenance/training details are not specified in the repository. The UI applies a fire/smoke confidence setting after detector inference; the current UI default is 0.40.

The worker initializes both YOLO components with a permissive internal inference confidence of 0.05, then applies the UI’s fire/person confidence settings to returned detections/tracks. The internal value is an inference pre-filter, not the operator-facing acceptance threshold. Record both values if a future run changes either.

Every accepted detection is drawn as a box. A frame contributes a positive observation to `TemporalPersistence` only when at least one accepted detection has class name `fire`. The temporal rule uses a 10-observation window and requires 7 positive observations. Confirmation is not possible until the window is full. The worker emits `FIRE` and `Fire cleared` events when the confirmed state changes. Fire evidence shown in the UI distinguishes warming, possible, and confirmed states.

Smoke boxes are drawn, but smoke is not counted as a positive fire observation and does not use a corresponding 7-of-10 smoke rule. The notification list emits `SMOKE` and “Smoke no longer detected” events on edges of the confidence-filtered smoke detections. These are raw detection-state notices, not temporal smoke confirmation; temporal persistence remains fire-only.

### 4.2 Person tracking and track history

The person tracker loads `models/yolo26s.pt`, filters to the person class, and calls Ultralytics tracking with `persist=True` and the `bytetrack.yaml` tracker configuration. The current UI person-confidence default is 0.35. Track history smooths ground-point movement and supports path display. Track IDs are tracker outputs; they are not permanent real-world identities and may be lost or reused.

The application’s latest-frame scheduling means that tracking/inference runs on submitted frames, not necessarily every camera frame. Formal frame-rate and ID-stability behavior have not been evaluated.

### 4.3 Restricted zones and intrusion transitions

The UI stores a polygon in normalized coordinates. The worker converts it to frame pixels and evaluates each tracked person’s ground point. For intrusion monitoring, it remembers each track’s prior inside/outside state and emits an `INTRUSION` event only when the observed state changes from outside to inside. It emits `EXIT` on inside-to-outside. Initial observation inside the polygon is not, by itself, an outside-to-inside transition.

The intrusion path is shown only after that track has entered through the observed transition. This is a rule-based state distinction, not a claim of intent or unauthorized behavior. A missing track, imperfect ground point, or tracker-ID change can affect whether the transition is observed. The current UI supports one polygon for the active scene.

`VisionEvent` carries the source-time offset and recording path when a request comes from a video. The event panel displays that offset to milliseconds and clicking a row pauses playback and seeks to the triggering frame. Live events use the frame's local wall-clock timestamp and do not carry a seekable recording position. The panel resets when the active source changes so old events cannot be applied to a different recording. Because the latest-frame worker may skip frames, the timestamp is the processed frame where the rule fired, not guaranteed ground-truth onset time.

### 4.4 Dwell / loitering

`DwellTracker` records a source-time entry value per track while its ground point remains inside the active zone. It emits the `LOITERING` event once when elapsed time reaches the configured threshold, then continues to report loitering for that continuous visit. Leaving the zone resets that track’s dwell state. The current desktop default is 15 seconds, with a UI range from 1 to 3600 seconds.

Dwell is distinct from intrusion: a track can dwell in-zone without an observed entry transition, and intrusion can occur without reaching the dwell threshold. Dwell uses tracked IDs and currently has no dedicated missing-track grace/expiry policy. Do not infer robustness to occlusion or track reassignment from the unit tests.

### 4.5 Face embedding and saved gallery

The face pipeline uses InsightFace `FaceAnalysis` with model pack `buffalo_l`, ONNX Runtime `CPUExecutionProvider`, and detector size 640×640. Face embeddings are converted to `float32`; InsightFace supplies normalized embeddings, and the registry normalizes vectors before storage/matching. The gallery file is `data/face_registry.npz`, containing identity strings and embedding arrays, loaded with `allow_pickle=False`.

For each query, `FaceRegistry` computes a dot product against every stored embedding and gives each identity its maximum sample similarity. Because vectors are normalized, the dot product is cosine similarity. The score is not a probability. The registry reports the best and second-best identity/score and a margin when two identities exist.

Current match-state boundaries:

| Best similarity | Registry state | Meaning in current code |
|---:|---|---|
| `< 0.35` | `NO_MATCH` | No candidate identity is returned. |
| `0.35` to `< 0.50` | `CANDIDATE` | A candidate is available for diagnostics; it does not count toward track confirmation. |
| `>= 0.50` | `STRONG_MATCH` | Strong evidence supplied to track identity memory. |

The margin is informational only; there is no margin threshold or margin-based acceptance gate. The two cutoffs are provisional and must be interpreted in light of the small reported evaluation in the experiment log.

### 4.6 Face-to-person association

The current association baseline assigns a face to a person track only if the face-box center lies inside exactly one person box and that person box contains exactly one face center. Ambiguous, uncontained, or multiply occupied cases are omitted rather than forced to an identity. This is the **implemented baseline**. Highest IoU and nearest-person alternatives are design candidates in the experiment log, not implemented or empirically compared conclusions.

### 4.7 Track identity memory

The saved gallery and runtime identity memory solve different problems:

- **NPZ gallery:** long-lived identity names and face embedding samples used for similarity queries.
- **Track identity memory:** in-memory state associating a current tracker ID with accumulated strong observations and, after confirmation, a retained display name.

Only `STRONG_MATCH` results enter the memory evidence. Defaults are 3 observations for one identity within a rolling 2.0-source-second evidence window. Once confirmed, the name is retained while that track remains active, including frames with no detected face. Pending evidence expires outside the window. A missing track expires after 2.0 source seconds; a later track reusing that ID starts without the old identity. One conflicting strong observation does not replace a confirmed name. Conflicting identities that both reach quorum leave an unconfirmed track unknown. The UI exposes a confirmation count from 1 to 10; changing the count resets the runtime analysis state.

The notification list emits one `IDENTITY` event when a track first reaches its configured strong-match quorum. Unconfirmed candidate or strong-match progress is not added as a separate event on every frame.

These are initial, configurable rules. The 3/2-second settings are not derived from the fire 7-of-10 rule and have not been validated as optimal. User-reported count experiments and qualitative outcomes are in the experiment log.

### 4.8 Face registration

Two user-facing registration methods are present:

1. **Image registration:** choose one or more JPG/JPEG/PNG files, each expected to contain exactly one detectable face. All embeddings are computed before the registry is changed. If processing fails, the existing gallery remains intact. Successful batches append to the NPZ file.
2. **Guided live registration:** the app uses Webcam 0 and presents a circular crop with five prompts: front, slight left, slight right, chin up, and chin down. The user manually captures a frontal reference. For the remaining views, the worker estimates normalized horizontal and vertical nose-landmark ratios relative to that reference and captures after three consecutive checks meet the target. A cumulative circular ring shows completed views and current three-check progress; a translucent head guide and directional arrow illustrate the requested movement. A **Capture manually** override is available. The worker accepts only frames with exactly one detected face. Five embeddings are held temporarily in memory, then appended to the gallery using a temporary file and atomic replacement. Captured frames are not stored as photo files.

Names are matched case-insensitively when adding samples; the spelling of the first stored identity is retained. The pose values are coarse 2D landmark ratios, not calibrated head-pose angles. Their initial cutoffs are provisional and can be affected by camera viewpoint and landmark noise. The user reported that the pose-registration workflow worked in a camera trial on 2026-11-14, but device, resolution, lighting, per-pose outcomes, and capture errors were not recorded; this is not a validation study. No measured benefit of five views has been established. The gallery is biometric data and remains local/ignored by Git; it should not be committed or treated as a public sample artifact.

The **View Registered People** dialog lists identities and sample counts and supports adding or removing samples by person. Removal requires explicit confirmation, matches the name case-insensitively, and removes every saved embedding for that identity using the same temporary-file/atomic-replacement pattern as registration. The original source image files are not changed. Removing the final identity writes a valid empty NPZ gallery, which loads as an empty registry and returns `NO_MATCH` for queries. The worker resets track identity memory after a successful removal so it does not keep displaying a name removed from the gallery.

## 5. Current configuration snapshot

These values describe commit `a477155` plus the timestamped notification changes recorded on 2026-11-14. UI settings may change during a run; record actual values for every experiment.

| Setting | Current default / rule | Evidence status |
|---|---|---|
| Fire/smoke UI confidence | 0.40 | Implemented default; not a validated optimum. |
| Person UI confidence | 0.35 | Implemented default; not a validated optimum. |
| YOLO worker inference pre-filter | 0.05 | Internal model threshold before the UI-level filters; not the displayed operating threshold. |
| Fire temporal window | 10 observations | Implemented. |
| Positive fire observations to confirm | 7 of 10 after window fills | Implemented; performance not benchmarked. |
| Dwell threshold | 15 seconds | UI default; operator-configurable. |
| Face candidate floor | 0.35 cosine similarity | Provisional, small reported pair evaluation. |
| Face strong-match cutoff | 0.50 cosine similarity | Provisional, small reported pair evaluation. |
| Strong observations for track identity | 3 | UI default, range 1–10; initial testable policy. |
| Identity evidence window | 2.0 source seconds | Implemented; not calibrated. |
| Missing-track identity TTL | 2.0 source seconds | Implemented; not calibrated. |
| Live yaw pose cue | nose horizontal ratio change of 0.12 inter-eye spans | Initial heuristic; not a pose angle or camera-validated cutoff. |
| Live pitch pose cue | nose vertical ratio change of 0.08 eye-to-mouth spans | Initial heuristic; not a pose angle or camera-validated cutoff. |
| Live pose stability | 3 consecutive checks; timer interval 250 ms | Implemented; actual interval may be longer while inference is busy. |
| Live frontal reference | manually captured first sample | Required before the four relative auto-capture prompts. |
| Fire detector model path | `models/firedetect-11s.pt` | Local weight file excluded from repository. |
| Person tracker model path | `models/yolo26s.pt` | Local weight file excluded from repository. |
| Face model/runtime | `buffalo_l`, CPU ONNX Runtime, 640×640 | Implemented configuration. |

## 6. Development history and decisions

The source history recorded here includes the project milestones through `875244d` and later changes. At the start of the timestamped-notifications work, `origin/main` pointed to `a477155`. Use `git rev-parse HEAD` for the current exact revision. The major repository milestones are:

| Date | Revision / milestone | Project state added or changed |
|---|---|---|
| 2026-07-12 | `2516631` project foundation | Repository initialized. |
| 2026-08-29 | `2e5564a`, `a8aeab1` | Initial live fire/smoke baseline and project naming/README. |
| 2026-09-20 | `aa60b93`, `21c9678`, `cfc7024`; tag `v0.2.0` | Temporal fire reasoning, person tracking/intrusion, and desktop UI. |
| 2026-10-02 | `1a005cd` | Zone dwell/loitering implementation. |
| 2026-10-25 | `9471de2` | Face recognition gallery baseline and face-to-track pipeline foundations. |
| 2026-10-28 | `5ac6680` | Face registration, persistent NPZ integration, and track identity memory. |
| 2026-10-31 | `875244d` | Registered-people gallery manager and guided live face registration. |
| 2026-11-13 | Pose-guided registration | Relative landmark pose checks and automatic capture added for four non-frontal prompts. |
| 2026-11-14 | `eac7919` | Progress ring, head guide, directional arrow, clearer registration instructions, and the research-context record were pushed; user reported the pose workflow worked in a camera trial. |
| 2026-11-14 | `a477155` | Registered-person removal with confirmation, case-insensitive sample deletion, atomic NPZ persistence, and empty-gallery support. |
| 2026-11-14 | Local notifications feature | Timestamped fire, smoke-edge, intrusion, exit, loitering, and identity events added; recording events seek to their trigger frame. |

The release tag remains `v0.2.0`; later features are unreleased development at this snapshot. Commit dates and subjects document repository history, not necessarily the exact date a user first tested or conceived each feature.

Key implementation decisions and their current rationale/boundaries:

- **Separate modules and domain values:** keep camera, detection, tracking, reasoning, recognition, zones, and UI responsibilities inspectable and testable.
- **Latest-frame processing:** prioritize responsiveness and current imagery over queueing every frame; the tradeoff is skipped inference observations under load.
- **Fire-only persistence:** the confirmed event state is currently based on a 7-of-10 fire observation rule; smoke remains a displayed detection.
- **Explicit intrusion transitions:** generate zone entry only for an observed outside-to-inside transition, and show the associated approach path only after entry.
- **Independent dwell state:** measure continuous in-zone time separately from crossing events.
- **Conservative face association:** leave ambiguous geometric cases unknown instead of forcing a face to a track.
- **Separate face match states from track confirmation:** a `CANDIDATE` similarity result is not strong identity evidence; only strong matches count toward the per-track quorum.
- **Local NPZ registry:** persist normalized embeddings and identity keys, not selected/captured photos. Use atomic replacement so a failed write does not partially replace the gallery.
- **Provisional pose automation:** compare 2D landmark ratios against a manually captured frontal reference; expose a manual override and keep cutoffs provisional until camera checks or a controlled study support them.
- **Chapter-by-chapter teaching and checks:** development discussions have used focused changes, explanations of data flow, tests after implementation, and a separate experiment log so that engineering progress is not mistaken for experimental evidence.

## 7. Evidence inventory and claim boundaries

The detailed dated records are in [experiment_log.md](experiment_log.md). The key evidence that can be carried into a paper is summarized here so it is not mistaken for stronger evidence than it is.

### 7.1 User-reported face-similarity evaluation

The handoff records a 2026-10-25 comparison using `buffalo_l`/CPU embeddings from 16 images belonging to four anonymized participants (A: 6, B: 4, C: 3, D: 3). It compared every same-identity pair and every cross-identity pair, yielding 27 genuine pairs and 93 impostor pairs. The reported genuine scores ranged 0.5939–0.8645 (mean 0.7212); impostor scores ranged -0.1264–0.2052 (mean 0.0358). The reported threshold sweep gave zero FAR and zero FRR from 0.30 through 0.55; at 0.20, FAR was 1.08%; at 0.60, FRR was 7.41%.

This was a small user-reported run recovered from the handoff, not rerun at commit `875244d` or the 2026-11-13 local working tree. The raw per-pair scores, image-level split, confidence intervals, and original run artifacts are not present in the repository. Pair scores reuse images, so pair counts are not independent sample counts. Treat 0.35/0.50 as provisional development cutoffs inside the observed gap, not generalizable error-rate results. Full details and the explanation for the two different cutoff roles are in the experiment log.

### 7.2 Other face observations

- The handoff reports an earlier three-pair anecdotal check on 2026-10-25 (one genuine pair at 0.6152; two impostor pairs at 0.1031 and 0.0887). This is not a rate estimate.
- A later still-image pipeline check reported one registered query at similarity 0.7146 and one unregistered query at 0.2095. It used a synthetic full-image person box; it validates plumbing for those images, not real YOLO association or video accuracy.
- A synthetic identity-memory replay reused one still-image embedding at timestamps 0.0, 0.5, and 1.0 seconds to verify quorum, plus synthetic track presence/absence. It validates state-flow behavior, not live video.
- The user-reported count=1, count=3, and count=5 observations in the experiment log lack clip IDs and complete raw frame outputs. The count=5 notes describe confirmation progress and retention across a short face disappearance; they are qualitative pilot notes, not a controlled study.
- The guided live-registration implementation snapshot reported 59 passing tests. The 2026-11-13 local pose-guidance update adds tests for normalized landmarks, direction cutoffs, worker output, and three-check auto-advance; its full-suite result is recorded in the experiment log. These are software tests, not recognition-accuracy or live-camera results.

### 7.3 Fire/model and software checks

The experiment log records a real fire-model inference regression check on one repository sample image with two fire detections (confidences 0.863 and 0.838). This is not a dataset-level fire evaluation. The repository tests cover units such as temporal logic, dwell rules, face association, registry persistence, and identity memory. An optional real-model integration test is dependent on local model weights and sample media.

At the 2026-10-31 implementation snapshot, the recorded software checks were 59 passing tests, Ruff lint/format checks, a Black-engine formatting check for the changed Python files, `git diff --check`, and an offscreen Qt dialog smoke check. The offscreen check used synthetic frames; it did not exercise a physical webcam or InsightFace live inference. The 2026-11-13 pose-guidance check uses mocked landmarks and synthetic frames; it also does not establish performance on a physical webcam. The registered-person removal update brought the suite to 74 passing tests. The timestamped-notifications update brings it to 81 and includes synthetic event and seek checks; neither update establishes detector accuracy or live operational performance. See the dated log entries for scope and evidence limits.

### 7.4 Claims currently supportable

The repository can support **implementation claims** such as:

- a modular desktop pipeline combines the listed detectors and deterministic reasoning components;
- fire detections are temporally confirmed by the specified 7-of-10 rule;
- intrusion events depend on observed outside-to-inside tracked transitions;
- dwell is measured per track while continuously inside a polygon;
- face association uses unique face-center containment;
- the registry compares normalized face embeddings using cosine similarity and exposes candidate/strong states;
- track identity memory requires configurable strong evidence and retains names with explicit source-time rules;
- live registration presents a frontal reference and four prompted views; three consecutive heuristic landmark checks trigger capture, with a manual override, and embeddings are stored in the local NPZ gallery.

The current evidence does **not** support general claims about detector accuracy, safety, FAR/FRR on unseen people, person-tracking reliability, association superiority, identity-memory accuracy, improvement from five-view registration, real-time frame rate, CPU/memory load, performance under lighting/occlusion/camera changes, or generalization to other cameras and populations.

## 8. Reproducibility and evaluation plan

### 8.1 Reproduce application and software checks

The README is the user-facing setup guide. The current package metadata requires Python 3.13 or newer and specifies Ultralytics, InsightFace, ONNX Runtime, NumPy, OpenCV, and PyQt5 dependencies. Model weights are not checked into Git. The expected local model paths are:

```text
models/firedetect-11s.pt
models/yolo26s.pt
```

The current automated software checks are run from the repository root:

```powershell
python -m pytest -q
ruff check .
black --check <changed-python-files>
```

The known face-embedding evaluation script is:

```powershell
python scripts/evaluate_face_embeddings.py <dataset-root>
```

It expects subdirectories by identity with JPG/JPEG/PNG files and prints all within-identity genuine pair scores, all cross-identity impostor pair scores, summaries, and a threshold sweep. It does not currently write a structured result file automatically. Preserve stdout/raw outputs and a data manifest for future runs; the original 2026-10-25 raw scores are not preserved here.

For a research run, also record: full commit hash; model hashes and model provenance; Python/dependency versions; operating system and hardware; data source and consent/handling; camera/video identifiers; identity and image counts; train/validation/test boundaries; exact settings; source-time interval; skipped/failed frames; raw outputs; and the date/operator. Keep personally identifiable media and biometric embeddings private and outside Git.

### 8.2 Face-similarity evaluation needed next

The recovered pairwise sweep is exploratory. A stronger study should separate identities or capture sessions between enrollment and query, specify the unit of analysis, include varied camera/lighting/pose/quality conditions, and report counts and uncertainty. Threshold selection should use a development set and be evaluated on held-out data. Do not tune and report on the same comparisons as if they were independent validation.

### 8.3 Video and identity-memory evaluation needed next

Use annotated clips with person-track/identity ground truth and timestamps. Freeze the registry, detector/model versions, association rule, face thresholds, video segment, and playback conditions while varying one policy at a time. Candidate identity-memory count comparisons already documented in the experiment log are 1, 3, and 5 strong observations. Define before analysis:

- eligible person-track episodes and what counts as visible face evidence;
- correct/incorrect named assignments and the denominator for precision;
- coverage denominator (frames, seconds, or track episodes);
- confirmation delay from first eligible strong match;
- retention after face disappearance while the same track remains active;
- track-loss, ID reuse, ambiguous association, and conflicting-identity handling;
- repeated-trial/clip strategy and uncertainty reporting.

For fire/smoke and intrusion, use labeled videos and report class-specific precision/recall or event-level measures, false alarms per duration, missed-event counts, event delay, and source/hardware context. For system behavior, profile end-to-end latency, inference FPS, skipped frames, CPU/GPU/memory, and camera disconnections. These are proposed measurements, not results.

## 9. Limitations and open work

Known implementation/evaluation limits in the current working tree include:

- no dataset-level fire/smoke or person-detector benchmark in the repository;
- no validated class-specific operating threshold for fire, smoke, or person detection;
- one restricted polygon and one primary webcam source (Webcam 0) or recorded video;
- dwell state is keyed to tracker IDs and lacks dedicated missing-track grace handling;
- no controlled evaluation of geometric face-to-person association alternatives;
- no live annotated-video evaluation of face accuracy or identity-memory error rates;
- provisional face cutoffs and identity-memory timings;
- guided registration pose cutoffs use uncalibrated 2D landmark ratios; no physical-camera validation is recorded yet;
- no quantitative evidence that multiple registration views improve recognition;
- no persisted application-settings/zone workflow established as a research capability;
- model-weight provenance and hashes are not included in the repository;
- no external literature review or formal bibliography is maintained in these project files.

The README roadmap and limitations remain useful for product planning. They should not be presented as completed research contributions.

## 10. Privacy and data handling

Face photographs, videos, raw identity-labeled comparisons, and the NPZ biometric gallery are local/private artifacts and are excluded from Git. The `.gitignore` excludes `data/*.npz` and designated local face/video/data paths. The current commits contain source, tests, README, and research documentation; they do not contain the local face registry.

The research log anonymizes participants as A–D, but anonymization in a text file does not make the underlying face media anonymous. Store consent records, media, manifests, and raw results according to the project owner’s applicable privacy and ethics process. Do not commit them or add identifiable details to the public experiment log.

## 11. Paper-writing workflow and source map

Use these two research files as the project-specific writing base:

1. **This context file:** project summary, system architecture, algorithm behavior, parameters, implementation decisions, current claim boundaries, reproducibility notes, and gaps.
2. **[The experiment log](experiment_log.md):** dated results, user-reported observations, threshold analysis, comparison proposals, anomalies, and run-record template.

Consult the current source when a method detail has changed after snapshot `a477155`. The README remains the setup and operator guide. `docs/architecture.md` is currently only a short environment decision note, so it is not a replacement for this context file.

Suggested paper organization (a proposal, not an adopted paper outline):

1. Problem motivation and scope
2. Related work (requires a separate literature search and citations)
3. System architecture and data flow
4. Methods: fire persistence, person tracking/zone rules, face matching/association, identity memory
5. Experimental setup and data governance
6. Results, with implementation checks separated from measured system performance
7. Limitations, privacy, and threats to validity
8. Conclusion and next evaluation steps

Before converting any log entry into a result claim, preserve its evidence label: **source-inspected**, **reported**, **observed in a run**, or **proposed**. Use the run record template in the log for every new comparison. The current project files preserve the project’s available implementation context and reported evidence; scholarly citations for YOLO, ByteTrack, InsightFace/ArcFace, temporal reasoning, face-recognition evaluation, and privacy/ethics must still be selected from authoritative publications and cited in the paper.

## 12. Source-of-truth order

For future work, resolve contradictions in this order:

1. Current source code at the paper’s named commit, for implemented behavior.
2. Dated entries in `experiment_log.md`, for what was actually reported or run and its evidence status.
3. This context document, for project synthesis and paper-writing boundaries.
4. README, for user-facing setup/capabilities.
5. Historical handoff material, only where it is explicitly identified as recovered/user-reported evidence and not superseded by current code.

Always name the commit used for a paper’s method description. Do not silently treat planned features, user recollections, local-only media, and tested code as interchangeable evidence.
