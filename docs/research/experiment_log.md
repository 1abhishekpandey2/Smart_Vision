# Smart Vision research and experiment log

This file records methods, comparisons, decisions, and observed evidence that may support a later research paper. It is separate from application behavior and must not be treated as evidence of results unless a dated run record contains actual inputs and outputs.

## Evidence rules

- Label each item as **reported**, **source-inspected**, **proposed**, or **observed in a run**.
- Keep proposed methods separate from implemented behavior.
- For an experiment, record the code revision, environment, data/video identifiers, settings, procedure, raw output location, and result. Do not replace raw output with a conclusion.
- Record unsuccessful and ambiguous runs as well as successful ones.
- Do not claim measured accuracy, latency, or robustness without an actual run and a defined denominator.
- Preserve thresholds and parameters as provisional until supported by evaluation.

## Project baseline carried into this log

- **Reported, 2026-10-27:** Chapter 8 completed; the face registry is persisted in `data/face_registry.npz`.
- **Source-inspected, local workspace, 2026-10-28:** `FaceRegistry.match` returns the best accepted identity, top-two diagnostics, margin, and a `MatchState`. Default similarity cutoffs are 0.35 and 0.50; they are provisional. Margin is diagnostic.
- **Source-inspected, local workspace, 2026-10-28:** the desktop worker runs face recognition when enabled, associates faces to uniquely containing person boxes, and confirms/retains identities by track using a provisional observation window and timeout.
- **Not established:** measured face-to-track association accuracy, identity-memory error rate, latency, or behavior on a controlled annotated video set.
- A user-reported qualitative video check found the identity confirmation and retention behavior worked; exact clip, frame outputs, and metrics were not recorded.

## Recovered face-similarity threshold experiment

**Status:** user-reported run from 2026-10-25, recovered from the handoff; not rerun in this workspace. The handoff provides aggregate results, not the raw per-pair score file.

### Method

- Model: InsightFace `buffalo_l` on CPU (`CPUExecutionProvider`), producing normalized 512-dimensional embeddings.
- Similarity: dot product of normalized vectors, equivalent to cosine similarity. It is a score, not a probability.
- Dataset: 16 images across four anonymized participants: A (6), B (4), C (3), and D (3). Participant A's six samples included blurred, far, front-clear, low-light, slight-left, and slight-right conditions.
- Pair protocol: compare every same-identity pair as genuine and every cross-identity pair as impostor. This produced 27 genuine pairs and 93 impostor pairs.
- Rates: `FAR = impostor scores >= threshold / all impostor pairs`; `FRR = genuine scores < threshold / all genuine pairs`; `TAR = 1 - FRR`.

### Observed aggregate scores

| Pair group | Count | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|
| Genuine / same identity | 27 | 0.5939 | 0.8645 | 0.7212 |
| Impostor / different identities | 93 | -0.1264 | 0.2052 | 0.0358 |

| Threshold | FAR | FRR | TAR |
|---:|---:|---:|---:|
| 0.20 | 1.08% | 0.00% | 100.00% |
| 0.30 | 0.00% | 0.00% | 100.00% |
| 0.35 | 0.00% | 0.00% | 100.00% |
| 0.40 | 0.00% | 0.00% | 100.00% |
| 0.45 | 0.00% | 0.00% | 100.00% |
| 0.50 | 0.00% | 0.00% | 100.00% |
| 0.55 | 0.00% | 0.00% | 100.00% |
| 0.60 | 0.00% | 7.41% | 92.59% |

### How the provisional thresholds were reached

The small evaluation showed a clean observed gap between its impostor maximum (0.2052) and genuine minimum (0.5939). At 0.20, one of 93 impostor pairs was accepted; from 0.30 through 0.55, the reported table showed no false accepts or false rejects; at 0.60, two of 27 genuine pairs were rejected. This supported choosing development cutoffs inside the observed gap rather than near either error boundary. The current registry uses 0.35 as the candidate floor and 0.50 as the strong-evidence cutoff. These values distinguish candidate from strong evidence within the gap; the recovered results do not establish that 0.35 and 0.50 are universal operating points or that 0.50 is a calibrated probability.

#### Why there are two cutoffs

| Cutoff | Role in the registry | How the evaluation supports it | What the cutoff does not mean |
|---:|---|---|---|
| 0.35 | Candidate floor: below it, return `NO_MATCH`; from 0.35 up to 0.50, return `CANDIDATE`. | The observed impostor maximum was 0.2052, and the sweep reported zero FAR at 0.30 and above through 0.55. Thus 0.35 sat above all impostor scores in this small evaluation while still below its genuine scores. | It is not the threshold used by Track Identity Memory to confirm a name. A candidate is diagnostic evidence, not an established identity. |
| 0.50 | Strong-evidence cutoff: at or above it, return `STRONG_MATCH`. | 0.50 was inside the reported zero-FAR/zero-FRR sweep band (0.30–0.55) and 0.0939 below the weakest observed genuine pair (0.5939). Track Identity Memory counts only this state. | It is not a probability of identity and does not guarantee a correct live match. |

The handoff says the clean gap motivated the provisional settings, but it does not preserve a separate contemporaneous explanation for choosing the exact pair 0.35 and 0.50 from among other values in the zero-error band. The comparison above explains their distinct operational roles from the current registry contract and the reported sweep. The numerical distances are calculated from the reported aggregate bounds; they are not a separate experiment or a claim of optimality.

**Limits:** only four identities and 16 images were used. Pair comparisons share images and are not independent observations. The reported zero error rates apply only to these pairs and this model/setup; they do not establish performance on unseen people, arbitrary CCTV quality, live video, face-to-track association, or identity memory. The handoff contains aggregate counts/rates but not raw per-pair scores, confidence intervals, or a held-out test protocol.

**Separate earlier comparison:** the handoff also reports one genuine image pair at 0.6152 and two impostor pairs at 0.1031 and 0.0887 on 2026-10-25. These three examples preceded the four-identity evaluation and are anecdotal pair checks, not the basis for a rate estimate.

**Margin note:** later gallery examples reported an unregistered-person query with top score 0.2095 and a registered-person query with top score 0.7146. The top-two margin was added as a diagnostic; no margin threshold was evaluated or used as an acceptance gate in the recovered version.

## Design comparison: face-to-person association

**Status:** proposed comparison, not an experiment. No measured results.

| Candidate rule | Basic operation | Strength | Main failure mode to measure |
|---|---|---|---|
| Face-center containment | Associate a face with the person box containing its center; accept only when exactly one box contains it. | Simple and easy to inspect; ambiguity can remain unknown. | Misses association when boxes are inaccurate or overlap; face center may fall outside a partial person box. |
| Highest face/person IoU | Choose the person box with greatest intersection-over-union with the face box. | Familiar geometric score. | Face boxes are much smaller than person boxes, so raw IoU may be low and poorly discriminate candidates. |
| Nearest person-box center | Choose the nearest person center under a distance cutoff. | Can associate when containment fails. | Center distance depends on person size and scene layout; cutoff needs calibration. |

**Working baseline, implemented on 2026-10-28:** `app/recognition/association.py` uses face-center containment. It returns an association only when exactly one person box contains a face center and exactly one face maps to that track. Zero/multiple containing person boxes and multiple faces mapped to one track are omitted as ambiguous. The desktop integration is described below; this baseline has not been measured. Compare alternatives on the same annotated clips before claiming one is better.

**Implementation update, 2026-10-28:** the desktop path has an opt-in “Face Recognition” control. When enabled with person tracking, `InsightFaceEmbedder.detect_faces(frame.image)` produces current-frame face boxes and embeddings; the worker applies the association helper and queries the saved NPZ registry for associated faces. Strong matches are accumulated by track; after the configured quorum, the identity is retained while the person track remains active. InsightFace and the gallery load lazily. The integration has not been measured on a controlled annotated video set.

### Implementation checks, 2026-10-28

- Environment: project `.venv`, Python 3.13.5.
- `python -m pytest -q`: **39 passed**. This includes unit tests for unique/ambiguous face-to-track association, face-box/embedding conversion using a fake analyzer, and worker label behavior using fake person/face components, plus the existing camera, YOLO fake-model, and reasoning tests.
- `python tests/detection/test_yolo_real.py`: completed real fire-model inference on the repository sample image and reported two fire detections (confidence 0.863 and 0.838). This is a fire-model regression check, not evidence about face recognition.
- Import smoke check for `InsightFaceEmbedder`, `VisionWorker`, and `MainWindow`: passed.
- Actual face-pipeline check using the saved NPZ gallery and `InsightFaceEmbedder.detect_faces`: a registered-person query produced one face, associated to a synthetic full-image track, and matched the registered identity as `strong_match` at 0.7146; an unregistered-person query produced one face, associated to the same synthetic track geometry, and remained `no_match` at 0.2095. InsightFace emitted its existing `face_align.estimate` deprecation warning.
- A local visual record of those still-image results exists but is excluded from Git because it contains identifiable face photos.
- Limits: no live face-recognition video run was performed. The tests do not measure CPU/frame-rate impact, face detection quality, association quality on CCTV, threshold error rates, or identity persistence after occlusion.
- The actual face-pipeline check used a synthetic person box covering each entire still image. It validates face detection, embedding, unique-containment plumbing, and gallery matching on these two queries; it does not validate association against YOLO person boxes or live video.

## Track Identity Memory implementation check

**Status:** source implementation and tests completed on 2026-10-28; configured values are provisional.

**User-reported video check, 2026-10-28:** the user reports that they checked the feature and “everything is working fine.” No clip identifier, environment, screenshots, per-frame outputs, or quantitative measurements were supplied, so this is a qualitative user-reported check and cannot yet support paper metrics.

- New module: `app/recognition/identity_memory.py`.
- Policy: only `STRONG_MATCH` identities enter per-track evidence. The current settings require 3 observations of one identity within 2.0 source-time seconds. When that quorum is reached, the name is retained while the track remains active, even on frames without a face. Pending evidence ages out after its 2.0-second window. A track missing for more than 2.0 seconds loses its memory; a returning reused ID starts unknown. Confirmed identity does not switch on one conflicting strong match. The worker resets memory when face recognition is disabled, person tracking is disabled, or runtime state is reset.
- These settings were selected as an initial testable policy, not learned from the face evaluation and not borrowed from fire persistence. Their latency, false-assignment behavior, and appropriate timeout still need comparison on annotated video.
- Verification: recognition unit suite **17 passed**; full `pytest -q` **47 passed** (Python 3.13.5); UI/worker imports passed; `git diff --check` passed.
- Iteration note: the first identity-memory test run had one failed expectation. The scenario expected an identity to remain unknown after it had already received three strong observations. The test was corrected to use mixed evidence that never reaches quorum; implementation behavior matched the stated first-quorum policy.
- Actual-model state-flow check: one registered-person embedding matched the saved registry at 0.7146. Replaying that same strong result at synthetic timestamps 0.0, 0.5, and 1.0 seconds produced unknown, unknown, then `{12: 'registered_identity'}`. With track 12 active but no face evidence at 1.5 seconds, memory retained the name. After the track was absent through 4.0 seconds, the same ID returned unknown at 4.1 seconds.
- Limits: the replay reused one still-image embedding at different synthetic times and a synthetic track ID; it is a state-flow check, not a video experiment. No live YOLO association, changing face evidence, occlusion footage, track-ID reuse footage, frame-rate impact, or operational error rate has been measured.

## Planned comparison: identity confirmation count

**Status:** proposed protocol; user-reported count=1 and count=3 pilots are recorded below.

**Experiment control added, 2026-10-28:** the Intelligence panel now exposes “Strong matches to confirm” (range 1–10, default 3), allowing the planned count comparison without editing code between runs. Changing the control routes through the existing analysis-configuration reset, so prior pending/confirmed identity state is cleared before the next run. This is a parameter control, not a finding.

**Control test result:** full `pytest -q` passed, **48 passed** (Python 3.13.5); recognition tests passed, **18 passed**; UI/worker import smoke check and `git diff --check` passed. An initial focused run caught a test-placement error (`NameError` for an embedder variable); the occlusion assertions were moved back to their intended test, after which the suite passed.

**Progress display added, 2026-10-28:** the worker now shows `Confirming n/N` alongside a current strong `Face:` match until the quorum is reached; then it shows `Identity:`. This exposes the actual count of processed strong observations and resolves second-level timestamp ambiguity. Tests for progress counts, confirmation, and worker labels pass; current full suite remains **48 passed**.

### Question

How does the number of strong face observations required to name a track affect incorrect assignments, correct-name coverage, and time to confirmation?

### Conditions and controls

- Compare `required_observations` values **1, 3, and 5**. The current implementation default is 3; 1 and 5 are comparison conditions, not recommendations.
- Keep the registry, face cutoffs (0.35 candidate / 0.50 strong), two-second evidence window, two-second missing-track timeout, face-center association rule, video clip, playback speed, and machine conditions fixed.
- Reset analysis state and restart from the same video position before each run. Record source-video timestamps, not wall-clock time.
- Use a segment where the same registered person is clearly visible long enough to produce face matches, then turns away or is occluded while the person track remains. Include a stranger segment if available.
- Capture the track ID and the timestamp of the first `Face:` strong-evidence label and first `Identity:` remembered label. Note whether the name remains while the face is absent and the same track ID remains active. Record any wrong name, unexpected reset, or missed confirmation.

### Measures

- Wrong named tracks / named track episodes.
- Correct-name coverage over eligible tracked-person time.
- Confirmation delay in source seconds from first strong face evidence to `Identity:` label.
- Retention duration after the last visible face observation while the track stays active.
- Whether the identity clears after the track has been missing longer than the configured timeout.

These measures need frame or episode denominators and known ground truth before paper claims. The first pass is a pilot to find practical behavior; it is not enough by itself to estimate general performance.

### Run sheet

| Count | Clip and segment | Ground-truth person | First strong timestamp | First identity timestamp | Correct? | Retained after face absent? | Notes / raw output |
|---:|---|---|---:|---:|---|---|---|
| 1 | User reports 0–1 s segment; clip ID not supplied | Not stated for this run | Within 0–1 s | Within 0–1 s | Not stated for this run | Not reported | User observed candidate, strong `Face`, then `Identity`; exact timestamps unavailable |
| 3 | User reports 0–1 s segment; clip ID not supplied | Not stated for this run | Within 0–1 s | Within 0–1 s | Not stated for this run | Not reported | User saw confirmation progress and reports identity confirmation at 3 observations within the first second; exact timestamps and clip ID unavailable |
| 5 | User-reported segment; clip ID not supplied | Registered identity; name not supplied | `Confirming 4/5` around 1.15 s; later segment began at `1/5` around 8 s and reached `3/5` around 9 s; user reports later reaching 5/5 around 9–10.5 s | Exact `Identity:` transition not reported | Yes, user confirmed correct identity | Yes, user reports the confirmed name was retained after the face disappeared | Candidate frames did not count. Each strong observation remains in the rolling evidence window until it is more than 2 source-time seconds old. Same `Person #` stayed through the short face gap. |

## In-app face registration

**Status:** UI and worker workflow implemented on 2026-10-28; actual user photo registration not yet performed through the visible app.

- The Sources panel includes an identity-name field and a multi-image selector for JPG/JPEG/PNG files.
- Each selected image uses `InsightFaceEmbedder.embed_image`, which rejects unreadable images, zero faces, and multiple faces. The selected batch is all-or-nothing: all embeddings are computed before the gallery is changed.
- The worker loads the current `data/face_registry.npz` when present, appends samples under the entered identity (case-insensitively; the first stored spelling is preserved), writes a temporary NPZ, then atomically replaces the target. On success it refreshes its cached gallery and clears track identity memory. The user’s current gallery was not changed during checks.
- Tests: full `pytest -q --basetemp=data/.pytest-run-full` passed, **51 passed** (Python 3.13.5); `ruff check` on changed UI/worker/registration-test files passed; offscreen `MainWindow` construction confirmed the registration widgets instantiate.
- Real sample check: using a pre-initialized InsightFace embedder, the worker registration method added a registered-person query under a test identity in a temporary copy of the NPZ; the output reported one image registered and the new identity key was present. Existing gallery samples remained in that copy.
- Limits / failed checks: the headless Qt worker-thread harness exited with code 1 during native model loading, without a Python exception; a separate direct model initialization and non-threaded registration check passed. Computer-use initialization also failed in this environment, so the actual file-dialog click path was not exercised. The normal app’s visible registration flow still needs the user’s first live check.

## Design comparison: when to remember a matched identity

**Status:** user-reported pilots for counts 1, 3, and 5, plus a proposed controlled evaluation. No quantitative results.

| Candidate rule | Basic operation | Strength | Main failure mode to measure |
|---|---|---|---|
| One strong observation | Store identity after one `STRONG_MATCH`. | Lowest delay and simplest state. | A single false strong match can name the wrong track. |
| K consecutive strong observations | Require an uninterrupted run of strong matches for the same identity. | Rejects isolated matches. | A missed/turned face breaks the run; behavior depends on frame rate. |
| M-of-N strong observations | Require a count of same-identity strong observations within a rolling window. | Tolerates intermittent face visibility. | Adds state and delay; the window and count require measurement. |

The fire rule of 7 positive frames in a 10-frame window is **not** a face-identity rule. Do not reuse those values by assumption. A fair comparison should vary confirmation policy while keeping clips, face association, registry, and thresholds fixed.

## Suggested evaluation measures (not yet measured)

- **Assignment precision:** correctly named track assignments divided by all named track assignments.
- **Assignment coverage:** frames or track episodes with a name divided by eligible frames or episodes; define the denominator before each run.
- **False assignment count:** track episodes assigned an incorrect identity.
- **Time to first correct name:** elapsed source time from first eligible visible face to a correct remembered name.
- **Persistence after face occlusion:** duration for which the correct name remains on the same continuing track after its face is no longer detected.
- **Recovery/conflict behavior:** outcomes when an identity changes, a tracker ID is reacquired/reused, or observations are ambiguous.

These measures need annotated video segments and a written protocol. Until those exist, there are no performance claims.

## Run record template

Copy this section for each actual comparison or test.

### YYYY-MM-DD — short experiment name

- **Status:** observed in a run
- **Question / hypothesis:**
- **Code revision and local changes:**
- **Environment:**
- **Data/video IDs and annotations:**
- **Method and settings:**
- **Procedure:**
- **Raw output path:**
- **Results:**
- **Failures / anomalies:**
- **Interpretation and limits:**
- **Next controlled comparison:**

## 2026-10-28 — Repository quality and clone checks

**Status:** observed code-quality and test run; not a vision-performance experiment.

- Base revision: `9471de2` plus the uncommitted changes reviewed for this update.
- Environment: Windows, project `.venv`, Python 3.13.5.
- Full test suite: `python -m pytest -q --basetemp=.pytest_cache/smart-vision-release` — **53 passed**. Local YOLO weights and sample media were present for this run. The optional real-model test skips when either local asset is absent on a fresh clone.
- Ruff: `python -m ruff check .` — passed.
- Black: formatted the repository's 66 Python source, script, and test files; a Black-engine comparison then confirmed all 66 match the configured formatting.
- Privacy boundary for the Git update: identity-labeled raw comparisons, local sample/video data, the face-result image, and the NPZ face gallery are excluded from the repository. Participant names in this log are anonymized.
- Limits: these checks establish code health for this checkout, not detection accuracy, operational reliability, or general face-recognition performance.
