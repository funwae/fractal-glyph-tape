# Multimodal Fractal Experience Tape (Text + Video)

This document specifies how to unify:

- Fractal Glyph Tape (FGT) for text,
- Fractal Video Tape (FVT) for video,
- glyph-drive-3d for 3D visualization,

into a **single multimodal "Fractal Experience Tape"**.

## 1. Concept

> Experiences (sessions, calls, coding sessions, etc.) consist of:
> - text (chat, transcripts, logs),
> - video (screen, camera, UI),
> - time.

We want:

- one address space where:
  - a text phrase and a video tile share the same `FractalAddress`,
  - agents can navigate both.

## 2. Address alignment

We reuse `FractalAddress` from `@glyph/common-address`.

- For an experience stream:
  - `world` = tenant / environment,
  - `region` = experience type (e.g. `"support-call-1234"`),
  - `tri_path` = spatial motif location (FGT-driven),
  - `depth` = detail level (text summary vs raw + visual detail),
  - `time_slice` = frame/time index.

### 2.1 Text (FGT)

- Each phrase family used in the transcript gets:
  - `FractalAddress` with appropriate `tri_path` and `depth`.
- Time-sensitivity:
  - `time_slice` increments as the session progresses.

### 2.2 Video (FVT)

- FVT already uses:
  - a quadtree / fractal tile path per frame.
- We define a **mapping**:
  - `FVTMapper.toFVTCoords(addr)` and the inverse.
- For an experience:
  - `tri_path` approximates global visual layout,
  - `tilePath` is a more fine-grained refinement.

## 3. Storage model

For each `FractalAddress`, we can attach:

- `text_payload` (glyph-coded + optional raw text),
- `video_payload` (FVT tile/segment or pointer),
- `meta` (actor, tags, timestamps).

Implementation detail:

- Keep FGT and FVT storage formats as-is.
- Overlay a **multimodal index**:

```json
{
  "address": "earthcloud/support-call-1234#573@d1t102",
  "has_text": true,
  "has_video": true,
  "text_ref": "fgt://clusters/...",
  "video_ref": "fvt://streams/..."
}
```

## 4. Viewer integration

Use glyph-drive-3d as the primary **experience viewer**:

* Each Császár cell:

  * indicates content types (text/video/both).
* On click:

  * show:

    * relevant snippet of transcript,
    * video frame/segment from FVT.

Optionally:

* Show FGT fractal map (2D) for text-only exploration.

## 5. Demo scenario

Pick a narrow domain:

* Example: **"Dev Session Demo"**:

  * Record a coding session (screen video + audio).
  * Generate transcript via ASR.
  * Build FGT for transcript; FVT for video.
  * Assign unified addresses per event.

Use a web UI that:

* Shows 3D drive,
* Lets user click around,
* Queries FGMS for:

  * text summary,
  * relevant video clip,
  * and answers questions using both.
