# SmartCrop Studio

**AI-powered portrait standardization — in your browser.**

SmartCrop Studio turns inconsistent human photos into perfectly aligned,
uniform, professional portraits at scale. Drop in a batch, pick an output size,
click **Generate** — the composition engine levels the eyes, centers the face,
and normalizes scale so every portrait shares the same eye level and face size.
Ready for ID cards, banners, team grids, passport forms, school/HR databases.

Everything runs **100% in the browser**. There is no backend and no upload —
your photos never leave your device.

---

## Run it

It's a single self-contained file. Put `smartcrop.html` in a folder and serve it:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/smartcrop.html
```

> An internet connection is needed on first load: the AI engine
> (MediaPipe Tasks-Vision + models) is fetched from a CDN. When ready, the
> sidebar shows **“AI engine ready · GPU”** and **Generate** activates.

(The repo also keeps the app split as `index.html` + `styles.css` + `app.js`;
`smartcrop.html` is the inlined build of those three.)

---

## How it works

| Layer | Implementation | Status |
|-------|----------------|--------|
| Face landmark detection | MediaPipe FaceLandmarker (478 pts + iris), **GPU delegate** | ✅ Live |
| Composition engine | Affine engine in **Web Workers** (OffscreenCanvas): eye-leveling, centering, scale normalization | ✅ Live |
| Background removal | MediaPipe selfie segmentation + **edge-aware matting** | ✅ Live |
| Quality checker | Per-image 0–100 score, gates expensive stages | ✅ Live |
| Export | PNG / JPG, smart single-vs-ZIP, team contact sheet | ✅ Live |
| Shoulder detection / framing | Pose model | 🔜 Roadmap |
| AI completion (shoulders/body) | Firefly / SD inpainting | 🔜 Roadmap |

### Composition rules

| Preset | Canvas | Eye line | Face width |
|--------|--------|----------|------------|
| 1:1 Square | 1080 × 1080 | 40% | 35% |
| 4:5 Portrait | 1080 × 1350 | 38% | 32% |
| Passport | 413 × 531 | 45% | 50% |

The engine detects the iris centers and face silhouette, then builds an affine
transform (rotate → scale → translate) mapping the eye line to the preset’s
target at the target face width. **Original pixels are only ever transformed —
never beautified, smoothed, or regenerated — so identity is preserved by
construction.**

---

## Performance

The batch engine is built for large jobs (school/HR batches of 500–5000):

- **Pipelined**: serial GPU detection overlaps a **Web Worker pool**
  (`navigator.hardwareConcurrency`) that composes + encodes in parallel.
- **Lazy intake**: files aren’t decoded until processed (adding 5000 files is
  instant); `createImageBitmap` decodes off-thread; source bitmaps are freed
  immediately.
- **Throttled UI** (250 ms) with in-place row updates and a virtualized queue —
  no full-DOM rebuild per image.
- **Live Performance panel**: throughput (img/s), processed/remaining, ETA,
  avg quality, JS heap, and per-stage benchmark (decode / detect /
  compose+encode / avg) plus accelerator + worker count.

Throughput is detection-bound and scales with your GPU. (On a real GPU,
detection runs ~10–30 ms/img; the worker pool hides encode behind it.)

---

## Edge-aware matting (background removal)

To avoid the jagged/blocky cutouts of naive segmentation, background removal
runs a proper refinement stack on the coarse mask:

`coarse mask → bilinear upsample → guided filter (luminance-guided, edge-aware)
→ smoothstep feather → bilinear upsample to full-res → straight alpha`

The guided filter snaps the alpha boundary onto real image edges (hair,
shoulders) and keeps soft transitions; RGB is never altered, so hair / skin /
clothing texture is preserved.

---

## Backgrounds, formats & filenames

- **Background modes**: Keep original · Solid white · Custom color · Transparent.
- **Transparent forces PNG** (JPEG can’t hold alpha) with a visible warning;
  no white/gray is ever composited behind a transparent export.
- **Filenames are preserved exactly** — no `_aligned` / `_processed` /
  `_final` / `_smartcrop` suffixes — for single download, batch, and inside ZIP.
- **Smart export**: 1 processed image downloads directly; >1 auto-ZIPs (only the
  ZIP archive name may differ — names inside stay original). “Export as ZIP”
  can force an archive.

---

## Workflow / UI

One screen, no hidden menus:

1. **Upload** — drag & drop, file picker, or folder. **Generate Portraits**
   button sits right in the input panel.
2. **Generated Results** panel shows live thumbnails + status + quality score.
3. **Preview** with before/after compare slider; a **Download** bar appears
   directly under the preview when results are ready (label switches
   Image / ZIP by count).
4. **Exports** tab for batch ZIP and team contact sheet (4×5 / 5×5 / 10×10).

Dark cyan, Linear/Midjourney-inspired theme with a glassmorphism footer.

---

## Usage tips

- ⌘/Ctrl + Enter triggers Generate.
- Double-click the preview to download the selected portrait.
- “No face detected” images are marked failed and skipped; everything else
  is processed.

---

## Tech

Vanilla JS (ES modules), HTML5 Canvas / OffscreenCanvas, Web Workers,
MediaPipe Tasks-Vision, JSZip, Sora/Inter. No build step.

Design & developed by **Satyam** — Founder & Product Architect of SmartCrop Studio.
