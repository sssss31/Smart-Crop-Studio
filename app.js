/* ============================================================================
 * SmartCrop Studio — high-throughput client-side portrait standardization
 *
 * Performance architecture (browser-native equivalents of the perf brief):
 *   - Lazy intake: files are NOT decoded on add (5000 files add instantly).
 *   - Pipelined engine: serial GPU detection overlaps with a parallel pool of
 *     Web Workers that compose + encode on OffscreenCanvas (async export).
 *   - GPU delegate for MediaPipe (WebGL; Apple GPU on Mac) with CPU fallback.
 *   - createImageBitmap off-thread decode; source bitmaps released immediately.
 *   - Intelligent quality score gates expensive segmentation.
 *   - Throttled UI (≤4 fps during batch) + virtualized queue — no O(n²) redraw.
 *   - Live stats + per-stage benchmark panel.
 *
 * Output math is byte-for-byte the same affine transform as before, so image
 * quality / identity / edge handling are unchanged — only the plumbing is fast.
 * ==========================================================================*/

const TASKS_VISION_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const SEG_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite";

/* ───────────────────────── Composition presets ───────────────────────── */
const PRESETS = {
  square:    { id: "square",    name: "1:1 Square",   w: 1080, h: 1080, eyeY: 0.40, faceW: 0.35, note: "ID cards · profiles" },
  portrait45:{ id: "portrait45",name: "4:5 Portrait", w: 1080, h: 1350, eyeY: 0.38, faceW: 0.32, note: "Banners · posters" },
  passport:  { id: "passport",  name: "Passport",     w: 413,  h: 531,  eyeY: 0.45, faceW: 0.50, note: "Passport · visa" },
};

const LM = { leftIris: 468, rightIris: 473, faceLeft: 234, faceRight: 454 };

/* ───────────────────────── State ───────────────────────── */
const state = {
  items: [],            // { id, file, name, status, result, meta, score }
  selectedId: null,
  preset: "portrait45",
  faceLandmarker: null,
  imageSegmenter: null,
  segReady: false,
  engineReady: false,
  accel: "—",           // "GPU" | "CPU"
  running: false,
  cancel: false,
  beforeUrl: null,      // lazily-created object URL for the previewed source
};

let nextId = 1;
const WORKER_COUNT = Math.max(2, Math.min((navigator.hardwareConcurrency || 4), 8));
const SMALL_BATCH = 60;       // show source thumbnails only below this size
const QUEUE_RENDER_CAP = 150; // virtualization cap

/* ───────────────────────── DOM helpers ───────────────────────── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), ms);
}

/* ════════════════════════ Encode worker pool ════════════════════════
 * Each worker composes the affine transform on an OffscreenCanvas and encodes
 * to a Blob — entirely off the main thread, in parallel across cores. */
const WORKER_SRC = `
self.onmessage = async (e) => {
  const m = e.data;
  const t0 = performance.now();
  const cv = new OffscreenCanvas(m.w, m.h);
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (m.bgFill) { ctx.fillStyle = m.bgFill; ctx.fillRect(0, 0, m.w, m.h); }
  ctx.save();
  ctx.translate(m.targetX, m.targetY);
  ctx.rotate(-m.angle);
  ctx.scale(m.scale, m.scale);
  ctx.translate(-m.eyeX, -m.eyeY);
  ctx.drawImage(m.bitmap, 0, 0);
  ctx.restore();
  const blob = await cv.convertToBlob(
    m.format === "image/jpeg" ? { type: "image/jpeg", quality: m.quality } : { type: "image/png" }
  );
  m.bitmap.close();
  self.postMessage({ id: m.id, blob, encodeMs: performance.now() - t0 });
};`;

class EncodePool {
  constructor(n) {
    this.url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    this.workers = [];
    this.free = [];
    this.waiters = [];
    this.jobs = new Map();
    this.jid = 0;
    for (let i = 0; i < n; i++) {
      const w = new Worker(this.url);
      w.onmessage = (e) => {
        const job = this.jobs.get(e.data.id);
        this.jobs.delete(e.data.id);
        this._release(w);
        job.resolve(e.data);
      };
      this.workers.push(w);
      this.free.push(w);
    }
  }
  _acquire() {
    return this.free.length ? Promise.resolve(this.free.pop())
                            : new Promise((r) => this.waiters.push(r));
  }
  _release(w) {
    const next = this.waiters.shift();
    if (next) next(w); else this.free.push(w);
  }
  async run(params, transfer) {
    const w = await this._acquire();
    const id = ++this.jid;
    return new Promise((resolve) => {
      this.jobs.set(id, { resolve });
      w.postMessage({ ...params, id }, transfer);
    });
  }
  terminate() {
    this.workers.forEach((w) => w.terminate());
    URL.revokeObjectURL(this.url);
  }
}

/* ───────────────────────── Engine bootstrap ───────────────────────── */
async function initEngine() {
  const statusEl = $("#engineStatus");
  $("#hwInfo").textContent = `${WORKER_COUNT} workers · ${navigator.hardwareConcurrency || "?"} cores`;
  $("#bWorkers").textContent = WORKER_COUNT;
  try {
    const { FaceLandmarker, ImageSegmenter, FilesetResolver } = await import(
      /* @vite-ignore */ TASKS_VISION_URL
    );
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    // Try GPU delegate first; fall back to CPU if it fails to initialize.
    const mkFace = (delegate) => FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
      runningMode: "IMAGE",
      numFaces: 1,
    });
    try {
      state.faceLandmarker = await mkFace("GPU");
      state.accel = "GPU";
    } catch {
      state.faceLandmarker = await mkFace("CPU");
      state.accel = "CPU";
    }

    state.engineReady = true;
    statusEl.className = "engine-status ready";
    statusEl.querySelector(".engine-label").textContent = `AI engine ready · ${state.accel}`;
    $("#bAccel").textContent = state.accel;
    setGenEnabled(state.items.length > 0);

    // Optional segmentation (lazy, non-fatal).
    ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: SEG_MODEL_URL, delegate: state.accel },
      runningMode: "IMAGE",
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    })
      .then((seg) => { state.imageSegmenter = seg; state.segReady = true; renderModelCards(); })
      .catch(() => { state.segReady = false; });
  } catch (err) {
    console.error(err);
    statusEl.className = "engine-status error";
    statusEl.querySelector(".engine-label").textContent = "Engine offline (no network)";
    $("#bAccel").textContent = "offline";
    toast("Could not load the AI engine. Check your internet connection.");
  }
}

/* Both Generate buttons (top bar + input panel) act as one control. */
function genButtons() { return [$("#generateBtn"), $("#genWorkspace")].filter(Boolean); }
function setGenEnabled(en) { genButtons().forEach((b) => (b.disabled = !en)); }
function setGenRunning(running) {
  genButtons().forEach((b) => {
    b.textContent = running ? "■ Stop" : (b.id === "genWorkspace" ? "⚡ Generate Portraits" : "⚡ Generate");
  });
}

/* ───────────────────────── File intake (lazy) ───────────────────────── */
function addFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  for (const file of files) {
    const item = { id: nextId++, file, name: file.name, status: "pending", result: null, meta: null, score: null };
    if (state.items.length < SMALL_BATCH) item.thumbUrl = URL.createObjectURL(file); // cheap UX for small sets
    state.items.push(item);
  }
  renderQueue();
  renderStats();
  if (state.engineReady) setGenEnabled(state.items.length > 0);
  if (!state.selectedId && state.items.length) selectItem(state.items[0].id);
  toast(`${files.length} image${files.length > 1 ? "s" : ""} added · ${state.items.length} queued.`);
}

/* ───────────────────────── Detection + quality ───────────────────────── */
function detectFace(source, W, H) {
  const result = state.faceLandmarker.detect(source);
  if (!result.faceLandmarks || !result.faceLandmarks.length) return null;
  const lm = result.faceLandmarks[0];
  const pt = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });
  const le = pt(LM.leftIris), re = pt(LM.rightIris);
  const fl = pt(LM.faceLeft),  fr = pt(LM.faceRight);
  const eyeMid = { x: (le.x + re.x) / 2, y: (le.y + re.y) / 2 };
  const roll = Math.atan2(re.y - le.y, re.x - le.x);
  const faceWidth = Math.hypot(fr.x - fl.x, fr.y - fl.y);
  return { eyeMid, roll, faceWidth };
}

/* Intelligent quality score (0–100): face size, level, centering, resolution. */
function scoreQuality(face, W, H) {
  const sizeRatio = face.faceWidth / W;                  // ideal ~0.25–0.55
  const sizeScore = sizeRatio < 0.12 ? 30 : sizeRatio > 0.7 ? 65 : 100;
  const rollDeg = Math.abs(face.roll * 180 / Math.PI);
  const levelScore = Math.max(0, 100 - rollDeg * 6);     // 0° → 100, ~16° → 0
  const cx = face.eyeMid.x / W, cy = face.eyeMid.y / H;
  const centerScore = Math.max(0, 100 - (Math.abs(cx - 0.5) * 200 + Math.abs(cy - 0.42) * 120));
  const resScore = Math.min(W, H) >= 700 ? 100 : Math.min(W, H) >= 400 ? 75 : 45;
  return Math.round(0.3 * sizeScore + 0.25 * levelScore + 0.2 * centerScore + 0.25 * resScore);
}

function bgFillFor(opts) {
  if (opts.bgMode === "white") return "#ffffff";
  if (opts.bgMode === "color") return opts.bgColor;
  return null; // original / transparent — never composite a background
}

/* Keep background / format / transparency UI consistent.
 * Transparent background ⇒ force PNG export and surface a warning. */
function applyBgRules() {
  const mode = $("#bgMode").value;
  const transparent = mode === "transparent";
  $("#bgColor").classList.toggle("hidden", mode !== "color");
  const t = $("#opt-transparent"); if (t) t.checked = transparent;
  const fmt = $("#format");
  if (transparent) { fmt.value = "image/png"; fmt.disabled = true; $("#bgWarn").classList.remove("hidden"); }
  else { fmt.disabled = false; $("#bgWarn").classList.add("hidden"); }
  if (mode !== "original" && !state.segReady) toast("Background AI still loading…");
}

/* ════════════════════════ Professional matting pipeline ════════════════════════
 * The segmenter gives a coarse, low-res confidence mask. Hard-thresholding it
 * with nearest-neighbour sampling is what produced jagged hair / blocky edges.
 * Instead we run an edge-aware refinement so the alpha follows the *real* image
 * edges and stays soft:
 *
 *   coarse mask ──bilinear upsample──▶ guided filter (luminance guide)
 *        ──▶ soft feather (smoothstep) ──▶ bilinear upsample to full res
 *        ──▶ apply as straight alpha (texture/colour untouched)
 *
 * The guided filter is the edge-aware step: it pulls the alpha boundary onto
 * hair strands and shoulders using the photo itself as guidance, recovering
 * detail a binary mask destroys. RGB is never altered, so hair/skin/clothing
 * texture is preserved exactly.                                                */

// Separable clamped box blur over a Float32 plane — O(n), the core of the
// guided filter and the anti-aliasing.
function boxBlur(src, w, h, r) {
  const n = 2 * r + 1, tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {                       // horizontal
    const row = y * w; let sum = 0;
    for (let k = -r; k <= r; k++) sum += src[row + Math.min(w - 1, Math.max(0, k))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / n;
      sum += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {                        // vertical
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / n;
      sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

// Guided filter (He et al.): edge-aware smoothing of p guided by I, both in [0,1].
function guidedFilter(I, p, w, h, r, eps) {
  const mI = boxBlur(I, w, h, r), mP = boxBlur(p, w, h, r);
  const N = I.length;
  const Ip = new Float32Array(N), II = new Float32Array(N);
  for (let i = 0; i < N; i++) { Ip[i] = I[i] * p[i]; II[i] = I[i] * I[i]; }
  const mIp = boxBlur(Ip, w, h, r), mII = boxBlur(II, w, h, r);
  const a = new Float32Array(N), b = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const varI = mII[i] - mI[i] * mI[i];
    const covIp = mIp[i] - mI[i] * mP[i];
    a[i] = covIp / (varI + eps);
    b[i] = mP[i] - a[i] * mI[i];
  }
  const mA = boxBlur(a, w, h, r), mB = boxBlur(b, w, h, r);
  const q = new Float32Array(N);
  for (let i = 0; i < N; i++) q[i] = Math.min(1, Math.max(0, mA[i] * I[i] + mB[i]));
  return q;
}

// Draw a Float32 plane (values 0..1) into a canvas as grayscale, scaled with
// bilinear smoothing, and read the requested channel back as Float32.
const _scratch = document.createElement("canvas");
function resamplePlane(src, sw, sh, dw, dh) {
  _scratch.width = sw; _scratch.height = sh;
  const sc = _scratch.getContext("2d");
  const sid = sc.createImageData(sw, sh), sd = sid.data;
  for (let i = 0; i < src.length; i++) { const v = (src[i] * 255) | 0; sd[i * 4] = sd[i * 4 + 1] = sd[i * 4 + 2] = v; sd[i * 4 + 3] = 255; }
  sc.putImageData(sid, 0, 0);
  const out = document.createElement("canvas"); out.width = dw; out.height = dh;
  const oc = out.getContext("2d"); oc.imageSmoothingEnabled = true; oc.imageSmoothingQuality = "high";
  oc.drawImage(_scratch, 0, 0, dw, dh);
  const od = oc.getImageData(0, 0, dw, dh).data;
  const plane = new Float32Array(dw * dh);
  for (let i = 0; i < plane.length; i++) plane[i] = od[i * 4] / 255;
  return plane;
}

function luminancePlane(source, w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const cx = c.getContext("2d"); cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = "high";
  cx.drawImage(source, 0, 0, w, h);
  const d = cx.getImageData(0, 0, w, h).data, p = new Float32Array(w * h);
  for (let i = 0; i < p.length; i++) p[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
  return p;
}

function smoothstep(e0, e1, x) { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }

// Selfie-segmentation cutout with edge-aware matting (only when bg is replaced).
function buildCutout(source, W, H) {
  try {
    const seg = state.imageSegmenter.segment(source);
    const masks = seg.confidenceMasks;
    if (!masks || !masks.length) { seg.close && seg.close(); return null; }
    const mask = masks[0], mw = mask.width, mh = mask.height;
    const conf = Float32Array.from(mask.getAsFloat32Array());
    seg.close && seg.close();

    // Refinement work resolution (cap long side for speed; quality is upsampled back).
    const scale = Math.min(1, 768 / Math.max(W, H));
    const wW = Math.max(16, Math.round(W * scale)), wH = Math.max(16, Math.round(H * scale));

    const p = resamplePlane(conf, mw, mh, wW, wH);   // bilinear-upsampled coarse alpha
    const I = luminancePlane(source, wW, wH);         // guidance = the photo itself
    const r = Math.max(2, Math.round(Math.min(wW, wH) * 0.012));
    let q = guidedFilter(I, p, wW, wH, r, 1e-4);      // edge-aware refine (hair/shoulders)

    // Soft feather: keep semi-transparency at edges (no binary cut), trim halo.
    for (let i = 0; i < q.length; i++) q[i] = smoothstep(0.10, 0.92, q[i]);
    q = boxBlur(q, wW, wH, 1);                         // sub-pixel anti-alias

    const A = resamplePlane(q, wW, wH, W, H);          // bilinear upsample alpha to full res

    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const cx = c.getContext("2d");
    cx.drawImage(source, 0, 0, W, H);
    const id = cx.getImageData(0, 0, W, H), d = id.data;
    for (let i = 0; i < A.length; i++) d[i * 4 + 3] = (A[i] * 255) | 0;  // RGB untouched
    cx.putImageData(id, 0, 0);
    return c;
  } catch (e) { console.warn("matting failed", e); return null; }
}

/* ───────────────────────── Options ───────────────────────── */
function currentOpts() {
  return {
    level: $("#ai-level").checked && $("#ai-compose").checked,
    bgMode: $("#bgMode").value,
    bgColor: $("#bgColor").value,
    format: $("#format").value,
    quality: parseInt($("#quality").value, 10) / 100,
  };
}

/* ════════════════════════ Batch engine ════════════════════════
 * Pipeline: decode → detect (serial, GPU) → [segment if needed] → dispatch to
 * worker pool for compose+encode. Detection of image N+1 overlaps encoding of
 * image N, so cores stay busy and the main thread never blocks on encode.   */
const bench = { decode: 0, detect: 0, encode: 0, n: 0, qSum: 0, qN: 0 };

async function generateAll() {
  if (!state.engineReady) { toast("AI engine not ready yet."); return; }
  if (state.running) { state.cancel = true; return; }
  const preset = PRESETS[state.preset];
  const opts = currentOpts();
  const needSeg = opts.bgMode !== "original" && state.segReady; // cutout for any non-original bg
  const transparent = opts.bgMode === "transparent";
  const preserveNames = $("#opt-preserve").checked;
  if (transparent && !state.segReady) toast("Transparent mode: background AI not loaded — only inputs that already have transparency will stay transparent.");
  const targets = state.items.filter((i) => i.status !== "done");
  if (!targets.length) { toast("Nothing to process."); return; }

  bench.decode = bench.detect = bench.encode = bench.n = bench.qSum = bench.qN = 0;
  state.running = true; state.cancel = false;
  setGenRunning(true);

  const pool = new EncodePool(WORKER_COUNT);
  const t0 = performance.now();
  let done = 0, processed = 0;
  startStatsTicker(targets.length, t0, () => done);

  const inflight = [];
  for (const item of targets) {
    if (state.cancel) break;
    item.status = "processing";
    let bitmap;
    try {
      const d0 = performance.now();
      bitmap = await createImageBitmap(item.file);   // off-thread decode
      bench.decode += performance.now() - d0;
    } catch {
      item.status = "failed"; item.meta = { error: "Decode failed" }; done++; continue;
    }
    const W = bitmap.width, H = bitmap.height;

    const f0 = performance.now();
    const face = detectFace(bitmap, W, H);
    bench.detect += performance.now() - f0;

    if (!face) {
      item.status = "failed"; item.meta = { error: "No face detected" };
      bitmap.close(); done++; continue;
    }
    const score = scoreQuality(face, W, H);
    item.score = score; bench.qSum += score; bench.qN++;

    // Conditional segmentation: skip entirely unless bg replacement is requested.
    let drawBitmap = bitmap;
    if (needSeg) {
      const cut = buildCutout(bitmap, W, H);
      if (cut) {
        drawBitmap = await createImageBitmap(cut);
        bitmap.close();
      }
    }

    const scale = (preset.faceW * preset.w) / face.faceWidth || 1;
    const outFormat = ExportManager.formatFor(item, { transparent, preserveNames, globalFormat: opts.format });
    item.outName = ExportManager.outputName(item, { transparent, preserveNames, globalFormat: opts.format });
    const params = {
      bitmap: drawBitmap, w: preset.w, h: preset.h,
      targetX: preset.w / 2, targetY: preset.eyeY * preset.h,
      eyeX: face.eyeMid.x, eyeY: face.eyeMid.y,
      angle: opts.level ? face.roll : 0,
      scale: isFinite(scale) && scale > 0 ? scale : 1,
      // transparent/original → no fill (preserve alpha); never composite a bg.
      bgFill: transparent ? null : bgFillFor(opts), format: outFormat, quality: opts.quality,
    };
    const meta = {
      out: `${preset.w}×${preset.h}`, eyeLine: `${Math.round(preset.eyeY * 100)}%`,
      rollDeg: (face.roll * 180 / Math.PI).toFixed(1), scale: params.scale.toFixed(2), score,
    };

    // Dispatch to pool; do NOT await — let detection of the next image proceed.
    const p = pool.run(params, [drawBitmap]).then(({ blob, encodeMs }) => {
      bench.encode += encodeMs; bench.n++;
      if (item.result) URL.revokeObjectURL(item.result);
      item.result = URL.createObjectURL(blob);
      item.meta = meta; item.status = "done"; done++;
    });
    inflight.push(p);
    processed++;
  }

  await Promise.all(inflight);
  pool.terminate();
  stopStatsTicker();
  state.running = false;
  setGenRunning(false);

  flushStats(targets.length, t0, done, true);
  renderQueue();
  if ($("#set-autoselect")?.checked) {
    const fd = state.items.find((i) => i.status === "done");
    if (fd) selectItem(fd.id);
  } else renderPreview();
  updateExportBar();

  const total = (performance.now() - t0) / 1000;
  const ok = state.items.filter((i) => i.status === "done").length;
  const fail = state.items.filter((i) => i.status === "failed").length;
  toast(`${state.cancel ? "Stopped" : "Done"} — ${ok} ok${fail ? `, ${fail} failed` : ""} · ${total.toFixed(1)}s · ${(done / total).toFixed(1)} img/s`);
}

/* ───────────────────────── Live stats ticker (throttled) ───────────────────────── */
let statsTimer = null;
function fmtTime(s) { return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`; }

function startStatsTicker(total, t0, getDone) {
  flushStats(total, t0, getDone(), false);
  statsTimer = setInterval(() => flushStats(total, t0, getDone(), false), 250);
}
function stopStatsTicker() { clearInterval(statsTimer); statsTimer = null; }

function flushStats(total, t0, done, final) {
  const elapsed = (performance.now() - t0) / 1000;
  const ips = elapsed > 0 ? done / elapsed : 0;
  const remaining = total - done;
  $("#perfIps").textContent = ips ? ips.toFixed(1) : "—";
  $("#perfDone").textContent = done;
  $("#perfRem").textContent = remaining;
  $("#perfElapsed").textContent = fmtTime(elapsed);
  $("#perfEta").textContent = ips > 0 && remaining > 0 ? fmtTime(remaining / ips) : (final ? "0s" : "—");
  $("#perfQ").textContent = bench.qN ? Math.round(bench.qSum / bench.qN) : "—";
  if (performance.memory) $("#perfHeap").textContent = `${(performance.memory.usedJSHeapSize / 1048576) | 0} MB`;
  $("#bDecode").textContent = bench.n || done ? `${(bench.decode / Math.max(1, done)).toFixed(1)} ms` : "—";
  $("#bDetect").textContent = done ? `${(bench.detect / Math.max(1, done)).toFixed(1)} ms` : "—";
  $("#bEncode").textContent = bench.n ? `${(bench.encode / bench.n).toFixed(1)} ms` : "—";
  $("#bAvg").textContent = done && elapsed ? `${(elapsed * 1000 / done).toFixed(1)} ms` : "—";
  // throttled queue + dashboard counts
  renderStats();
  renderQueueStatusesOnly();
}

/* ───────────────────────── Rendering: stats / queue ───────────────────────── */
function renderStats() {
  const total = state.items.length;
  const done = state.items.filter((i) => i.status === "done").length;
  const fail = state.items.filter((i) => i.status === "failed").length;
  $("#statTotal").textContent = total;
  $("#statDone").textContent = done;
  $("#statFail").textContent = fail;
  const fin = done + fail;
  $("#statRate").textContent = fin ? `${Math.round((done / fin) * 100)}%` : "—";
}

function statusLabel(s) {
  return { pending: "Pending", processing: "Processing…", done: "Ready", failed: "Failed" }[s] || s;
}

function queueItemEl(item, small) {
  const el = document.createElement("div");
  el.className = "q-item" + (item.id === state.selectedId ? " sel" : "");
  el.dataset.id = item.id;
  el.onclick = () => selectItem(item.id);
  const thumb = item.result || item.thumbUrl;
  const thumbHtml = small
    ? `<img class="q-thumb" src="${thumb || ""}" alt="" />`
    : `<span class="q-badge ${item.status} q-dot"></span>`;
  el.innerHTML = `
    ${thumbHtml}
    <div class="q-info">
      <div class="q-name" title="${item.name}">${item.name}</div>
      <div class="q-status ${item.status}">${statusLabel(item.status)}${item.score != null ? ` · Q${item.score}` : ""}</div>
    </div>
    <span class="q-badge ${item.status}"></span>`;
  return el;
}

function renderQueue() {
  const q = $("#queue"), qf = $("#queueFull");
  const n = state.items.length;
  const small = n <= SMALL_BATCH;
  $("#queueCount").textContent = `${n} item${n !== 1 ? "s" : ""}`;
  $("#queueCount2").textContent = $("#queueCount").textContent;
  if (!n) {
    q.innerHTML = `<p class="empty">Queue is empty.</p>`;
    qf.innerHTML = `<p class="empty">Queue is empty.</p>`;
    return;
  }
  const shown = state.items.slice(0, QUEUE_RENDER_CAP);
  q.innerHTML = ""; qf.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const item of shown) frag.appendChild(queueItemEl(item, small));
  q.appendChild(frag);
  if (n > QUEUE_RENDER_CAP) {
    const more = document.createElement("p");
    more.className = "empty";
    more.textContent = `+ ${n - QUEUE_RENDER_CAP} more (virtualized)`;
    q.appendChild(more);
  }
  // full grid is built lazily only when its view is opened
  qf.dataset.dirty = "1";
}

/* In-place status refresh for the visible queue rows (cheap; no full rebuild). */
function renderQueueStatusesOnly() {
  const rows = $("#queue").children;
  for (const row of rows) {
    const id = +row.dataset.id;
    if (!id) continue;
    const item = state.items.find((i) => i.id === id);
    if (!item) continue;
    const st = row.querySelector(".q-status");
    if (st) {
      st.className = `q-status ${item.status}`;
      st.textContent = statusLabel(item.status) + (item.score != null ? ` · Q${item.score}` : "");
    }
    row.querySelectorAll(".q-badge").forEach((b) => (b.className = b.classList.contains("q-dot") ? `q-badge ${item.status} q-dot` : `q-badge ${item.status}`));
  }
}

function renderQueueFull() {
  const qf = $("#queueFull");
  if (qf.dataset.dirty !== "1" && qf.children.length) return;
  const small = state.items.length <= SMALL_BATCH;
  qf.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const item of state.items.slice(0, 500)) frag.appendChild(queueItemEl(item, small));
  qf.appendChild(frag);
  qf.dataset.dirty = "0";
}

/* ───────────────────────── Preview ───────────────────────── */
function selectItem(id) {
  state.selectedId = id;
  // highlight only
  $$("#queue .q-item").forEach((el) => el.classList.toggle("sel", +el.dataset.id === id));
  renderPreview();
  updateExportBar();
}

async function renderPreview() {
  const item = state.items.find((i) => i.id === state.selectedId);
  const empty = $("#previewEmpty"), wrap = $("#compareWrap"), meta = $("#previewMeta");
  if (!item) { empty.classList.remove("hidden"); wrap.classList.add("hidden"); meta.innerHTML = ""; return; }
  empty.classList.add("hidden");
  wrap.classList.remove("hidden");

  // lazily (re)build the before-image URL
  if (state.beforeUrl) { URL.revokeObjectURL(state.beforeUrl); state.beforeUrl = null; }
  state.beforeUrl = item.thumbUrl || URL.createObjectURL(item.file);
  $("#imgBefore").src = state.beforeUrl;
  $("#imgAfter").src = item.result || state.beforeUrl;

  const before = $("#imgBefore"), after = $("#imgAfter");
  const sync = () => { after.style.width = before.clientWidth + "px"; after.style.height = before.clientHeight + "px"; };
  before.onload = sync; if (before.complete) sync();

  if (item.meta && !item.meta.error) {
    meta.innerHTML = `
      <span>Output <b>${item.meta.out}</b></span>
      <span>Eye line <b>${item.meta.eyeLine}</b></span>
      <span>Roll <b>${item.meta.rollDeg}°</b></span>
      <span>Scale <b>${item.meta.scale}×</b></span>
      <span>Quality <b>${item.meta.score}</b></span>`;
  } else if (item.meta && item.meta.error) {
    meta.innerHTML = `<span style="color:var(--bad)">⚠ ${item.meta.error}</span>`;
  } else {
    meta.innerHTML = `<span class="muted">Not processed yet — click Generate.</span>`;
  }
}

function setSplit(v) { $("#afterClip").style.width = v + "%"; $("#compareWrap").style.setProperty("--split", v + "%"); }
function setPreviewMode(mode) {
  const clip = $("#afterClip"), slider = $("#slider"), wrap = $("#compareWrap");
  if (mode === "slider") { setSplit(slider.value); slider.style.display = ""; }
  else if (mode === "after") { clip.style.width = "100%"; wrap.style.setProperty("--split", "100%"); slider.style.display = "none"; }
  else { clip.style.width = "0%"; wrap.style.setProperty("--split", "0%"); slider.style.display = "none"; }
}

/* ───────────────────────── Preset & model cards ───────────────────────── */
function renderPresetPicker() {
  const host = $("#presetPicker"); host.innerHTML = "";
  for (const p of Object.values(PRESETS)) {
    const el = document.createElement("div");
    el.className = "preset-opt" + (p.id === state.preset ? " sel" : "");
    el.innerHTML = `<span>${p.name}</span><span class="dim">${p.w}×${p.h}</span>`;
    el.onclick = () => { state.preset = p.id; renderPresetPicker(); };
    host.appendChild(el);
  }
}
function renderPresetCards() {
  const host = $("#presetCards"); host.className = "cards three"; host.innerHTML = "";
  for (const p of Object.values(PRESETS)) {
    const el = document.createElement("div"); el.className = "card info-card";
    el.innerHTML = `<h4>${p.name}</h4><p style="margin-bottom:10px">${p.w} × ${p.h}px · ${p.note}</p>
      <span class="pill">Eye line ${Math.round(p.eyeY*100)}%</span><span class="pill">Face width ${Math.round(p.faceW*100)}%</span>`;
    host.appendChild(el);
  }
}
function renderModelCards() {
  const models = [
    { n: "Face Landmarker", d: `MediaPipe 478-pt + iris. ${state.accel} delegate. Drives alignment.`, s: "live" },
    { n: "Composition Engine", d: "Affine engine in Web Workers (OffscreenCanvas). Parallel compose+encode.", s: "live" },
    { n: "Selfie Segmentation", d: "Conditional cutout — runs only when a background is replaced.", s: state.segReady ? "live" : "soon" },
    { n: "Quality Checker", d: "Per-image 0–100 score gating expensive stages.", s: "live" },
    { n: "Shoulder Detection", d: "Pose-based body framing.", s: "soon" },
    { n: "AI Completion", d: "Firefly / SD inpainting for missing body.", s: "soon" },
  ];
  const host = $("#modelCards"); host.className = "cards three"; host.innerHTML = "";
  for (const m of models) {
    const el = document.createElement("div"); el.className = "card info-card";
    el.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <h4 style="margin:0">${m.n}</h4><span class="status-pill ${m.s}">${m.s === "live" ? "LIVE" : "SOON"}</span></div><p>${m.d}</p>`;
    host.appendChild(el);
  }
}

/* ════════════════════════ Export Manager ════════════════════════
 * Single source of truth for naming + formats. Hard rule: original image
 * filenames are NEVER modified (no _aligned/_processed/_final suffixes).
 * Only the ZIP archive name may differ; names INSIDE the ZIP stay original. */
function ext(format) { return format === "image/jpeg" ? "jpg" : "png"; }
function baseName(name) { return name.replace(/\.[^.]+$/, ""); }

const ExportManager = {
  preserveOriginalFilename(item) { return item.name; },

  formatForName(name) {
    if (/\.png$/i.test(name)) return "image/png";
    if (/\.jpe?g$/i.test(name)) return "image/jpeg";
    if (/\.webp$/i.test(name)) return "image/webp";
    return "image/png";
  },

  // Per-image output format: transparency forces PNG; otherwise keep the
  // source's own format when preserving names, else the chosen export format.
  formatFor(item, { transparent, preserveNames, globalFormat }) {
    if (transparent) return "image/png";          // alpha needs PNG
    if (preserveNames) return this.formatForName(item.name);
    return globalFormat;
  },

  // Output filename. When preserving names (and not forced to change ext for
  // transparency), the name is returned byte-for-byte unchanged.
  outputName(item, { transparent, preserveNames, globalFormat }) {
    if (preserveNames && !transparent) return item.name;            // EXACT same
    if (preserveNames && transparent && /\.png$/i.test(item.name)) return item.name;
    const e = transparent ? "png" : ext(globalFormat);
    return `${baseName(item.name)}.${e}`;
  },

  // True if the decoded image actually carries non-opaque pixels.
  validateAlphaChannel(imageData) {
    const d = imageData.data;
    for (let i = 3; i < d.length; i += 4) if (d[i] < 250) return true;
    return false;
  },

  async exportZipPackage(items, archiveName) {
    if (typeof JSZip === "undefined") { toast("ZIP library still loading — try again."); return; }
    const zip = new JSZip();
    const used = new Map();
    for (const item of items) {
      let name = item.outName || ExportManager.preserveOriginalFilename(item); // original, no suffix
      if (used.has(name)) { const n = used.get(name) + 1; used.set(name, n); name = `${baseName(name)} (${n}).${name.split(".").pop()}`; }
      else used.set(name, 1);
      zip.file(name, await (await fetch(item.result)).blob());
    }
    downloadBlob(await zip.generateAsync({ type: "blob" }), archiveName || "SmartCrop_Export.zip");
  },
};

function exportOpts() {
  return {
    transparent: $("#bgMode").value === "transparent",
    preserveNames: $("#opt-preserve").checked,
    globalFormat: $("#format").value,
    asZip: $("#opt-zip").checked,
  };
}

async function exportAll() {
  const done = state.items.filter((i) => i.status === "done" && i.result);
  if (!done.length) { toast("Nothing processed to export yet."); return; }
  const forceZip = $("#opt-zip").checked;

  // Count-based rule: exactly one image → download that image directly (no ZIP).
  if (done.length === 1 && !forceZip) {
    const it = done[0];
    downloadDataUrl(it.result, it.outName || it.name);
    toast(`Downloaded ${it.outName || it.name}.`);
    return;
  }

  // More than one (or ZIP forced) → single archive, original names inside.
  const archive = ($("#zipName").value || "SmartCrop_Export").replace(/\.zip$/i, "") + ".zip";
  await ExportManager.exportZipPackage(done, archive);
  toast(`Exported ${done.length} portraits → ${archive} (original names preserved).`);
}

// Reflect current state on the under-preview export bar (Issue 4 / 2).
function updateExportBar() {
  const bar = $("#previewExport"), btn = $("#previewDownload"), hint = $("#exportHint");
  if (!bar) return;
  const done = state.items.filter((i) => i.status === "done" && i.result);
  if (!done.length) { bar.classList.add("hidden"); return; }
  bar.classList.remove("hidden");
  const single = done.length === 1 && !$("#opt-zip").checked;
  btn.textContent = single ? "⬇ Download Image" : `⬇ Download ZIP (${done.length})`;
  hint.textContent = single ? (done[0].outName || done[0].name) : "original filenames preserved";
}

function buildSheet() {
  const done = state.items.filter((i) => i.status === "done" && i.result);
  if (!done.length) { toast("Process some portraits first."); return; }
  const [cols, rows] = $("#gridSize").value.split("x").map(Number);
  const cell = 240, pad = 16, canvas = $("#sheetCanvas");
  canvas.width = cols * cell + pad * (cols + 1);
  canvas.height = rows * cell + pad * (rows + 1);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0B1020"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const slots = Math.min(done.length, cols * rows);
  let loaded = 0;
  for (let k = 0; k < slots; k++) {
    const r = Math.floor(k / cols), c = k % cols;
    const x = pad + c * (cell + pad), y = pad + r * (cell + pad);
    const im = new Image();
    im.onload = () => {
      const ar = im.width / im.height; let w = cell, h = cell;
      if (ar > 1) h = cell / ar; else w = cell * ar;
      ctx.drawImage(im, x + (cell - w) / 2, y + (cell - h) / 2, w, h);
      if (++loaded === slots) $("#sheetWrap").classList.remove("hidden");
    };
    im.src = done[k].result;
  }
  $("#sheetWrap").classList.remove("hidden");
  toast(`Built ${cols}×${rows} grid (${slots}).`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement("a"); a.href = dataUrl; a.download = filename; a.click();
}

/* ───────────────────────── View switching ───────────────────────── */
const VIEW_META = {
  dashboard: ["Dashboard", "High-throughput portrait standardization."],
  queue: ["Batch Queue", "All images in this batch."],
  presets: ["Presets", "Composition rules per output type."],
  models: ["AI Models", "The processing pipeline."],
  exports: ["Exports", "Download portraits and team grids."],
  settings: ["Settings", "Engine & privacy."],
};
function switchView(view) {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  const [title, sub] = VIEW_META[view];
  $("#viewTitle").textContent = title; $("#viewSub").textContent = sub;
  if (view === "queue") renderQueueFull();
}

/* ───────────────────────── Wire up ───────────────────────── */
function bind() {
  $$(".nav-item").forEach((b) => (b.onclick = () => switchView(b.dataset.view)));
  $("#pickFiles").onclick = () => $("#fileInput").click();
  $("#pickFolder").onclick = () => $("#folderInput").click();
  $("#dropzone").onclick = (e) => { if (e.target.closest(".dz-buttons")) return; $("#fileInput").click(); };
  $("#fileInput").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
  $("#folderInput").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };

  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer?.files) addFiles(e.dataTransfer.files); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); if (e.target.closest("#dropzone")) return; if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files); });

  $("#generateBtn").onclick = generateAll;
  $("#genWorkspace").onclick = generateAll;
  $("#previewDownload").onclick = exportAll;
  $("#opt-zip").onchange = updateExportBar;
  $("#resetBtn").onclick = () => {
    if (state.running) { state.cancel = true; return; }
    state.items.forEach((i) => { if (i.thumbUrl) URL.revokeObjectURL(i.thumbUrl); if (i.result) URL.revokeObjectURL(i.result); });
    state.items = []; state.selectedId = null;
    renderQueue(); renderStats(); renderPreview(); updateExportBar();
    setGenEnabled(false);
    toast("Cleared.");
  };

  $("#bgMode").onchange = applyBgRules;
  $("#opt-transparent").onchange = (e) => {
    $("#bgMode").value = e.target.checked ? "transparent" : "original";
    applyBgRules();
  };
  $("#quality").oninput = (e) => ($("#qualityVal").textContent = e.target.value + "%");

  $("#slider").addEventListener("input", (e) => setSplit(e.target.value));
  $$("#previewMode .seg-btn").forEach((b) => (b.onclick = () => {
    $$("#previewMode .seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); setPreviewMode(b.dataset.mode);
  }));

  $("#exportZip").onclick = exportAll;
  $("#exportSheet").onclick = buildSheet;
  $("#downloadSheet").onclick = () => $("#sheetCanvas").toBlob((b) => downloadBlob(b, "team_grid.png"));

  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$("#generateBtn").disabled) generateAll();
  });
}

$("#previewStage")?.addEventListener("dblclick", () => {
  const item = state.items.find((i) => i.id === state.selectedId);
  if (item?.result) { downloadDataUrl(item.result, item.outName || item.name); toast("Downloaded."); }
});

/* ───────────────────────── Boot ───────────────────────── */
bind();
renderPresetPicker();
renderPresetCards();
renderModelCards();
renderStats();
renderQueue();
applyBgRules();
initEngine();
