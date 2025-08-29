// Client-side face embedding using InsightFace (ArcFace) via ONNX Runtime Web loaded from CDN
// Runs fully in the browser; no npm deps. Returns 512-dim normalized embeddings.

let ortLoaded: Promise<any> | null = null;
let recogSession: any | null = null;

const ORT_CDN = (process.env.NEXT_PUBLIC_ORT_WEB_CDN || "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js").trim();
const INSIGHT_RECOG_URL = (process.env.NEXT_PUBLIC_INSIGHTFACE_RECOG_URL || "https://cdn.jsdelivr.net/npm/insightface.js@0.1.0/models/buffalo_l/w600k_r50.onnx").trim();
const INPUT_SIZE = Number.parseInt(process.env.NEXT_PUBLIC_INSIGHTFACE_INPUT_SIZE || "112", 10);
const CHANNEL_ORDER = (process.env.NEXT_PUBLIC_INSIGHTFACE_CHANNEL_ORDER || "BGR").toUpperCase() as "RGB" | "BGR";

async function loadScriptOnce(src: string): Promise<void> {
  if (typeof window === 'undefined') return;
  const existing = Array.from(document.scripts).find(s => s.src === src);
  if (existing) return;
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

async function ensureOrt(): Promise<any> {
  if (typeof window === 'undefined') return null;
  if ((window as any).ort) return (window as any).ort;
  if (!ortLoaded) {
    ortLoaded = (async () => {
      await loadScriptOnce(ORT_CDN);
      const ort = (window as any).ort;
      if (!ort) throw new Error('onnxruntime-web not available');
      // Prefer WASM; WebGL may fail on some devices. wasmPaths auto-resolve from CDN.
      try {
        ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 1);
        ort.env.wasm.simd = true;
      } catch {}
      return ort;
    })();
  }
  return ortLoaded;
}

function centerCropBox(w: number, h: number, scale = 0.7) {
  const size = Math.floor(Math.min(w, h) * scale);
  const x = Math.max(0, Math.floor((w - size) / 2));
  const y = Math.max(0, Math.floor((h - size) / 2));
  return { x, y, size };
}

async function detectFaceBox(imgBitmap: ImageBitmap): Promise<{ x: number; y: number; width: number; height: number } | null> {
  // Use built-in FaceDetector when available; otherwise fallback to center crop.
  // @ts-ignore
  if (typeof window !== 'undefined' && (window as any).FaceDetector) {
    try {
      // @ts-ignore
      const detector = new (window as any).FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
      // FaceDetector API accepts ImageBitmap in modern browsers
      const faces = await detector.detect(imgBitmap as any);
      if (Array.isArray(faces) && faces.length > 0) {
        const box = faces[0].boundingBox as DOMRectReadOnly;
        return { x: Math.max(0, Math.floor(box.x)), y: Math.max(0, Math.floor(box.y)), width: Math.floor(box.width), height: Math.floor(box.height) };
      }
    } catch {
      // ignore and fallback
    }
  }
  const { x, y, size } = centerCropBox(imgBitmap.width, imgBitmap.height, 0.7);
  return { x, y, width: size, height: size };
}

async function ensureRecognizer(): Promise<any> {
  if (recogSession) return recogSession;
  const ort: any = await ensureOrt();
  if (!ort) return null;
  recogSession = await ort.InferenceSession.create(INSIGHT_RECOG_URL, { executionProviders: ['wasm'] }).catch(async () => {
    // Fallback: default provider
    return await ort.InferenceSession.create(INSIGHT_RECOG_URL);
  });
  return recogSession;
}

function toCHWFloat(imageData: ImageData, size: number, order: 'RGB' | 'BGR'): Float32Array {
  const { data, width, height } = imageData;
  // Resize to size x size using canvas
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D not available');
  const tmp = document.createElement('canvas');
  tmp.width = width; tmp.height = height;
  const tctx = tmp.getContext('2d');
  if (!tctx) throw new Error('Canvas 2D not available');
  tctx.putImageData(imageData, 0, 0);
  ctx.drawImage(tmp, 0, 0, size, size);
  const resized = ctx.getImageData(0, 0, size, size).data;

  const out = new Float32Array(3 * size * size);
  const mean = 127.5;
  const std = 128; // scale to [-1, 1]
  let p = 0;
  for (let i = 0; i < size * size; i++) {
    const r = resized[p++];
    const g = resized[p++];
    const b = resized[p++];
    p++; // skip alpha
    const rn = (r - mean) / std;
    const gn = (g - mean) / std;
    const bn = (b - mean) / std;
    if (order === 'RGB') {
      out[i] = rn; // R
      out[i + size * size] = gn; // G
      out[i + 2 * size * size] = bn; // B
    } else {
      out[i] = bn; // B
      out[i + size * size] = gn; // G
      out[i + 2 * size * size] = rn; // R
    }
  }
  return out;
}

async function bitmapFromFile(file: File): Promise<ImageBitmap> {
  const url = URL.createObjectURL(file);
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export type FaceEmbedding = Float32Array;

export async function getFaceEmbedding(file: File): Promise<FaceEmbedding | null> {
  if (typeof window === 'undefined') return null;
  try {
    const session = await ensureRecognizer();
    if (!session) return null;

    const bitmap = await bitmapFromFile(file);
    const box = await detectFaceBox(bitmap);
    if (!box) return null;

    // Crop face region with margin
    const margin = 0.2; // 20% margin
    const cx = Math.max(0, Math.floor(box.x - box.width * margin));
    const cy = Math.max(0, Math.floor(box.y - box.height * margin));
    const cw = Math.min(bitmap.width - cx, Math.floor(box.width * (1 + 2 * margin)));
    const ch = Math.min(bitmap.height - cy, Math.floor(box.height * (1 + 2 * margin)));

    const canvas = document.createElement('canvas');
    canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(bitmap as any, cx, cy, cw, ch, 0, 0, cw, ch);
    const imgData = ctx.getImageData(0, 0, cw, ch);

    const chw = toCHWFloat(imgData, INPUT_SIZE, CHANNEL_ORDER);
    const ort: any = (window as any).ort;
    const input = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds: Record<string, any> = {};
    const inName = session.inputNames?.[0] || 'data';
    feeds[inName] = input;

    const results = await session.run(feeds);
    const outName = session.outputNames?.[0] || Object.keys(results)[0];
    const output = results[outName];
    const feat = output?.data as Float32Array | undefined;
    if (!feat) return null;

    // L2 normalize
    let sum = 0;
    for (let i = 0; i < feat.length; i++) sum += feat[i] * feat[i];
    const norm = Math.sqrt(sum) || 1;
    const emb = new Float32Array(feat.length);
    for (let i = 0; i < feat.length; i++) emb[i] = feat[i] / norm;
    return emb;
  } catch {
    return null;
  }
}

export async function countFaces(file: File): Promise<number> {
  if (typeof window === 'undefined') return 0;
  try {
    // @ts-ignore
    const hasFD = (window as any).FaceDetector != null;
    const bitmap = await bitmapFromFile(file);
    if (hasFD) {
      try {
        // @ts-ignore
        const detector = new (window as any).FaceDetector({ fastMode: true, maxDetectedFaces: 5 });
        const faces = await detector.detect(bitmap as any);
        return Array.isArray(faces) ? faces.length : 0;
      } catch {}
    }
    const box = await detectFaceBox(bitmap);
    return box ? 1 : 0;
  } catch {
    return 0;
  }
}

export type LivenessResult = { ok: true } | { ok: false; reason: string };

export async function checkLiveness(video: HTMLVideoElement, opts?: { durationMs?: number; minMoves?: number }): Promise<LivenessResult> {
  if (typeof window === 'undefined') return { ok: false, reason: 'no-window' };
  const durationMs = Math.max(2000, Math.min(15000, opts?.durationMs ?? 6000));
  const minMoves = Math.max(1, Math.min(10, opts?.minMoves ?? 2));

  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return { ok: false, reason: 'no-canvas' };

  // @ts-ignore
  const hasFD = (window as any).FaceDetector != null;
  // @ts-ignore
  const detector = hasFD ? new (window as any).FaceDetector({ fastMode: true, maxDetectedFaces: 1 }) : null;

  let lastCenter: { x: number; y: number } | null = null;
  let motionBursts = 0;
  let samples = 0;
  let lastFrame: Uint8ClampedArray | null = null;

  const sample = async (): Promise<void> => {
    ctx.drawImage(video, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);

    let roi = { x: 0, y: 0, width: w, height: h } as { x: number; y: number; width: number; height: number };
    if (detector) {
      try {
        const faces = await detector.detect(canvas as unknown as any);
        if (faces && faces[0]) {
          const b = faces[0].boundingBox as DOMRectReadOnly;
          roi = { x: Math.max(0, Math.floor(b.x)), y: Math.max(0, Math.floor(b.y)), width: Math.floor(b.width), height: Math.floor(b.height) };
        }
      } catch {}
    }

    const center = { x: roi.x + roi.width / 2, y: roi.y + roi.height / 2 };
    if (lastCenter) {
      const dx = center.x - lastCenter.x;
      const dy = center.y - lastCenter.y;
      const dist = Math.hypot(dx, dy);
      if (dist > Math.max(4, Math.min(w, h) * 0.01)) motionBursts++;
    }
    lastCenter = center;

    let motion = 0;
    if (lastFrame) {
      const stride = 4;
      const x0 = Math.max(0, roi.x), y0 = Math.max(0, roi.y);
      const x1 = Math.min(w, roi.x + roi.width), y1 = Math.min(h, roi.y + roi.height);
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const idx = (y * w + x) * stride;
          const d = Math.abs(img.data[idx] - (lastFrame as Uint8ClampedArray)[idx]) + Math.abs(img.data[idx+1] - (lastFrame as Uint8ClampedArray)[idx+1]) + Math.abs(img.data[idx+2] - (lastFrame as Uint8ClampedArray)[idx+2]);
          if (d > 45) motion++;
        }
      }
    }
    lastFrame = img.data;

    if (motion > (roi.width * roi.height) / (2*2*120)) motionBursts++;

    samples++;
  };

  const start = Date.now();
  while (Date.now() - start < durationMs) {
    await sample();
    await new Promise(r => setTimeout(r, 220));
  }

  if (samples < 3) return { ok: false, reason: 'not-enough-samples' };
  if (motionBursts >= minMoves) return { ok: true };
  return { ok: false, reason: 'no-motion' };
}

export function cosineSimilarity(a: FaceEmbedding, b: FaceEmbedding): number {
  const len = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

export async function getAugmentedEmbeddings(file: File, rotations: number[] = [-25, -15, 0, 15, 25], flips: boolean[] = [false, true]): Promise<FaceEmbedding[]> {
  if (typeof window === 'undefined') return [];
  const session = await ensureRecognizer();
  if (!session) return [];
  const bitmap = await bitmapFromFile(file);
  const box = await detectFaceBox(bitmap);
  if (!box) return [];

  const margin = 0.3;
  const cx = Math.max(0, Math.floor(box.x - box.width * margin));
  const cy = Math.max(0, Math.floor(box.y - box.height * margin));
  const cw = Math.min(bitmap.width - cx, Math.floor(box.width * (1 + 2 * margin)));
  const ch = Math.min(bitmap.height - cy, Math.floor(box.height * (1 + 2 * margin)));

  const canvas = document.createElement('canvas');
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext('2d');
  if (!ctx) return [];

  const ort: any = (window as any).ort;
  const embs: FaceEmbedding[] = [];

  for (const flip of flips) {
    for (const deg of rotations) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      ctx.translate(cw / 2, ch / 2);
      if (flip) ctx.scale(-1, 1);
      const rad = (deg * Math.PI) / 180;
      ctx.rotate(rad);
      ctx.drawImage(bitmap as any, cx, cy, cw, ch, -cw / 2, -ch / 2, cw, ch);

      const imgData = ctx.getImageData(0, 0, cw, ch);
      const chw = toCHWFloat(imgData, INPUT_SIZE, CHANNEL_ORDER);
      const input = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const feeds: Record<string, any> = {};
      const inName = session.inputNames?.[0] || 'data';
      feeds[inName] = input;
      try {
        const results = await session.run(feeds);
        const outName = session.outputNames?.[0] || Object.keys(results)[0];
        const output = results[outName];
        const feat = output?.data as Float32Array | undefined;
        if (!feat) continue;
        let sum = 0;
        for (let i = 0; i < feat.length; i++) sum += feat[i] * feat[i];
        const norm = Math.sqrt(sum) || 1;
        const emb = new Float32Array(feat.length);
        for (let i = 0; i < feat.length; i++) emb[i] = feat[i] / norm;
        embs.push(emb);
      } catch {}
    }
  }

  return embs;
}

export async function compareFacesAdvanced(ref: File, probe: File, opts?: { rotations?: number[]; allowFlip?: boolean }): Promise<{ best: number; refCount: number; probeCount: number }> {
  const rotations = opts?.rotations ?? [-25, -15, 0, 15, 25];
  const flips = (opts?.allowFlip ?? true) ? [false, true] : [false];
  const [refEmb, probeEmb] = await Promise.all([
    getAugmentedEmbeddings(ref, rotations, flips),
    getAugmentedEmbeddings(probe, rotations, flips)
  ]);
  let best = 0;
  for (const a of refEmb) {
    for (const b of probeEmb) {
      const s = cosineSimilarity(a, b);
      if (s > best) best = s;
    }
  }
  return { best, refCount: refEmb.length, probeCount: probeEmb.length };
}
