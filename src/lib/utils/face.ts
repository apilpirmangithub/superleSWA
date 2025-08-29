// Lightweight face embedding using MediaPipe FaceLandmarker via CDN
// No npm deps required; runs fully on-device in the browser

let landmarker: any | null = null;
let vision: any | null = null;
let loadingPromise: Promise<void> | null = null;

const MP_VERSION = (process.env.NEXT_PUBLIC_MEDIAPIPE_TASKS_VISION_VERSION || "0.10.3").trim();
const MAX_FACES = Number.parseInt(process.env.NEXT_PUBLIC_FACE_MAX_FACES || '2', 10);
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODULE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

async function ensureFaceLandmarker() {
  if (landmarker) return;
  if (loadingPromise) return loadingPromise;

  const tryLoad = async (moduleBase: string, wasmBase: string) => {
    // Dynamic ESM import from CDN (works in modern browsers)
    const mod = await import(/* @vite-ignore */ /* webpackIgnore: true */ moduleBase + "/vision_bundle.mjs");
    const { FilesetResolver, FaceLandmarker } = mod as any;
    const filesetResolver = await FilesetResolver.forVisionTasks(wasmBase);
    const lm = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: moduleBase + "/face_landmarker.task" },
      runningMode: "IMAGE",
      numFaces: Math.max(1, Math.min(5, isNaN(MAX_FACES) ? 2 : MAX_FACES)),
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    return { mod, lm };
  };

  loadingPromise = (async () => {
    if (typeof window === 'undefined') return;
    const candidates = [
      { moduleBase: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`, wasmBase: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm` },
      { moduleBase: `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}`, wasmBase: `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/wasm` },
    ];
    for (const c of candidates) {
      try {
        const { mod, lm } = await tryLoad(c.moduleBase, c.wasmBase);
        vision = mod;
        landmarker = lm;
        return;
      } catch (e) {
        console.warn('FaceLandmarker load failed on', c.moduleBase, e);
      }
    }
    landmarker = null;
  })();

  await loadingPromise;
}

async function imageBitmapFromFile(file: File): Promise<ImageBitmap> {
  return await createImageBitmap(file);
}

export type FaceEmbedding = Float32Array;

// Build a normalized landmark-based embedding (translation/scale invariant)
export async function getFaceEmbedding(file: File): Promise<FaceEmbedding | null> {
  if (typeof window === 'undefined') return null;
  await ensureFaceLandmarker();
  if (!landmarker) return null;

  const bitmap = await imageBitmapFromFile(file);
  const result = landmarker.detect(bitmap as any);
  // @ts-ignore
  const faces = result?.faceLandmarks as { x: number; y: number; z?: number }[][] | undefined;
  if (!faces || faces.length === 0) return null;
  const pts = faces[0];
  if (!pts || pts.length === 0) return null;

  // Choose eye indices for scale reference (MediaPipe 468 landmarks)
  const LEFT_EYE_IDX = 33; // left eye outer
  const RIGHT_EYE_IDX = 263; // right eye outer
  const pL = pts[Math.min(LEFT_EYE_IDX, pts.length - 1)];
  const pR = pts[Math.min(RIGHT_EYE_IDX, pts.length - 1)];
  const eyeDist = Math.hypot((pL.x - pR.x), (pL.y - pR.y)) || 1e-6;

  // Center by mean
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= pts.length; my /= pts.length;

  // Build vector from a subset for size: take every 4th point to ~117 pairs
  const vec: number[] = [];
  for (let i = 0; i < pts.length; i += 4) {
    const p = pts[i];
    vec.push((p.x - mx) / eyeDist, (p.y - my) / eyeDist);
  }
  // L2 normalize
  let norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  const emb = new Float32Array(vec.map(v => v / norm));
  return emb;
}

export async function countFaces(file: File): Promise<number> {
  if (typeof window === 'undefined') return 0;
  await ensureFaceLandmarker();
  if (!landmarker) return 0;
  const bitmap = await imageBitmapFromFile(file);
  const result = landmarker.detect(bitmap as any);
  // @ts-ignore
  const faces = result?.faceLandmarks as { x: number; y: number; z?: number }[][] | undefined;
  return Array.isArray(faces) ? faces.length : 0;
}

export function cosineSimilarity(a: FaceEmbedding, b: FaceEmbedding): number {
  const len = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

// Lightweight liveness: detect one blink + slight head/pose movement within duration
export async function checkLiveness(videoEl: HTMLVideoElement, opts?: { durationMs?: number; moveThreshold?: number; blinkThreshold?: number }): Promise<{ ok: boolean; reason?: string }> {
  if (typeof window === 'undefined') return { ok: false, reason: 'no-window' };
  await ensureFaceLandmarker();
  if (!landmarker) return { ok: false, reason: 'no-landmarker' };

  const durationMs = opts?.durationMs ?? 6000;
  const moveThreshold = opts?.moveThreshold ?? 0.02; // normalized move
  const blinkThresh = opts?.blinkThreshold ?? 0.5;   // blendshape score

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { ok: false, reason: 'no-canvas' };

  let start = performance.now();
  let blinked = false;
  let moved = false;
  let lastCenter: {x:number;y:number}|null = null;
  let lastBlinkScore = 0;

  return new Promise((resolve) => {
    const step = () => {
      const now = performance.now();
      if (now - start > durationMs) {
        resolve({ ok: blinked && moved, reason: !blinked ? 'no-blink' : (!moved ? 'no-move' : undefined) });
        return;
      }
      const vw = videoEl.videoWidth || 640;
      const vh = videoEl.videoHeight || 480;
      canvas.width = vw; canvas.height = vh;
      try {
        ctx.drawImage(videoEl, 0, 0, vw, vh);
        // Detect on canvas directly
        const result = landmarker.detect(canvas as any);
        // @ts-ignore
        const faces = result?.faceLandmarks as { x: number; y: number; z?: number }[][] | undefined;
        // @ts-ignore
        const blends = result?.faceBlendshapes as { categories: { categoryName: string; score: number }[] }[] | undefined;
        if (faces && faces.length > 0) {
          const pts = faces[0];
          // center
          let mx = 0, my = 0;
          for (const p of pts) { mx += p.x; my += p.y; }
          mx /= pts.length; my /= pts.length;
          const center = { x: mx, y: my };
          if (lastCenter) {
            const dx = center.x - lastCenter.x;
            const dy = center.y - lastCenter.y;
            const dist = Math.hypot(dx, dy);
            if (dist > moveThreshold) moved = true;
          }
          lastCenter = center;

          if (blends && blends[0]?.categories) {
            const cat = blends[0].categories;
            const eyeBlinkL = cat.find(c => c.categoryName.toLowerCase().includes('eyeblinkleft'))?.score ?? 0;
            const eyeBlinkR = cat.find(c => c.categoryName.toLowerCase().includes('eyeblinkright'))?.score ?? 0;
            const blinkScore = Math.max(eyeBlinkL, eyeBlinkR);
            // detect transition high after low (simple blink event)
            if (blinkScore >= blinkThresh && lastBlinkScore < 0.2) blinked = true;
            lastBlinkScore = blinkScore;
          }
        }
      } catch {}
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}
