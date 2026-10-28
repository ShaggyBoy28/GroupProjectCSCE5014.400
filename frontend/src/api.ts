// API types and client for the Flask backend (see docs/API.md).

// Fields marked "(planned)" are optional: the UI hides what they power until the backend sends them.
// See docs/API.md for the full contract.

/** A class name from the model, e.g. "JUMPING". New classes from a retrained model work without UI changes. */
export type ActivityLabel = string;

/** Segment label: an activity, or "UNCERTAIN" when smoothing is on and confidence is under the threshold. */
export const UNCERTAIN = 'UNCERTAIN';

export interface VideoMetadata {
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  durationS: number;
}

export interface VideoInfo extends VideoMetadata {
  name: string;
  analyzed: boolean;
}

/** (planned, extension 2) Live performance numbers for the latest frame. */
export interface LiveMetrics {
  frameMs: number | null; // total processing time for this frame
  poseMs: number | null; // Detectron2 pose detection
  lstmMs: number | null; // LSTM inference (null before the first 32-frame window)
  fps: number | null; // frames processed per second (cumulative active throughput)
  memoryMb: number | null; // backend process memory
}

export interface ProgressEvent {
  type: 'progress';
  percent: number;
  frame: number;
  totalFrames: number;
  /** Raw top class of the latest prediction. */
  label: ActivityLabel | null;
  /** (planned, extension 1) Label after smoothing / majority vote; "UNCERTAIN" when under the threshold. */
  smoothedLabel?: ActivityLabel | null;
  /** (planned, extension 1) Confidence of the shown label, 0–1. */
  confidence?: number | null;
  /** (planned, extension 2) */
  metrics?: LiveMetrics;
}

export type AnalysisEvent = ProgressEvent | { type: 'done' } | { type: 'error'; message: string };

export interface Segment {
  label: ActivityLabel;
  startFrame: number;
  endFrame: number;
  startS: number;
  endS: number;
  durationS: number;
  avgConfidence: number;
  peakConfidence: number;
}

export interface ActivityTime {
  label: ActivityLabel;
  seconds: number;
  share: number;
}

export interface Prediction {
  frame: number;
  timeS: number;
  label: ActivityLabel;
  confidence: number;
  probabilities: number[];
}

export interface BreakdownSummary {
  mainActivity: ActivityLabel | null;
  mainActivityShare: number;
  labelChanges: number;
  averageConfidence: number;
  predictionCount: number;
  firstPredictionS: number | null;
}

/** (planned, extension 1) How labels were smoothed. When present, segments use the smoothed labels. */
export interface Smoothing {
  method: string; // e.g. "majority vote" or "moving average"
  window: number; // number of predictions combined
  threshold: number; // minimum confidence to show a label, 0–1
  rawLabelChanges: number; // label changes before smoothing
  uncertainS: number; // seconds labelled UNCERTAIN
}

/** (planned, extension 2) Performance of this analysis run. */
export interface RunPerformance {
  totalS: number | null;
  avgFrameMs: number | null;
  avgPoseMs: number | null;
  avgLstmMs: number | null;
  avgFps: number | null;
  peakMemoryMb: number | null;
  framesProcessed?: number;
  predictionCount?: number;
}

export interface Breakdown {
  video: VideoMetadata & { name: string };
  windowSize: number;
  /** Class order of `predictions[].probabilities`. */
  labels: ActivityLabel[];
  summary: BreakdownSummary;
  smoothing?: Smoothing;
  performance?: RunPerformance;
  segments: Segment[];
  timePerActivity: ActivityTime[];
  predictions: Prediction[];
}

/** (planned, extension 2) Accuracy of the model on the held-out test set. */
export interface ModelMetrics {
  dataset: string; // e.g. "Berkeley MHAD test split"
  sampleCount: number;
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perClass: { label: ActivityLabel; precision: number; recall: number; f1: number; support: number }[];
  /** confusionMatrix[actual][predicted] counts, in `labels` order. */
  confusionMatrix: number[][];
}

/** Model info: its classes and (planned) test-set accuracy. */
export interface ModelInfo {
  labels: ActivityLabel[];
  windowSize: number;
  metrics?: ModelMetrics | null;
}

/** Used until the backend serves GET /api/model. */
export const DEFAULT_MODEL: ModelInfo = {
  labels: ['JUMPING', 'JUMPING_JACKS', 'BOXING', 'WAVING_2HANDS', 'WAVING_1HAND', 'CLAPPING_HANDS'],
  windowSize: 32,
  metrics: null,
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error ?? `Request failed (${response.status})`);
  return body as T;
}

const videoPath = (name: string) => `/videos/${encodeURIComponent(name)}`;

export const api = {
  getSample: () => request<VideoInfo>('/sample'),
  getVideo: (name: string) => request<VideoInfo>(videoPath(name)),
  getBreakdown: (name: string) => request<Breakdown>(`${videoPath(name)}/breakdown`),
  /** Falls back to DEFAULT_MODEL when the endpoint doesn't exist yet. */
  getModel: () =>
    request<Partial<ModelInfo>>('/model').then(
      (info) => ({ ...DEFAULT_MODEL, ...info }),
      () => DEFAULT_MODEL,
    ),

  uploadVideo: (file: File) => {
    const form = new FormData();
    form.append('video', file);
    return request<{ name: string }>('/videos', { method: 'POST', body: form });
  },

  videoFileUrl: (name: string) => `/api${videoPath(name)}/file`,
  resultVideoUrl: (name: string, download = false) =>
    `/api${videoPath(name)}/result${download ? '?download=1' : ''}`,

  /** Starts the analysis and calls onEvent for each server-sent event. Returns a function that stops listening. */
  analyze: (name: string, onEvent: (event: AnalysisEvent) => void) => {
    const source = new EventSource(`/api${videoPath(name)}/analyze`);
    source.onmessage = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as AnalysisEvent;
      if (event.type !== 'progress') source.close();
      onEvent(event);
    };
    source.onerror = () => {
      source.close();
      onEvent({ type: 'error', message: 'Lost connection to the server while analyzing.' });
    };
    return () => source.close();
  },
};
