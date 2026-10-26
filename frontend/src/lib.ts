import { useEffect, useState } from 'react';

import { api, UNCERTAIN, type ActivityLabel, type LiveMetrics, type VideoMetadata } from './api';

// ---------- Activities ----------

export interface Activity {
  label: ActivityLabel;
  name: string;
  color: string;
}

/** Display names and colors for the current 6 classes (colors from the Figma mockup). */
export const ACTIVITIES: readonly Activity[] = [
  { label: 'JUMPING', name: 'Jumping', color: '#3478f6' },
  { label: 'JUMPING_JACKS', name: 'Jumping jacks', color: '#8e5cf7' },
  { label: 'BOXING', name: 'Boxing', color: '#e5484d' },
  { label: 'WAVING_2HANDS', name: 'Waving (2 hands)', color: '#12a5a5' },
  { label: 'WAVING_1HAND', name: 'Waving (1 hand)', color: '#f28c28' },
  { label: 'CLAPPING_HANDS', name: 'Clapping hands', color: '#30a46c' },
];

/** Color for frames before the first prediction (the 32-frame buffer). */
export const IDLE_COLOR = '#d2d2d7';

/** Colors for classes added later (extension 3), assigned in order of first appearance. */
const EXTRA_COLORS = ['#c2410c', '#0e7490', '#a21caf', '#4d7c0f', '#b45309', '#1d4ed8', '#be123c', '#047857'];
const extraColors = new Map<string, string>();

/** Name and color for any class. Unknown classes get a readable name ("SIT_DOWN" -> "Sit down") and a spare color. */
export function getActivity(label: ActivityLabel): Activity {
  if (label === UNCERTAIN) return { label, name: 'Uncertain', color: IDLE_COLOR };
  const known = ACTIVITIES.find((a) => a.label === label);
  if (known) return known;
  if (!extraColors.has(label)) extraColors.set(label, EXTRA_COLORS[extraColors.size % EXTRA_COLORS.length]!);
  const words = label.toLowerCase().replace(/_/g, ' ');
  return { label, name: words.charAt(0).toUpperCase() + words.slice(1), color: extraColors.get(label)! };
}

// ---------- Formatting ----------

export const formatPercent = (fraction: number) => `${Math.round(fraction * 100)}%`;
export const formatSeconds = (seconds: number) => `${seconds.toFixed(1)} s`;
export const formatTimeRange = (start: number, end: number) => `${start.toFixed(1)}–${end.toFixed(1)} s`;
export const formatFps = (fps: number) => `${fps < 10 ? fps.toFixed(1) : Math.round(fps)} fps`;
export const formatMs = (ms: number) => `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
export const formatMb = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);
export const formatResolution = (v: VideoMetadata) => `${v.width}×${v.height}`;

/** "10.1 seconds · 304 frames · 30 fps" */
export const formatVideoSummary = (v: VideoMetadata) =>
  `${v.durationS.toFixed(1)} seconds · ${v.frameCount} frames · ${formatFps(v.fps)}`;

/** "10 seconds · 1280×720 · 30 fps" */
export const formatSampleSummary = (v: VideoMetadata) =>
  `${Math.round(v.durationS)} seconds · ${formatResolution(v)} · ${formatFps(v.fps)}`;

/** Tick marks every `step` seconds from 0 to the end of the video. */
export function timeTicks(durationS: number, step = 2) {
  const ticks: number[] = [];
  for (let t = 0; t <= durationS; t += step) ticks.push(t);
  return ticks;
}

// ---------- Hooks ----------

export type Resource<T> = { status: 'loading' } | { status: 'error'; error: string } | { status: 'ready'; data: T };

/** Loads data from the API whenever `key` changes. */
export function useApiResource<T>(load: () => Promise<T>, key: string): Resource<T> {
  const [state, setState] = useState<{ key: string; value: Resource<T> }>({ key, value: { status: 'loading' } });

  useEffect(() => {
    let active = true;
    load().then(
      (data) => active && setState({ key, value: { status: 'ready', data } }),
      (error: Error) => active && setState({ key, value: { status: 'error', error: error.message } }),
    );
    return () => {
      active = false;
    };
  }, [key]); // `load` is a new function every render; `key` says what it loads

  return state.key === key ? state.value : { status: 'loading' };
}

export interface AnalysisState {
  status: 'running' | 'done' | 'error';
  percent: number;
  frame: number;
  totalFrames: number;
  /** Raw top class of the latest prediction. */
  label: ActivityLabel | null;
  /** Smoothed label (extension 1); undefined until the backend sends it. */
  smoothedLabel?: ActivityLabel | null;
  confidence?: number | null;
  /** Live performance (extension 2); undefined until the backend sends it. */
  metrics?: LiveMetrics;
  error?: string;
}

const START: AnalysisState = { status: 'running', percent: 0, frame: 0, totalFrames: 0, label: null };

/** Runs the server-side analysis for a video and tracks its progress. */
export function useAnalysis(video: string): AnalysisState {
  const [state, setState] = useState<AnalysisState>(START);

  useEffect(() => {
    setState(START);
    return api.analyze(video, (event) => {
      if (event.type === 'progress') setState({ ...event, status: 'running' });
      else if (event.type === 'done') setState((s) => ({ ...s, status: 'done', percent: 100 }));
      else setState((s) => ({ ...s, status: 'error', error: event.message }));
    });
  }, [video]);

  return state;
}
