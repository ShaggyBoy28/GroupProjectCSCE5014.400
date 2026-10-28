import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { api, type LiveMetrics } from '../api';
import { ActivityDot, Breadcrumb, Card, Page, Stat, StatusMessage } from '../components';
import { formatFps, formatMb, formatMs, formatPercent, getActivity, useAnalysis } from '../lib';

export function AnalyzingPage() {
  const { video = '' } = useParams();
  const navigate = useNavigate();
  const { status, percent, frame, totalFrames, label, smoothedLabel, confidence, metrics, error } = useAnalysis(video);
  // extension 1: show the smoothed label once the backend sends one, otherwise the raw label
  const shown = smoothedLabel !== undefined ? smoothedLabel : label;
  const showRaw = smoothedLabel !== undefined && label !== null && label !== smoothedLabel;

  // open the results as soon as the server says the analysis is done
  useEffect(() => {
    if (status === 'done') navigate(`/results/${encodeURIComponent(video)}`, { replace: true });
  }, [status, navigate, video]);

  if (status === 'error') {
    return (
      <Page>
        <Breadcrumb current={video} />
        <StatusMessage title="The analysis stopped." message={error} showHomeLink />
      </Page>
    );
  }

  return (
    <Page className="gap-8">
      <Breadcrumb current={video} />
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-[32px] leading-[44px] font-bold tracking-[-0.8px] md:text-[39.1px]">
          Analyzing your video.
        </h1>
        <p className="text-[15.9px] text-muted">
          This takes a few minutes. Keep this page open; results appear automatically.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="overflow-hidden rounded-card bg-black">
          <video
            src={api.videoFileUrl(video)}
            autoPlay
            muted
            loop
            playsInline
            aria-label="The uploaded input video"
            className="aspect-video w-full object-contain"
          />
        </div>
        <Card className="flex flex-col justify-center gap-7 px-8 py-10">
          <div className="flex flex-col gap-3.5">
            <p className="text-[57.6px] leading-16 font-bold tracking-[-1.92px]">{percent}%</p>
            <div
              role="progressbar"
              aria-label="Analysis progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
              className="h-1.5 overflow-hidden rounded-[3px] bg-fill"
            >
              <div className="h-full rounded-[3px] bg-accent transition-[width]" style={{ width: `${percent}%` }} />
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-5">
            <Stat
              label="Frames analyzed"
              value={totalFrames ? `${frame} / ${totalFrames}` : '—'}
              valueClassName="text-[25.3px]"
            />
            <Stat
              label="Current label"
              valueClassName="text-[21px] leading-[34px] tracking-[-0.21px]"
              value={
                <span className="flex items-center gap-2">
                  <ActivityDot label={shown} size={10} />
                  {shown ? getActivity(shown).name : 'Waiting…'}
                </span>
              }
              note={
                [
                  confidence != null && `${formatPercent(confidence)} confidence`,
                  showRaw && `raw: ${getActivity(label).name}`,
                ]
                  .filter(Boolean)
                  .join(' · ') || undefined
              }
            />
          </dl>
          <p className="text-[12.1px] leading-[18.85px] text-muted">Labels start after the first 32 frames.</p>
        </Card>
      </div>
      {metrics && <LivePerformance metrics={metrics} />}
    </Page>
  );
}

/** Extension 2: live performance numbers streamed by the backend with each frame. */
function LivePerformance({ metrics }: { metrics: LiveMetrics }) {
  return (
    <Card className="flex flex-col gap-5 px-6 py-7 md:px-8">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[19px] font-semibold tracking-[-0.19px]">Live performance</h2>
        <p className="text-[12.5px] text-muted">Latest frame</p>
      </div>
      <dl className="grid grid-cols-2 gap-6 md:grid-cols-5">
        <Stat label="Time per frame" value={formatMs(metrics.frameMs)} valueClassName="text-[24px]" />
        <Stat label="Throughput" value={formatFps(metrics.fps)} valueClassName="text-[24px]" />
        <Stat label="Pose detection" value={formatMs(metrics.poseMs)} valueClassName="text-[24px]" />
        <Stat
          label="LSTM inference"
          value={metrics.lstmMs == null ? '—' : formatMs(metrics.lstmMs)}
          valueClassName="text-[24px]"
        />
        <Stat label="Memory" value={formatMb(metrics.memoryMb)} valueClassName="text-[24px]" />
      </dl>
    </Card>
  );
}
