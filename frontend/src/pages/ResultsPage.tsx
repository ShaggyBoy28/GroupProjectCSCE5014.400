import { Link, NavLink, Outlet, useOutletContext, useParams } from 'react-router-dom';

import { api, type VideoInfo } from '../api';
import chevronUrl from '../assets/icons/chevron.svg';
import { Breadcrumb, Card, Page, PillLink, pillClassName, Stat, StatusMessage } from '../components';
import { formatFps, formatResolution, formatSeconds, formatVideoSummary, useApiResource } from '../lib';

/** The analyzed video's info, shared with the nested "Result video" and "Activity breakdown" views. */
export const useResultVideo = () => useOutletContext<VideoInfo>();

/** Header (title, buttons, view tabs) shared by both result views. */
export function ResultsLayout() {
  const { video = '' } = useParams();
  const info = useApiResource(() => api.getVideo(video), video);

  if (info.status !== 'ready' || !info.data.analyzed) {
    return (
      <Page>
        <Breadcrumb current={video} />
        {info.status === 'loading' && <StatusMessage title="Loading results…" />}
        {info.status === 'error' && (
          <StatusMessage title="We couldn't load this video." message={info.error} showHomeLink />
        )}
        {info.status === 'ready' && (
          <div className="flex flex-col items-center gap-3 py-24 text-center">
            <h1 className="text-[24px] font-semibold tracking-[-0.24px]">
              This video hasn&apos;t been analyzed yet.
            </h1>
            <PillLink to={`/analyzing/${encodeURIComponent(video)}`}>Analyze it now</PillLink>
          </div>
        )}
      </Page>
    );
  }

  const base = `/results/${encodeURIComponent(video)}`;
  const tabClass = ({ isActive }: { isActive: boolean }) =>
    `flex h-8 min-w-[150px] items-center justify-center rounded-[7px] px-4 text-[13px] md:min-w-[170px] ${
      isActive ? 'bg-white font-semibold shadow-[0_1px_1.5px_rgba(0,0,0,0.12)]' : 'hover:bg-white/50'
    }`;

  return (
    <Page className="gap-7">
      <Breadcrumb current={video} />
      <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        <div className="flex min-w-0 flex-col gap-1.5">
          <h1 className="truncate text-[33.3px] leading-[37.4px] font-bold tracking-[-0.68px]">{video}</h1>
          <p className="text-[14.3px] text-muted">{formatVideoSummary(info.data)}</p>
        </div>
        <div className="flex gap-2.5">
          <a href={api.resultVideoUrl(video, true)} download className={pillClassName('secondary')}>
            Download video
          </a>
          <PillLink to="/" variant="secondary">
            Analyze another
          </PillLink>
        </div>
      </div>
      <nav aria-label="Result views" className="flex justify-center">
        <div className="flex gap-0.5 rounded-[9px] bg-segmented p-0.5">
          <NavLink to={base} end className={tabClass}>
            Result video
          </NavLink>
          <NavLink to={`${base}/breakdown`} className={tabClass}>
            Activity breakdown
          </NavLink>
        </div>
      </nav>
      <Outlet context={info.data} />
    </Page>
  );
}

/** "Result video" tab: annotated video plus video stats. */
export function ResultVideoView() {
  const video = useResultVideo();

  return (
    <div className="grid grid-cols-1 items-start gap-5 md:grid-cols-[minmax(0,8fr)_minmax(0,4fr)]">
      <div className="overflow-hidden rounded-card bg-black">
        <video
          src={api.resultVideoUrl(video.name)}
          controls
          playsInline
          aria-label="Annotated result video with the detected skeleton and predicted action"
          className="aspect-video w-full object-contain"
        />
      </div>
      <div className="flex flex-col gap-5">
        <Card className="p-7">
          <dl className="grid grid-cols-2 gap-x-5 gap-y-6">
            <Stat label="Length" value={formatSeconds(video.durationS)} />
            <Stat label="Frames" value={video.frameCount} />
            <Stat label="Frame rate" value={formatFps(video.fps)} />
            <Stat label="Resolution" value={formatResolution(video)} />
          </dl>
        </Card>
        <Card className="flex flex-col p-7">
          <h2 className="text-[18.7px] font-semibold tracking-[-0.19px]">See how the model decided.</h2>
          <p className="mt-2 text-[14px] leading-[21.75px] text-muted">
            Confidence over time, a timeline of activities and how long each lasted.
          </p>
          <Link
            to="breakdown"
            className="mt-3.5 inline-flex items-center gap-1.5 self-start text-[13.7px] font-medium text-accent hover:underline"
          >
            Activity breakdown
            <img src={chevronUrl} width={8} height={12} alt="" />
          </Link>
        </Card>
      </div>
    </div>
  );
}
