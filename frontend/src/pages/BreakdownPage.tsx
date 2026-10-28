import { api, type Breakdown, type ModelMetrics, type RunPerformance, type Smoothing } from '../api';
import { ActivityDot, ActivityLegend, ActivityName, Card, CardHeader, Stat, StatusMessage } from '../components';
import {
  formatFps,
  formatMb,
  formatMs,
  formatPercent,
  formatSeconds,
  formatTimeRange,
  getActivity,
  timeTicks,
  useApiResource,
} from '../lib';
import { useResultVideo } from './ResultsPage';

/** "Activity breakdown" tab: summary, timeline, confidence chart, time per activity, segments. */
export function BreakdownView() {
  const video = useResultVideo();
  const breakdown = useApiResource(() => api.getBreakdown(video.name), video.name);
  const model = useApiResource(api.getModel, 'model');
  const modelMetrics = model.status === 'ready' ? model.data.metrics : null;

  if (breakdown.status === 'loading') return <StatusMessage title="Loading breakdown…" />;
  if (breakdown.status === 'error') {
    return <StatusMessage title="We couldn't load the breakdown." message={breakdown.error} showHomeLink />;
  }
  if (breakdown.data.predictions.length === 0) {
    return (
      <div className="flex flex-col gap-7">
        <StatusMessage
          title="No activity was detected."
          message="The model needs one person, whole body in view, for at least 32 frames before it can label anything."
        />
        {breakdown.data.performance && <PerformanceCard performance={breakdown.data.performance} />}
        {modelMetrics && <ModelAccuracyCard metrics={modelMetrics} labels={model.status === 'ready' ? model.data.labels : []} />}
      </div>
    );
  }

  const data = breakdown.data;
  return (
    <div className="flex flex-col gap-7">
      <SummaryCard data={data} />
      {data.smoothing && <SmoothingCard smoothing={data.smoothing} />}
      <Timeline data={data} />
      <ConfidenceChart data={data} />
      <div className="grid grid-cols-1 items-start gap-5 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <TimePerActivity data={data} />
        <SegmentsTable data={data} />
      </div>
      {data.performance && <PerformanceCard performance={data.performance} />}
      {modelMetrics && <ModelAccuracyCard metrics={modelMetrics} labels={model.status === 'ready' ? model.data.labels : []} />}
    </div>
  );
}

function SummaryCard({ data: { summary, windowSize, smoothing } }: { data: Breakdown }) {
  const main = summary.mainActivity;
  return (
    <Card className="px-6 py-8 md:px-9">
      <dl className="grid grid-cols-2 gap-6 md:grid-cols-4">
        <Stat
          label="Main activity"
          valueClassName="text-[24px] leading-[34px] tracking-[-0.48px]"
          value={
            main ? (
              <span className="flex items-center gap-2">
                <ActivityDot label={main} size={10} />
                {getActivity(main).name}
              </span>
            ) : (
              '—'
            )
          }
          note={`${formatPercent(summary.mainActivityShare)} of the analyzed video`}
        />
        <Stat
          label="Label changes"
          value={summary.labelChanges}
          note={
            smoothing
              ? `After smoothing (raw: ${smoothing.rawLabelChanges})`
              : 'Times the top activity switched'
          }
        />
        <Stat
          label="Average confidence"
          value={formatPercent(summary.averageConfidence)}
          note="Top-class probability"
        />
        <Stat
          label="Predictions"
          value={summary.predictionCount}
          note={`One per frame after the first ${windowSize}`}
        />
      </dl>
    </Card>
  );
}

function Timeline({ data: { segments, video, windowSize, labels, smoothing } }: { data: Breakdown }) {
  const pct = (seconds: number) => `${(seconds / video.durationS) * 100}%`;
  return (
    <Card className="flex flex-col gap-4 px-6 py-8 md:px-9">
      <CardHeader title="Timeline" note={`${segments.length} segment${segments.length === 1 ? '' : 's'}`} />
      <div className="relative h-7 overflow-hidden rounded-lg bg-fill">
        {segments.map((s) => (
          <div
            key={s.startFrame}
            title={`${getActivity(s.label).name} · ${formatTimeRange(s.startS, s.endS)}`}
            className="absolute inset-y-0 shadow-[inset_2px_0_0_white]"
            style={{ left: pct(s.startS), width: pct(s.durationS), backgroundColor: getActivity(s.label).color }}
          />
        ))}
      </div>
      <div className="relative h-4 text-[12px] text-muted" aria-hidden>
        {timeTicks(video.durationS).map((tick) => {
          const position = (tick / video.durationS) * 100;
          // shifting each label by its own position keeps 0s flush left and the last tick flush right
          return (
            <span
              key={tick}
              className="absolute top-0"
              style={{ left: `${position}%`, transform: `translateX(-${position}%)` }}
            >
              {tick}s
            </span>
          );
        })}
      </div>
      <ActivityLegend
        labels={labels}
        idleLabel={`First ${windowSize} frames`}
        showUncertain={Boolean(smoothing)}
        className="text-[12.7px]"
      />
    </Card>
  );
}

// SVG coordinate space for the confidence chart (it scales with the card width).
const CHART = { width: 1080, height: 290, left: 44, right: 1076, top: 12, bottom: 262 };

function ConfidenceChart({ data: { predictions, video, summary, labels, smoothing } }: { data: Breakdown }) {
  const x = (seconds: number) => CHART.left + (seconds / video.durationS) * (CHART.right - CHART.left);
  const y = (probability: number) => CHART.bottom - probability * (CHART.bottom - CHART.top);

  return (
    <Card className="flex flex-col gap-4 px-6 py-8 md:px-9">
      <CardHeader title="Confidence over time" note="Model probability for each activity, per frame" />
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${CHART.width} ${CHART.height}`}
          className="h-auto w-full min-w-[640px]"
          role="img"
          aria-label="Line chart of each activity's probability over the video"
        >
          {/* frames before the first prediction (the LSTM's input buffer) */}
          <rect
            x={CHART.left}
            y={CHART.top}
            width={Math.max(x(summary.firstPredictionS ?? 0) - CHART.left, 0)}
            height={CHART.bottom - CHART.top}
            className="fill-canvas"
          />
          {[0, 0.5, 1].map((level) => (
            <g key={level}>
              <line x1={CHART.left} x2={CHART.right} y1={y(level)} y2={y(level)} className="stroke-fill" />
              <text x={CHART.left - 10} y={y(level) + 4} textAnchor="end" className="fill-muted text-[12px]">
                {level * 100}%
              </text>
            </g>
          ))}
          {timeTicks(video.durationS).map((tick) => (
            <text
              key={tick}
              x={x(tick)}
              y={CHART.height - 6}
              textAnchor="middle"
              className="fill-muted text-[12px]"
            >
              {tick}s
            </text>
          ))}
          {/* extension 1: labels under this confidence are shown as Uncertain */}
          {smoothing && (
            <g>
              <line
                x1={CHART.left}
                x2={CHART.right}
                y1={y(smoothing.threshold)}
                y2={y(smoothing.threshold)}
                stroke="#1d1d1f"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
              />
              <text
                x={CHART.right}
                y={y(smoothing.threshold) - 6}
                textAnchor="end"
                className="fill-muted text-[12px]"
              >
                Threshold {formatPercent(smoothing.threshold)}
              </text>
            </g>
          )}
          {labels.map((label, classIndex) => (
            <polyline
              key={label}
              fill="none"
              stroke={getActivity(label).color}
              strokeWidth={1.6}
              strokeLinejoin="round"
              points={predictions
                .map((p) => `${x(p.timeS).toFixed(1)},${y(p.probabilities[classIndex] ?? 0).toFixed(1)}`)
                .join(' ')}
            />
          ))}
        </svg>
      </div>
    </Card>
  );
}

function TimePerActivity({ data: { timePerActivity } }: { data: Breakdown }) {
  return (
    <Card className="flex flex-col gap-[18px] px-6 py-8 md:px-9">
      <h2 className="text-[20.3px] font-semibold tracking-[-0.21px]">Time per activity</h2>
      <ul className="flex flex-col gap-[18px]">
        {timePerActivity.map((row) => (
          <li key={row.label} className="flex flex-col gap-[7px]">
            <div className="flex justify-between gap-4 text-[14.8px]">
              <span>{getActivity(row.label).name}</span>
              <span className="text-muted">
                {formatSeconds(row.seconds)} · {formatPercent(row.share)}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-[3px] bg-track">
              <div
                className="h-full rounded-[3px]"
                style={{
                  width: `${row.share * 100}%`,
                  minWidth: row.share > 0 ? 4 : 0,
                  backgroundColor: getActivity(row.label).color,
                }}
              />
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function SegmentsTable({ data: { segments } }: { data: Breakdown }) {
  const head = 'pt-[9px] pb-[10.5px] text-[12.7px] font-medium text-muted';
  const cell = 'border-t border-track py-[13px] text-[15px]';
  return (
    <Card className="flex flex-col gap-3 px-6 py-8 md:px-9">
      <h2 className="text-[20.5px] font-semibold tracking-[-0.21px]">Segments</h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] border-collapse">
          <thead>
            <tr>
              <th className={`${head} pr-2 text-left`}>Time</th>
              <th className={`${head} px-2 text-left`}>Activity</th>
              <th className={`${head} px-2 text-right`}>Duration</th>
              <th className={`${head} px-2 text-right`}>Avg.</th>
              <th className={`${head} pl-2 text-right`}>Peak</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((s) => (
              <tr key={s.startFrame}>
                <td className={`${cell} pr-2 whitespace-nowrap text-muted`}>{formatTimeRange(s.startS, s.endS)}</td>
                <td className={`${cell} px-2`}>
                  <ActivityName label={s.label} />
                </td>
                <td className={`${cell} px-2 text-right whitespace-nowrap`}>{formatSeconds(s.durationS)}</td>
                <td className={`${cell} px-2 text-right`}>{formatPercent(s.avgConfidence)}</td>
                <td className={`${cell} pl-2 text-right`}>{formatPercent(s.peakConfidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** Extension 1: how the shown labels were smoothed. */
function SmoothingCard({ smoothing }: { smoothing: Smoothing }) {
  return (
    <Card className="flex flex-col gap-5 px-6 py-8 md:px-9">
      <CardHeader title="Prediction smoothing" note="Reduces flicker between labels" />
      <dl className="grid grid-cols-2 gap-6 md:grid-cols-4">
        <Stat
          label="Method"
          value={smoothing.method.charAt(0).toUpperCase() + smoothing.method.slice(1)}
          valueClassName="text-[20px]"
        />
        <Stat
          label="Window"
          value={`${smoothing.window} predictions`}
          valueClassName="text-[20px]"
          note="Combined for each label"
        />
        <Stat
          label="Confidence threshold"
          value={formatPercent(smoothing.threshold)}
          valueClassName="text-[20px]"
          note="Below this the label is Uncertain"
        />
        <Stat label="Uncertain time" value={formatSeconds(smoothing.uncertainS)} valueClassName="text-[20px]" />
      </dl>
    </Card>
  );
}

/** Extension 2: averages for this analysis run. */
function PerformanceCard({ performance: p }: { performance: RunPerformance }) {
  return (
    <Card className="flex flex-col gap-5 px-6 py-8 md:px-9">
      <CardHeader title="Performance" note={`Whole video processed in ${formatSeconds(p.totalS)}`} />
      <dl className="grid grid-cols-2 gap-6 md:grid-cols-5">
        <Stat label="Avg. time per frame" value={formatMs(p.avgFrameMs)} valueClassName="text-[24px]" />
        <Stat label="Avg. throughput" value={formatFps(p.avgFps)} valueClassName="text-[24px]" />
        <Stat label="Avg. pose detection" value={formatMs(p.avgPoseMs)} valueClassName="text-[24px]" />
        <Stat label="Avg. LSTM inference" value={formatMs(p.avgLstmMs)} valueClassName="text-[24px]" />
        <Stat label="Peak memory" value={formatMb(p.peakMemoryMb)} valueClassName="text-[24px]" />
      </dl>
    </Card>
  );
}

/** Extension 2: the model's accuracy on the held-out test set (same for every video). */
function ModelAccuracyCard({ metrics: m, labels }: { metrics: ModelMetrics; labels: string[] }) {
  const head = 'pt-[9px] pb-[10.5px] text-[12.7px] font-medium text-muted';
  const cell = 'border-t border-track py-[11px] text-[15px]';
  return (
    <Card className="flex flex-col gap-6 px-6 py-8 md:px-9">
      <CardHeader title="Model accuracy" note={`${m.dataset} · ${m.sampleCount.toLocaleString()} windows`} />
      <dl className="grid grid-cols-2 gap-6 md:grid-cols-4">
        <Stat label="Accuracy" value={formatPercent(m.accuracy)} />
        <Stat label="Precision" value={formatPercent(m.macroPrecision)} note="Average over classes" />
        <Stat label="Recall" value={formatPercent(m.macroRecall)} note="Average over classes" />
        <Stat label="F1 score" value={formatPercent(m.macroF1)} note="Average over classes" />
      </dl>
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] border-collapse">
            <thead>
              <tr>
                <th className={`${head} pr-2 text-left`}>Activity</th>
                <th className={`${head} px-2 text-right`}>Precision</th>
                <th className={`${head} px-2 text-right`}>Recall</th>
                <th className={`${head} px-2 text-right`}>F1</th>
                <th className={`${head} pl-2 text-right`}>Samples</th>
              </tr>
            </thead>
            <tbody>
              {m.perClass.map((c) => (
                <tr key={c.label}>
                  <td className={`${cell} pr-2`}>
                    <ActivityName label={c.label} />
                  </td>
                  <td className={`${cell} px-2 text-right`}>{formatPercent(c.precision)}</td>
                  <td className={`${cell} px-2 text-right`}>{formatPercent(c.recall)}</td>
                  <td className={`${cell} px-2 text-right`}>{formatPercent(c.f1)}</td>
                  <td className={`${cell} pl-2 text-right text-muted`}>{c.support.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ConfusionMatrix matrix={m.confusionMatrix} labels={labels} />
      </div>
    </Card>
  );
}

/** Rows = actual class, columns = predicted class; cell shade = share of that row. */
function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-[12.7px] font-medium text-muted">Confusion matrix (rows: actual · columns: predicted)</p>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[3px] text-[12px]">
          <thead>
            <tr>
              <th />
              {labels.map((label) => (
                <th key={label} className="w-12 px-0.5 pb-1 align-bottom font-normal" title={getActivity(label).name}>
                  <ActivityDot label={label} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const total = row.reduce((sum, n) => sum + n, 0) || 1;
              return (
                <tr key={labels[i] ?? i}>
                  <th className="pr-2 text-left font-normal whitespace-nowrap">
                    {labels[i] ? <ActivityName label={labels[i]} /> : i}
                  </th>
                  {row.map((n, j) => {
                    const share = n / total;
                    return (
                      <td
                        key={j}
                        title={`${n} (${formatPercent(share)})`}
                        className="h-9 w-12 rounded-md text-center tabular-nums"
                        style={{
                          backgroundColor: `rgba(0, 102, 204, ${0.08 + share * 0.92})`,
                          color: share > 0.5 ? 'white' : '#1d1d1f',
                        }}
                      >
                        {formatPercent(share)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
