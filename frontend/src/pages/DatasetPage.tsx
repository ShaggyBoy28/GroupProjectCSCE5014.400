// Dataset page: upload the group's recordings and turn them into training data (Phase 2, Expanding the Dataset).
import { useCallback, useEffect, useId, useState } from 'react';

import { api, type DatasetClip, type DatasetJob, type DatasetOverview } from '../api';
import { ActivityName, Breadcrumb, Card, CardHeader, Page, PillButton, pillClassName, Stat, StatusMessage } from '../components';
import { getActivity } from '../lib';

const OTHER = '__other__';
const MIN_PEOPLE = 3; // leave-one-person-out cross-validation needs a few people per class
const VIDEO_TYPES = '.mp4,.mov,.m4v,.avi,.mkv,video/*';

const isActive = (job: DatasetJob) => job.status === 'queued' || job.status === 'processing';

export function DatasetPage() {
  const [data, setData] = useState<DatasetOverview | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(
    () =>
      api.getDataset().then(
        (overview) => {
          setData(overview);
          setLoadError(null);
        },
        (error: Error) => setLoadError(error.message),
      ),
    [],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // poll while clips are being processed
  const working = data?.jobs.some(isActive) ?? false;
  useEffect(() => {
    if (!working) return;
    const timer = window.setInterval(() => void refresh(), 1500);
    return () => window.clearInterval(timer);
  }, [working, refresh]);

  if (!data) {
    return (
      <Page>
        <StatusMessage
          title={loadError ? 'Could not load the dataset.' : 'Loading dataset…'}
          message={loadError ?? undefined}
          showHomeLink={!!loadError}
        />
      </Page>
    );
  }

  return (
    <Page className="gap-5">
      <Breadcrumb current="Dataset" />
      <div className="flex flex-col gap-2">
        <h1 className="text-[34px] font-bold tracking-[-0.9px]">Expand the dataset.</h1>
        <p className="max-w-[760px] text-[15px] leading-[1.5] text-muted">
          Upload recordings of one person doing one activity. Each video is run through the same pose detector the app
          uses and saved as training data for new classes and for the lab-versus-home comparison.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[3fr_2fr]">
        <AddClipsCard data={data} onAdded={refresh} />
        <JobsCard jobs={data.jobs} />
      </div>
      <SummaryCard data={data} />
      <ClipsCard clips={data.clips} busy={new Set(data.jobs.filter(isActive).map((j) => j.clipId))} onChange={refresh} />
      <NextStepCard data={data} />
    </Page>
  );
}

// ---------- Upload ----------

function AddClipsCard({ data, onAdded }: { data: DatasetOverview; onAdded: () => Promise<void> }) {
  const ids = { files: useId(), person: useId(), activity: useId(), other: useId(), environment: useId() };
  const [files, setFiles] = useState<File[]>([]);
  const [person, setPerson] = useState('');
  const [activity, setActivity] = useState('');
  const [otherActivity, setOtherActivity] = useState('');
  const [environment, setEnvironment] = useState('');
  const [uploading, setUploading] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [fileInputKey, setFileInputKey] = useState(0);

  const chosenActivity = activity === OTHER ? otherActivity.trim() : activity;
  const ready = files.length > 0 && person.trim() !== '' && chosenActivity !== '' && uploading === null;

  const onUpload = async () => {
    const failed: string[] = [];
    for (const [index, file] of files.entries()) {
      setUploading(`Uploading ${index + 1} of ${files.length}…`);
      try {
        await api.addDatasetClip(file, person, chosenActivity, environment);
      } catch (error) {
        failed.push(`${file.name}: ${(error as Error).message}`);
      }
      await onAdded();
    }
    setErrors(failed);
    setUploading(null);
    setFiles([]);
    setFileInputKey((k) => k + 1);
  };

  const field = 'h-11 w-full rounded-xl border border-black/10 bg-white px-3.5 text-[14px] focus:border-accent focus:outline-none';
  const label = 'text-[12.5px] font-medium text-muted';

  return (
    <Card className="flex flex-col gap-5 px-6 py-7 md:px-8">
      <CardHeader title="Add recordings" note="Same person and activity for every file" />
      <div className="flex flex-col gap-1.5">
        <label htmlFor={ids.files} className={label}>
          Videos
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <input
            key={fileInputKey}
            id={ids.files}
            type="file"
            multiple
            accept={VIDEO_TYPES}
            className="peer sr-only"
            disabled={uploading !== null}
            onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
          />
          <label
            htmlFor={ids.files}
            className={pillClassName(
              'secondary',
              'cursor-pointer peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent',
            )}
          >
            {files.length ? 'Choose other files…' : 'Choose videos…'}
          </label>
          <span className="min-w-0 truncate text-[13px] text-muted">
            {files.length === 0
              ? 'MP4 or MOV · whole body in view · 10–20 s each'
              : files.length === 1
                ? files[0]!.name
                : `${files.length} videos selected`}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <label htmlFor={ids.person} className={label}>
            Person
          </label>
          <input
            id={ids.person}
            className={field}
            list={`${ids.person}-list`}
            placeholder="e.g. josh"
            value={person}
            onChange={(e) => setPerson(e.target.value)}
          />
          <datalist id={`${ids.person}-list`}>
            {data.people.map((p) => (
              <option key={p} value={p} />
            ))}
          </datalist>
        </div>
        <div className="flex flex-col gap-1.5">
          <label htmlFor={ids.activity} className={label}>
            Activity
          </label>
          <select id={ids.activity} className={field} value={activity} onChange={(e) => setActivity(e.target.value)}>
            <option value="" disabled>
              Choose…
            </option>
            {data.activities.map((a) => (
              <option key={a} value={a}>
                {getActivity(a).name}
              </option>
            ))}
            <option value={OTHER}>Other (new class)…</option>
          </select>
        </div>
        {activity === OTHER && (
          <div className="flex flex-col gap-1.5">
            <label htmlFor={ids.other} className={label}>
              New activity name
            </label>
            <input
              id={ids.other}
              className={field}
              placeholder="e.g. squatting"
              value={otherActivity}
              onChange={(e) => setOtherActivity(e.target.value)}
            />
          </div>
        )}
        <div className="flex flex-col gap-1.5">
          <label htmlFor={ids.environment} className={label}>
            Room / setting <span className="font-normal">(optional)</span>
          </label>
          <input
            id={ids.environment}
            className={field}
            list={`${ids.environment}-list`}
            placeholder="e.g. living room"
            value={environment}
            onChange={(e) => setEnvironment(e.target.value)}
          />
          <datalist id={`${ids.environment}-list`}>
            {data.environments.map((env) => (
              <option key={env} value={env} />
            ))}
          </datalist>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-4">
        <PillButton onClick={onUpload} disabled={!ready}>
          {uploading ?? (files.length > 1 ? `Add ${files.length} videos` : 'Add video')}
        </PillButton>
        <p className="text-[12.5px] text-muted">One activity per video · camera still · one person in frame</p>
      </div>
      {errors.length > 0 && (
        <ul role="alert" className="flex flex-col gap-1 text-[13px] text-[#d70015]">
          {errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      )}
    </Card>
  );
}

// ---------- Processing queue ----------

function JobsCard({ jobs }: { jobs: DatasetJob[] }) {
  const recent = [...jobs].reverse().slice(0, 8);
  const waiting = jobs.filter(isActive).length;

  return (
    <Card className="flex flex-col gap-4 px-6 py-7 md:px-8">
      <CardHeader title="Processing" note={waiting ? `${waiting} in progress` : undefined} />
      {recent.length === 0 ? (
        <p className="text-[14px] leading-[1.5] text-muted">
          Uploaded videos are processed here one at a time, about a second per frame on a laptop CPU. You can keep
          uploading or leave the page while it works.
        </p>
      ) : (
        <ul className="flex flex-col gap-3.5">
          {recent.map((job) => (
            <JobRow key={job.id} job={job} />
          ))}
        </ul>
      )}
    </Card>
  );
}

function JobRow({ job }: { job: DatasetJob }) {
  const fraction = job.framesTotal ? job.framesDone / job.framesTotal : 0;
  const status =
    job.status === 'queued'
      ? 'Waiting…'
      : job.status === 'processing'
        ? `${job.framesDone} / ${job.framesTotal || '…'} frames`
        : job.status === 'done'
          ? `Done · ${job.windows ?? 0} training windows`
          : `Failed: ${job.error}`;

  return (
    <li className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3 text-[13px]">
        <span className="min-w-0 truncate">
          <ActivityName label={job.activity} /> <span className="text-muted">· {job.person}{job.environment !== 'unspecified' && ` · ${job.environment.replace(/-/g, ' ')}`}</span>
        </span>
        <span
          className={`shrink-0 ${job.status === 'error' || (job.status === 'done' && !job.windows) ? 'text-[#d70015]' : 'text-muted'}`}
        >
          {status}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-fill">
        <div
          className={`h-full rounded-full transition-[width] ${job.status === 'error' ? 'bg-[#d70015]' : 'bg-accent'}`}
          style={{ width: `${job.status === 'done' || job.status === 'error' ? 100 : Math.round(fraction * 100)}%` }}
        />
      </div>
      <span className="truncate text-[11.5px] text-muted">{job.file}</span>
    </li>
  );
}

// ---------- Totals ----------

function SummaryCard({ data }: { data: DatasetOverview }) {
  const { summary } = data;
  const maxWindows = Math.max(1, ...summary.perClass.map((c) => c.windows));

  return (
    <Card className="flex flex-col gap-6 px-6 py-7 md:px-8">
      <CardHeader title="Collected so far" note="Training windows are 32-frame slices, one every 6 frames" />
      <dl className="grid grid-cols-2 gap-5 md:grid-cols-4">
        <Stat label="Videos" value={summary.clips} />
        <Stat label="People" value={summary.people} />
        <Stat label="Rooms / settings" value={summary.environments} />
        <Stat label="Training windows" value={summary.windows.toLocaleString()} />
      </dl>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-left text-[13.5px]">
          <thead className="text-[12px] text-muted">
            <tr className="border-b border-fill">
              <th className="py-2 font-medium">Activity</th>
              <th className="py-2 font-medium">Videos</th>
              <th className="py-2 font-medium">People</th>
              <th className="w-[40%] py-2 font-medium">Training windows</th>
            </tr>
          </thead>
          <tbody>
            {summary.perClass.map((row) => {
              const fewPeople = row.clips > 0 && row.people < MIN_PEOPLE;
              return (
                <tr key={row.label} className="border-b border-fill last:border-0">
                  <td className="py-2.5">
                    <span className="inline-flex items-center gap-2">
                      <ActivityName label={row.label} />
                      {!row.inModel && (
                        <span className="rounded-full bg-fill px-2 py-0.5 text-[11px] text-muted">new class</span>
                      )}
                    </span>
                  </td>
                  <td className="py-2.5">{row.clips}</td>
                  <td className={`py-2.5 ${fewPeople ? 'text-[#b45309]' : ''}`} title={fewPeople ? `Aim for at least ${MIN_PEOPLE} people` : undefined}>
                    {row.people}
                    {fewPeople && ` (aim for ${MIN_PEOPLE}+)`}
                  </td>
                  <td className="py-2.5">
                    <div className="flex items-center gap-3">
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-fill">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${(row.windows / maxWindows) * 100}%`, backgroundColor: getActivity(row.label).color }}
                        />
                      </div>
                      <span className="w-12 text-right tabular-nums">{row.windows}</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[12.5px] leading-[1.5] text-muted">
        For reference, each original class has about 3,770 training windows from the Berkeley MHAD lab data. Recording the
        original six activities too lets the model tell activities apart rather than rooms, and gives the lab-versus-home
        comparison.
      </p>
    </Card>
  );
}

// ---------- Clip list ----------

function ClipsCard({ clips, busy, onChange }: { clips: DatasetClip[]; busy: Set<string>; onChange: () => Promise<void> }) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sorted = [...clips].sort((a, b) => a.activity.localeCompare(b.activity) || a.person.localeCompare(b.person));

  const onDelete = async (id: string) => {
    setError(null);
    try {
      await api.deleteDatasetClip(id);
    } catch (e) {
      setError((e as Error).message);
    }
    setConfirming(null);
    await onChange();
  };

  return (
    <Card className="flex flex-col gap-4 px-6 py-7 md:px-8">
      <CardHeader title="Videos" note="The picture is the first frame where a person was found" />
      {sorted.length === 0 ? (
        <p className="text-[14px] text-muted">No videos yet. Add some above.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-[13.5px]">
            <thead className="text-[12px] text-muted">
              <tr className="border-b border-fill">
                <th className="py-2 font-medium" />
                <th className="py-2 font-medium">Activity</th>
                <th className="py-2 font-medium">Person</th>
                <th className="py-2 font-medium">Room</th>
                <th className="py-2 font-medium">Take</th>
                <th className="py-2 font-medium">Length</th>
                <th className="py-2 font-medium">Person found</th>
                <th className="py-2 font-medium">Windows</th>
                <th className="py-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((clip) => {
                const found = clip.framesUsed ? 1 - (clip.framesNoPerson ?? 0) / clip.framesUsed : null;
                return (
                  <tr key={clip.id} className="border-b border-fill last:border-0">
                    <td className="py-2 pr-3">
                      <img
                        src={api.datasetPreviewUrl(clip.id)}
                        alt=""
                        loading="lazy"
                        className="h-12 w-16 rounded-md bg-fill object-cover"
                      />
                    </td>
                    <td className="py-2">
                      <ActivityName label={clip.activity} />
                    </td>
                    <td className="py-2">{clip.person}</td>
                    <td className="py-2 text-muted">{clip.environment === 'unspecified' ? '—' : clip.environment.replace(/-/g, ' ')}</td>
                    <td className="max-w-[140px] truncate py-2 text-muted" title={clip.file}>
                      {clip.take}
                    </td>
                    <td className="py-2 tabular-nums">{clip.durationS == null ? '—' : `${clip.durationS} s`}</td>
                    <td className={`py-2 tabular-nums ${found != null && found < 0.8 ? 'text-[#b45309]' : ''}`}>
                      {found == null ? '—' : `${Math.round(found * 100)}%`}
                    </td>
                    <td className={`py-2 tabular-nums ${clip.windows ? '' : 'text-[#d70015]'}`} title={clip.windows ? undefined : 'Too short: under 32 frames with a person'}>
                      {clip.windows ?? '—'}
                      {clip.windows === 0 && ' (too short)'}
                    </td>
                    <td className="py-2 text-right">
                      {confirming === clip.id ? (
                        <span className="inline-flex gap-3 text-[12.5px]">
                          <button type="button" className="text-[#d70015] hover:underline" onClick={() => void onDelete(clip.id)}>
                            Delete video
                          </button>
                          <button type="button" className="text-muted hover:underline" onClick={() => setConfirming(null)}>
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button
                          type="button"
                          className="text-[12.5px] text-muted hover:text-[#d70015] disabled:opacity-40"
                          disabled={busy.has(clip.id)}
                          onClick={() => setConfirming(clip.id)}
                        >
                          Remove
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {error && (
        <p role="alert" className="text-[13px] text-[#d70015]">
          {error}
        </p>
      )}
    </Card>
  );
}

// ---------- What to do next ----------

function NextStepCard({ data }: { data: DatasetOverview }) {
  const command = `python scripts/train_model.py cv --extra ${data.indexPath} --epochs 60 --lr 1e-3`;
  return (
    <Card className="flex flex-col gap-3 px-6 py-7 md:px-8">
      <CardHeader title="Next: train and evaluate" />
      <p className="text-[14px] leading-[1.5] text-muted">
        Once each activity has videos from a few people, run cross-validation from the project folder. It holds out
        each person in turn and reports accuracy per activity and per room. See docs/DATASET_EXPANSION.md.
      </p>
      <code className="overflow-x-auto rounded-xl bg-canvas px-4 py-3 text-[12.5px] whitespace-nowrap">{command}</code>
    </Card>
  );
}
