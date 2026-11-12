import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, DEFAULT_MODEL } from '../api';
import uploadIconUrl from '../assets/icons/upload.svg';
import { ActivityLegend, Card, Page, PillButton, PillLink, pillClassName } from '../components';
import { formatSampleSummary, useApiResource } from '../lib';

const SAMPLE_VIDEO = 'sample_video.mp4';

export function HomePage() {
  // the model's classes, so a retrained model with more activities shows up here automatically
  const model = useApiResource(api.getModel, 'model');
  const { labels, windowSize } = model.status === 'ready' ? model.data : DEFAULT_MODEL;
  const steps = [
    { title: 'Find the pose', body: 'Detectron2 locates 17 body keypoints in every frame.' },
    { title: `Collect ${windowSize} frames`, body: 'About one second of movement becomes one input.' },
    { title: 'Classify', body: `The LSTM scores ${labels.length} activities. The highest score is the label.` },
  ];

  return (
    <Page className="gap-10">
      <div className="hidden h-[26px] md:block" />
      <div className="flex flex-col items-center gap-3.5 pt-[7px] text-center">
        <h1 className="text-[40px] leading-[1.08] font-bold tracking-[-1.68px] md:text-[54.5px] md:leading-[58.8px]">
          Human activity,
          <br />
          detected.
        </h1>
        <p className="max-w-[620px] px-[22px] text-[17px] leading-[1.5] text-muted md:text-[19.7px] md:leading-[29.4px]">
          Upload a short clip of one person. The app reads their pose in every frame and an LSTM names what
          they&apos;re doing.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <UploadCard />
        <SampleCard />
      </div>
      <Card className="flex flex-col items-center gap-4 px-8 py-8 text-center md:flex-row md:justify-between md:text-left">
        <div className="flex flex-col gap-1">
          <h2 className="text-[20px] font-semibold">Classify live with your camera</h2>
          <p className="max-w-[620px] text-[14px] leading-6 text-muted">
            Turn on your camera to see pose-based activity predictions while you move.
          </p>
        </div>
        <PillLink to="/live">Start live camera</PillLink>
      </Card>
      <Card className="flex flex-col gap-7 px-6 py-9 md:px-10">
        <h2 className="text-[27px] font-bold tracking-[-0.56px]">How it works.</h2>
        <ol className="grid gap-8 md:grid-cols-3 md:gap-10">
          {steps.map((step, index) => (
            <li key={step.title} className="flex flex-col">
              <span className="text-[13px] font-semibold text-muted">0{index + 1}</span>
              <h3 className="mt-[7px] text-[19px] font-semibold tracking-[-0.19px]">{step.title}</h3>
              <p className="mt-1.5 max-w-[320px] text-[14px] leading-[21.75px] text-muted">{step.body}</p>
            </li>
          ))}
        </ol>
        <ActivityLegend labels={labels} className="border-t border-fill pt-[21px] text-[13.5px]" />
      </Card>
    </Page>
  );
}

function UploadCard() {
  const inputId = useId();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wrongType = file !== null && !file.name.toLowerCase().endsWith('.mp4');
  const message = wrongType ? 'Only .mp4 files are supported' : error;

  const onAnalyze = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const { name } = await api.uploadVideo(file);
      navigate(`/analyzing/${encodeURIComponent(name)}`);
    } catch (uploadError) {
      setError((uploadError as Error).message);
      setUploading(false);
    }
  };

  return (
    <Card className="flex min-h-[444px] flex-col items-center justify-center gap-[18px] px-8 py-16 text-center">
      <img src={uploadIconUrl} width={48} height={48} alt="" />
      <div className="flex flex-col gap-1.5">
        <h2 className="text-[24px] font-semibold tracking-[-0.24px]">Upload a video</h2>
        <p className="text-[14px] text-muted">MP4 only · one person, whole body in view</p>
      </div>
      <input
        id={inputId}
        type="file"
        accept="video/mp4,.mp4"
        className="peer sr-only"
        disabled={uploading}
        onChange={(event) => {
          setError(null);
          setFile(event.target.files?.[0] ?? null);
        }}
      />
      <label
        htmlFor={inputId}
        className={pillClassName(
          'secondary',
          'cursor-pointer peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent',
        )}
      >
        {file ? 'Choose another file…' : 'Choose file…'}
      </label>
      {file && <p className="-mt-2 max-w-full truncate text-[13px] text-muted">{file.name}</p>}
      <PillButton onClick={onAnalyze} disabled={!file || uploading || wrongType}>
        {uploading ? 'Uploading…' : 'Analyze'}
      </PillButton>
      {message && (
        <p role="alert" className="text-[13px] text-[#d70015]">
          {message}
        </p>
      )}
    </Card>
  );
}

function SampleCard() {
  const sample = useApiResource(api.getSample, SAMPLE_VIDEO);

  return (
    <Card className="flex flex-col overflow-hidden">
      {/* #t=0.1 makes browsers paint the first frame without playing */}
      <video
        src={`${api.videoFileUrl(SAMPLE_VIDEO)}#t=0.1`}
        preload="metadata"
        muted
        playsInline
        aria-label="First frame of the sample video"
        className="h-[250px] w-full bg-black object-cover"
      />
      <div className="flex items-center justify-between gap-4 px-7 py-6">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-[19.7px] font-semibold tracking-[-0.2px]">Try the sample</h2>
          <p className="text-[13.3px] text-muted">
            {sample.status === 'ready' ? formatSampleSummary(sample.data) : ' '}
          </p>
        </div>
        <PillLink to={`/analyzing/${SAMPLE_VIDEO}`}>Analyze sample</PillLink>
      </div>
    </Card>
  );
}
