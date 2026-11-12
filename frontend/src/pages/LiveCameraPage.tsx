import { useEffect, useRef, useState } from 'react';

import { api, type LiveFrameResult } from '../api';
import { ActivityName, Breadcrumb, Card, Page, PillButton } from '../components';

const FRAME_WIDTH = 640;
const FRAME_INTERVAL_MS = 120;
const JPEG_QUALITY = 0.75;

export function LiveCameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const activeRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const [starting, setStarting] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [windowSize, setWindowSize] = useState(32);
  const [prediction, setPrediction] = useState<Pick<LiveFrameResult, 'label' | 'confidence'> | null>(null);
  const [posesBuffered, setPosesBuffered] = useState(0);
  const [personDetected, setPersonDetected] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const stopCamera = async () => {
    activeRef.current = false;
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsLive(false);
    setStarting(false);

    const sessionId = sessionIdRef.current;
    sessionIdRef.current = null;
    if (sessionId) {
      try {
        await api.stopLiveSession(sessionId);
      } catch (stopError) {
        setError((stopError as Error).message);
      }
    }
  };

  useEffect(
    () => () => {
      activeRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      const sessionId = sessionIdRef.current;
      if (sessionId) void api.stopLiveSession(sessionId).catch(() => undefined);
    },
    [],
  );

  const startCamera = async () => {
    setError(null);
    setPrediction(null);
    setPosesBuffered(0);
    setPersonDetected(true);
    setStarting(true);
    activeRef.current = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: FRAME_WIDTH }, height: { ideal: 480 } },
      });
      if (!activeRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview is unavailable.');
      video.srcObject = stream;
      await video.play();

      const session = await api.startLiveSession();
      sessionIdRef.current = session.sessionId;
      if (!activeRef.current) {
        sessionIdRef.current = null;
        await api.stopLiveSession(session.sessionId);
        return;
      }
      setWindowSize(session.windowSize);
      setStarting(false);
      setIsLive(true);

      const sendNextFrame = async () => {
        if (!activeRef.current) return;
        const currentVideo = videoRef.current;
        const canvas = canvasRef.current;
        const sessionId = sessionIdRef.current;
        if (!currentVideo || !canvas || !sessionId || !currentVideo.videoWidth) {
          timerRef.current = window.setTimeout(() => void sendNextFrame(), FRAME_INTERVAL_MS);
          return;
        }

        try {
          const scale = Math.min(1, FRAME_WIDTH / currentVideo.videoWidth);
          canvas.width = Math.round(currentVideo.videoWidth * scale);
          canvas.height = Math.round(currentVideo.videoHeight * scale);
          const context = canvas.getContext('2d');
          if (!context) throw new Error('Could not prepare a camera frame.');
          context.drawImage(currentVideo, 0, 0, canvas.width, canvas.height);
          const frame = await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob(
              (blob) => (blob ? resolve(blob) : reject(new Error('Could not capture a camera frame.'))),
              'image/jpeg',
              JPEG_QUALITY,
            );
          });
          if (!activeRef.current) return;
          const result = await api.classifyLiveFrame(sessionId, frame);
          if (!activeRef.current) return;
          setPersonDetected(result.personDetected);
          setPosesBuffered(result.posesBuffered);
          setPrediction({ label: result.label, confidence: result.confidence });
        } catch (frameError) {
          if (activeRef.current) {
            setError((frameError as Error).message);
            await stopCamera();
            return;
          }
        }
        if (activeRef.current) {
          timerRef.current = window.setTimeout(() => void sendNextFrame(), FRAME_INTERVAL_MS);
        }
      };

      void sendNextFrame();
    } catch (cameraError) {
      if (activeRef.current) {
        activeRef.current = false;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;
        setStarting(false);
        setIsLive(false);
        setError((cameraError as Error).message);
      }
    }
  };

  let status: string;
  if (!isLive) {
    status = 'Start the camera when you are ready.';
  } else if (!personDetected) {
    status = 'No person detected. Move into view of the camera.';
  } else if (!prediction?.label) {
    status = `Collecting pose frames: ${posesBuffered} of ${windowSize}`;
  } else {
    status = 'Live prediction';
  }

  return (
    <Page className="gap-6">
      <Breadcrumb current="Live camera" />
      <div className="flex flex-col gap-2">
        <h1 className="text-[30px] font-bold tracking-[-0.8px]">Live activity classification</h1>
        <p className="max-w-[720px] text-[14px] leading-6 text-muted">
          Allow camera access to see predictions as you move. Frames are sent to this app&apos;s server for
          classification; no recording is saved.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(280px,1fr)]">
        <Card className="relative flex min-h-[300px] items-center justify-center overflow-hidden bg-black">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            aria-label="Live camera preview"
            className={`max-h-[70vh] w-full object-contain ${isLive ? '' : 'hidden'}`}
          />
          {!isLive && (
            <p className="px-8 text-center text-[14px] text-white/75">Camera preview appears here after you start.</p>
          )}
          {isLive && prediction?.label && (
            <div className="absolute left-4 top-4 rounded-full bg-black/70 px-4 py-2 text-[14px] font-semibold text-white">
              <ActivityName label={prediction.label} />
              {prediction.confidence !== null && ` · ${Math.round(prediction.confidence * 100)}%`}
            </div>
          )}
          <canvas ref={canvasRef} className="hidden" />
        </Card>

        <Card className="flex flex-col gap-6 px-7 py-8">
          <div className="flex flex-col gap-2">
            <p className="text-[12px] font-medium uppercase tracking-wide text-muted">Status</p>
            <p className="text-[18px] font-semibold" role="status">{status}</p>
            {isLive && (
              <p className="text-[13px] text-muted">
                The classifier needs a full pose window before it can predict.
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-3">
            {!isLive ? (
              <PillButton onClick={startCamera} disabled={starting}>
                {starting ? 'Starting camera…' : 'Start camera'}
              </PillButton>
            ) : (
              <PillButton variant="secondary" onClick={() => void stopCamera()}>
                Stop camera
              </PillButton>
            )}
          </div>
          {error && <p role="alert" className="text-[13px] text-[#d70015]">{error}</p>}
          <p className="border-t border-fill pt-4 text-[12px] leading-5 text-muted">
            For accurate poses, keep your full body visible and make sure the room is well lit. Camera access
            requires localhost or a secure HTTPS connection.
          </p>
        </Card>
      </div>
    </Page>
  );
}
