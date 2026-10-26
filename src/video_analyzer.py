import json
import os
import subprocess

import cv2
import ntpath
from .utils import filter_persons, draw_keypoints
from .lstm import WINDOW_SIZE
import time

import numpy as np
import torch
import torch.nn.functional as F

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

# how many frames to skip while inferencing
# configuring a higher value will result in better FPS (frames per rate), but accuracy might get impacted
SKIP_FRAME_COUNT = 0

def probe_video(video_path):
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": round(fps, 2),
        "frameCount": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "durationS": round(frame_count / fps, 3) if fps else 0,
    }
    cap.release()
    return info


def transcode_for_browser(src_path, dst_path):
    """Re-encode OpenCV's mp4v output as H.264 so browsers can play it.

    Uses the ffmpeg binary bundled with imageio-ffmpeg. Returns False (and
    leaves src_path untouched) if ffmpeg isn't available or fails.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return False
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", src_path, "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", "-an", dst_path]
    return subprocess.run(command, check=False).returncode == 0


def analyse_video_events(pose_detector, lstm_classifier, video_path):
    """Analyse a video for the React frontend.

    Yields progress events while it works, then writes next to the input:
      res_<name>.mp4   annotated video (H.264 when ffmpeg is available)
      res_<name>.json  every LSTM prediction with all 6 class probabilities
    and finally yields {"type": "done"}.
    """
    info = probe_video(video_path)
    folder, file_name = os.path.split(os.path.abspath(video_path))
    result_path = os.path.join(folder, 'res_{}'.format(file_name))
    raw_path = os.path.join(folder, 'res_raw_{}'.format(file_name))
    data_path = os.path.join(folder, 'res_{}.json'.format(os.path.splitext(file_name)[0]))

    cap = cv2.VideoCapture(video_path)
    tot_frames = max(info["frameCount"], 1)
    vid_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), info["fps"],
                                 (info["width"], info["height"]))
    buffer_window = []
    predictions = []
    label = None
    counter = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        if counter % (SKIP_FRAME_COUNT + 1) == 0:
            outputs = pose_detector(frame)
            persons, _ = filter_persons(outputs)
            if len(persons) >= 1:
                # only the first person detected is analysed (same as the original app)
                p = persons[0]
                draw_keypoints(p, img)
                features = []
                for row in p:
                    features.append(row[0])
                    features.append(row[1])
                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    model_input = torch.unsqueeze(torch.Tensor(np.array(buffer_window, dtype=np.float32)), dim=0)
                    with torch.no_grad():
                        prob = F.softmax(lstm_classifier(model_input), dim=1)[0]
                    probabilities = [float(x) for x in prob]
                    pred_index = int(np.argmax(probabilities))
                    label = LABELS[pred_index]
                    predictions.append({"frame": counter, "probabilities": probabilities})
                    buffer_window.pop(0)
                    buffer_window.append(features)
        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label), (int(info["width"] - 400), info["height"] - 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        counter += 1
        vid_writer.write(img)
        yield {
            "type": "progress",
            "percent": min(int(counter * 100 / tot_frames), 100),
            "frame": counter,
            "totalFrames": info["frameCount"],
            "label": label,
        }
    cap.release()
    vid_writer.release()

    if transcode_for_browser(raw_path, result_path):
        os.remove(raw_path)
    else:
        os.replace(raw_path, result_path)
    with open(data_path, 'w', encoding='utf-8') as data_file:
        json.dump({"video": {"name": file_name, **info}, "windowSize": WINDOW_SIZE,
                   "predictions": predictions}, data_file)
    print("Video processing finished in ", time.time() - start)
    yield {"type": "done"}


# analyse the video
def analyse_video(pose_detector, lstm_classifier, video_path):
    # open the video
    cap = cv2.VideoCapture(video_path)
    # width of image frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height of image frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames per second of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total number of frames in the video
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video output codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # extract the file name from video path
    file_name = ntpath.basename(video_path)
    # video writer
    vid_writer = cv2.VideoWriter('res_{}'.format(
        file_name), fourcc, 30, (width, height))
    # counter
    counter = 0
    # buffer to keep the output of detectron2 pose estimation
    buffer_window = []
    # start time
    start = time.time()
    label = None
    # iterate through the video
    while True:
        # read the frame
        ret, frame = cap.read()
        # return if end of the video
        if ret == False:
            break
        # make a copy of the frame
        img = frame.copy()
        if(counter % (SKIP_FRAME_COUNT+1) == 0):
            # predict pose estimation on the frame
            outputs = pose_detector(frame)
            # filter the outputs with a good confidence score
            persons, pIndicies = filter_persons(outputs)
            if len(persons) >= 1:
                # pick only pose estimation results of the first person.
                # actually, we expect only one person to be present in the video.
                p = persons[0]
                # draw the body joints on the person body
                draw_keypoints(p, img)
                # input feature array for lstm
                features = []
                # add pose estimate results to the feature array
                for i, row in enumerate(p):
                    features.append(row[0])
                    features.append(row[1])

                # append the feature array into the buffer
                # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    # convert input to tensor
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    # add extra dimension
                    model_input = torch.unsqueeze(model_input, dim=0)
                    # predict the action class using lstm
                    y_pred = lstm_classifier(model_input)
                    prob = F.softmax(y_pred, dim=1)
                    # get the index of the max probability
                    pred_index = prob.data.max(dim=1)[1]
                    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]
                    #print("Label detected ", label)

        # add predicted label into the frame
        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label),
                        (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        # increment counter
        counter += 1
        # write the frame into the result video
        vid_writer.write(img)
        # compute the completion percentage
        percentage = int(counter*100/tot_frames)
        # return the completion percentage
        yield "data:" + str(percentage) + "\n\n"
    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)


def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        out_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +
                  out_frame + b'\r\n')
        yield result
    print("finished video streaming")
