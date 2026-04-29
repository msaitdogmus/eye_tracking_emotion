# Eye Tracking + Emotion Analysis

Real-time and video-based eye tracking, gaze direction, off-screen detection, no-face detection, and emotion analysis using MediaPipe, OpenCV, and EmotiEffLib.

## Features

- Real-time webcam gaze tracking
- Video-based gaze analysis ( face_landmarker.task must be ! )
- Gaze direction classification
- NO_FACE detection
- OFF-SCREEN detection
- Emotion analysis
- Annotated output video export
- JSON report export
- Calibration-based gaze mapping
- Head-pose fallback logic

## Project Files

- gaze_core.py
- gaze_live.py
- gaze_video.py
- main.py

## Installation

```bash
conda create -n Mediapipe_Gaze_Estimation python=3.11 -y
conda activate Mediapipe_Gaze_Estimation
python -m pip install --upgrade pip setuptools wheel
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install mediapipe opencv-python numpy pillow onnx onnxruntime emotiefflib
```

## Live Mode

```bash
python main.py --no-emotion
python main.py
```

### Live Controls

- N = new calibration
- R = reset calibration
- ESC = exit

## Video Mode

```bash
python gaze_video.py --video "C:\path\to\video.mp4"
python gaze_video.py --video "C:\path\to\video.mp4" --output "annotated_output.mp4"
python gaze_video.py --video "C:\path\to\video.mp4" --report "report.json"
python gaze_video.py --video "C:\path\to\video.mp4" --output "annotated_output.mp4" --report "report.json"
```

### Video Controls

- SPACE = pause .. resume
- F = flip
- R = restart
- ESC = exit

## Notes

- Best performance is achieved after "CALİBRATİON" .........
- NO_FACE means no face was detected in the frame.
- OFF-SCREEN means the face is visible but attention is outside the screen.
- If calibration is unavailable, the system uses head-pose fallback logic.
