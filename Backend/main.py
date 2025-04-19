import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

@app.post("/track")
def track_motion(
    video: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_w: int = Form(...),
    roi_h: int = Form(...),
    blur: bool = Form(False),
    equalize: bool = Form(False),
    edges: bool = Form(False)
):
    # Save uploaded video to a temporary file
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video.file.read())
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # default FPS if not available
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0

    # Compute frame indices for the selected time range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    if end_frame == 0 or end_frame > total_frames - 1:
        end_frame = total_frames - 1
    if start_frame < 0:
        start_frame = 0
    if start_frame > end_frame:
        start_frame = end_frame

    # Initialize video processing at the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success, frame = cap.read()
    if not success:
        cap.release()
        return {"frames": [], "speeds": []}

    frame_h, frame_w = frame.shape[0], frame.shape[1]
    # Clamp ROI within frame boundaries
    roi_x = max(0, min(roi_x, frame_w - 1))
    roi_y = max(0, min(roi_y, frame_h - 1))
    roi_w = max(1, min(roi_w, frame_w - roi_x))
    roi_h = max(1, min(roi_h, frame_h - roi_y))

    # Preprocessing function for a frame (applied to grayscale)
    def preprocess_frame(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if equalize:
            gray = cv2.equalizeHist(gray)
        if edges:
            gray = cv2.Canny(gray, 50, 150)
        return gray

    # Prepare the first frame
    prev_gray = preprocess_frame(frame)
    # Detect feature points (corners) within the ROI of the first frame
    mask = np.zeros_like(prev_gray)
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255  # ROI mask
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if p0 is None or len(p0) == 0:
        # If no corners found, fall back to using the center of the ROI as a single point
        p0 = np.array([[[roi_x + roi_w / 2, roi_y + roi_h / 2]]], dtype=np.float32)

    # Lists to collect output frames and speeds
    output_frames_b64 = []
    speeds = []

    # Calculate initial center of tracked points (for frame 0)
    prev_points = p0
    prev_center = np.mean(prev_points.reshape(-1, 2), axis=0)
    speeds.append(0.0)  # speed is zero at the first frame (no previous movement)

    # Draw bounding box on the first frame for output
    if blur or equalize or edges:
        # Convert processed grayscale (or edge) image to 3-channel for drawing
        if len(prev_gray.shape) == 2:
            display_frame = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = prev_gray.copy()
    else:
        display_frame = frame.copy()
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
    _, buffer = cv2.imencode(".jpg", display_frame)
    output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

    # Process subsequent frames in the range
    for frame_index in range(start_frame + 1, end_frame + 1):
        success, frame = cap.read()
        if not success:
            break
        curr_gray = preprocess_frame(frame)

        # If no points to track (lost tracking), stop early
        if prev_points is None or len(prev_points) == 0:
            break

        # Calculate optical flow to track points in the new frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, 
                                              winSize=(15, 15), maxLevel=2,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if p1 is None:
            break

        # Select the successfully tracked points
        good_new = p1[st == 1]
        good_old = prev_points[st == 1]
        if len(good_new) == 0:
            break

        # Compute the centroid of the tracked points in the current frame
        current_center = np.mean(good_new.reshape(-1, 2), axis=0)
        # Calculate speed (pixels/second) based on centroid movement
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        dist_pixels = np.sqrt(dx**2 + dy**2)
        speed_px_per_s = dist_pixels * fps
        speeds.append(float(speed_px_per_s))

        # Update ROI top-left position by the movement (for visualization)
        roi_x = int(max(0, min(frame_w - roi_w, roi_x + dx)))
        roi_y = int(max(0, min(frame_h - roi_h, roi_y + dy)))

        # Draw bounding box on the current frame
        if blur or equalize or edges:
            # Use processed frame for display (grayscale or edges to BGR)
            if len(curr_gray.shape) == 2:
                display_frame = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = curr_gray.copy()
        else:
            display_frame = frame.copy()
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

        # Encode the frame to base64
        _, buffer = cv2.imencode(".jpg", display_frame)
        output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

        # Prepare for next iteration
        prev_gray = curr_gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        prev_center = current_center

    cap.release()
    # Clean up temporary video file
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return {"frames": output_frames_b64, "speeds": speeds}