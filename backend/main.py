import os
import cv2
import numpy as np
import base64
import traceback
import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import preprocess_frame, draw_box

# Configure logging for production visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("motion_tracker_backend")

app = FastAPI(
    title="Video Feature Tracking & Motion Analysis API",
    description="FastAPI service for tracking a region of interest (ROI) using Lucas-Kanade Optical Flow."
)

@app.post("/track")
async def track_motion(
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
    """
    HTTP POST endpoint that receives a video file and tracking parameters,
    runs Lucas-Kanade optical flow, and returns base64 encoded processed frames
    along with frame-by-frame velocity calculations.
    """
    temp_path = None
    try:
        # 📥 Save uploaded video to a unique temporary file to handle concurrency safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            contents = await video.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        logger.info(f"Received tracking request. Temp file created at: {temp_path}")

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("Failed to open video file with OpenCV.")
            return JSONResponse(status_code=400, content={"error": "Could not open video file"})

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            fps = 30.0  # fallback
            logger.warning(f"Video FPS is invalid or zero. Falling back to default: {fps}")
        if total_frames <= 0:
            logger.error("Video contains zero or unreadable frames.")
            return JSONResponse(status_code=400, content={"error": "Could not read video frames"})

        # 🎞 Frame range handling
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        if start_frame > end_frame:
            start_frame = end_frame

        logger.info(f"Tracking frames from {start_frame} to {end_frame} (Total frames: {total_frames}, FPS: {fps})")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        success, frame = cap.read()
        if not success:
            logger.warning("Could not read the first frame in the range.")
            cap.release()
            return {"frames": [], "speeds": []}

        frame_h, frame_w = frame.shape[:2]
        roi_x = max(0, min(roi_x, frame_w - 1))
        roi_y = max(0, min(roi_y, frame_h - 1))
        roi_w = max(1, min(roi_w, frame_w - roi_x))
        roi_h = max(1, min(roi_h, frame_h - roi_y))

        # 🎯 Detect initial tracking points in ROI
        prev_gray = preprocess_frame(frame, blur=blur, equalize=equalize, edges=edges)
        mask = np.zeros_like(prev_gray)
        mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255

        # Detect good corners to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)
        if p0 is None or len(p0) == 0:
            # Fallback to center point if no corners are found
            p0 = np.array([[[roi_x + roi_w / 2, roi_y + roi_h / 2]]], dtype=np.float32)

        output_frames_b64 = []
        speeds = []

        prev_points = p0
        prev_center = np.mean(prev_points.reshape(-1, 2), axis=0)
        speeds.append(0.0)

        # Process first display frame
        first_display = preprocess_frame(frame, blur=blur, equalize=equalize, edges=edges) if blur or equalize or edges else frame
        display_frame = cv2.cvtColor(first_display, cv2.COLOR_GRAY2BGR) if len(first_display.shape) == 2 else first_display
        display_frame = draw_box(display_frame, roi_x, roi_y, roi_w, roi_h)
        _, buffer = cv2.imencode(".jpg", display_frame)
        output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

        # 🔁 Frame-by-frame tracking
        # We read sequentially to maintain accurate and smooth optical flow tracking.
        for frame_index in range(start_frame + 1, end_frame + 1):
            success, frame = cap.read()
            if not success:
                break

            curr_gray = preprocess_frame(frame, blur=blur, equalize=equalize, edges=edges)
            if prev_points is None or len(prev_points) == 0:
                break

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_points, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            if p1 is None or st is None:
                break

            good_new = p1[st == 1]
            if len(good_new) == 0:
                break

            # Calculate center displacement
            current_center = np.mean(good_new.reshape(-1, 2), axis=0)
            dx, dy = current_center - prev_center
            dist_pixels = np.hypot(dx, dy)
            # Speed is distance in pixels times FPS (pixels/sec)
            speeds.append(float(dist_pixels * fps))

            # Move ROI box corresponding to tracking translation
            roi_x = int(max(0, min(frame_w - roi_w, roi_x + dx)))
            roi_y = int(max(0, min(frame_h - roi_h, roi_y + dy)))

            curr_display = preprocess_frame(frame, blur=blur, equalize=equalize, edges=edges) if blur or equalize or edges else frame
            display_frame = cv2.cvtColor(curr_display, cv2.COLOR_GRAY2BGR) if len(curr_display.shape) == 2 else curr_display
            display_frame = draw_box(display_frame, roi_x, roi_y, roi_w, roi_h)

            _, buffer = cv2.imencode(".jpg", display_frame)
            output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

            # Prepare for next frame
            prev_gray = curr_gray.copy()
            prev_points = good_new.reshape(-1, 1, 2)
            prev_center = current_center

        cap.release()
        logger.info("Optical flow tracking completed successfully.")
        return {"frames": output_frames_b64, "speeds": speeds}

    except Exception as e:
        logger.error("❌ SERVER ERROR:")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary video file: {temp_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
