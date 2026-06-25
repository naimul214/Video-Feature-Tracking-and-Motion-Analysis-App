import os
import cv2
import numpy as np
import base64
import traceback
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

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
    try:
        # 📥 Save uploaded video to a temporary file
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video.file.read())

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            fps = 30.0  # fallback
        if total_frames <= 0:
            return JSONResponse(status_code=400, content={"error": "Could not read video frames"})

        # 🎞 Frame range handling
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        if start_frame > end_frame:
            start_frame = end_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        success, frame = cap.read()
        if not success:
            cap.release()
            return {"frames": [], "speeds": []}

        frame_h, frame_w = frame.shape[:2]
        roi_x = max(0, min(roi_x, frame_w - 1))
        roi_y = max(0, min(roi_y, frame_h - 1))
        roi_w = max(1, min(roi_w, frame_w - roi_x))
        roi_h = max(1, min(roi_h, frame_h - roi_y))

        # 🧼 Preprocessing function
        def preprocess_frame(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if blur:
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if equalize:
                gray = cv2.equalizeHist(gray)
            if edges:
                gray = cv2.Canny(gray, 50, 150)
            return gray

        # 🎯 Detect initial tracking points in ROI
        prev_gray = preprocess_frame(frame)
        mask = np.zeros_like(prev_gray)
        mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)
        if p0 is None or len(p0) == 0:
            p0 = np.array([[[roi_x + roi_w / 2, roi_y + roi_h / 2]]], dtype=np.float32)

        output_frames_b64 = []
        speeds = []

        prev_points = p0
        prev_center = np.mean(prev_points.reshape(-1, 2), axis=0)
        speeds.append(0.0)

        def draw_box(img, x, y, w, h):
            return cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)

        first_display = preprocess_frame(frame) if blur or equalize or edges else frame
        display_frame = cv2.cvtColor(first_display, cv2.COLOR_GRAY2BGR) if len(first_display.shape) == 2 else first_display
        display_frame = draw_box(display_frame, roi_x, roi_y, roi_w, roi_h)
        _, buffer = cv2.imencode(".jpg", display_frame)
        output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

        # 🔁 Frame-by-frame tracking
        for frame_index in range(start_frame + 1, end_frame + 1, 3):
            success, frame = cap.read()
            if not success:
                break

            curr_gray = preprocess_frame(frame)
            if prev_points is None or len(prev_points) == 0:
                break

            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_points, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            if p1 is None or st is None:
                break

            good_new = p1[st == 1]
            good_old = prev_points[st == 1]
            if len(good_new) == 0:
                break

            current_center = np.mean(good_new.reshape(-1, 2), axis=0)
            dx, dy = current_center - prev_center
            dist_pixels = np.hypot(dx, dy)
            speeds.append(float(dist_pixels * fps))

            roi_x = int(max(0, min(frame_w - roi_w, roi_x + dx)))
            roi_y = int(max(0, min(frame_h - roi_h, roi_y + dy)))

            curr_display = preprocess_frame(frame) if blur or equalize or edges else frame
            display_frame = cv2.cvtColor(curr_display, cv2.COLOR_GRAY2BGR) if len(curr_display.shape) == 2 else curr_display
            display_frame = draw_box(display_frame, roi_x, roi_y, roi_w, roi_h)

            _, buffer = cv2.imencode(".jpg", display_frame)
            output_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

            prev_gray = curr_gray.copy()
            prev_points = good_new.reshape(-1, 1, 2)
            prev_center = current_center

        cap.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return {"frames": output_frames_b64, "speeds": speeds}

    except Exception as e:
        print("❌ SERVER ERROR:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
