import os
import base64
import tempfile
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Video Feature Tracking and Motion Analysis App")

# File uploader for .mp4 videos
uploaded_file = st.file_uploader("Upload a video (MP4 format)", type=["mp4"])
if uploaded_file is not None:
    # Save uploaded video to a temporary file for OpenCV processing
    video_bytes = uploaded_file.read()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_bytes)
    video_path = tfile.name

    # Open video to get duration and a frame for ROI selection
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Frames per second (default to 30 if unavailable)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    duration = total_frames / fps if total_frames else 0

    # Slider to select start and end times (in seconds) for processing
    if duration == 0:
        duration = 0.0
    start_time, end_time = st.slider(
        "Select time range (seconds)", 
        0.0, float(duration), 
        (0.0, float(min(duration, 5.0)) if duration else 0.0),  # default to first 5 seconds or full duration if shorter
        step=0.1
    )

    # Preprocessing option checkboxes
    st.markdown("**Preprocessing options:**")
    blur = st.checkbox("Apply Gaussian Blur")
    equalize = st.checkbox("Apply Histogram Equalization")
    edges = st.checkbox("Apply Edge Detection")

    # Read the first frame at the start time for ROI selection
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    success, frame = cap.read()
    cap.release()
    if not success:
        st.error("Unable to read a frame from the video at the selected start time.")
    else:
        # Convert BGR frame (OpenCV) to RGB (for display) and show it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        st.write("**Step 1:** Click the frame to choose the top-left corner of the ROI to track.")
        coords = streamlit_image_coordinates(img, key="roi_select")
        if coords:
            roi_x = int(coords["x"])
            roi_y = int(coords["y"])
            st.write(f"Selected top-left ROI corner at (x={roi_x}, y={roi_y}).")

            # Inputs for ROI width and height
            max_w = frame.shape[1] - roi_x
            max_h = frame.shape[0] - roi_y
            roi_w = st.number_input("ROI width (px)", min_value=1, max_value=int(max_w), value=min(50, int(max_w)))
            roi_h = st.number_input("ROI height (px)", min_value=1, max_value=int(max_h), value=min(50, int(max_h)))

            # (Optional) Display the ROI on the frame for confirmation
            # preview_frame = frame_rgb.copy()
            # cv2.rectangle(preview_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)
            # st.image(preview_frame, caption="Selected ROI highlighted")

            # Button to start tracking
            if st.button("Run Tracking"):
                # Determine backend URL from environment or use default (docker-compose service name or localhost)
                backend_url = os.getenv("FASTAPI_URL", "http://backend:8000")
                endpoint = f"{backend_url}/track"

                # Prepare payload for backend request
                fields = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_w": roi_w,
                    "roi_h": roi_h,
                    "blur": str(blur),          # booleans sent as strings ("True"/"False")
                    "equalize": str(equalize),
                    "edges": str(edges)
                }
                files = {"video": video_bytes}

                # Call the FastAPI backend
                with st.spinner("Processing video..."):
                    try:
                        response = requests.post(endpoint, data=fields, files=files)
                        response.raise_for_status()
                    except Exception as e:
                        st.error(f"Failed to get a response from the backend: {e}")
                        st.stop()
                result = response.json()
                frames_b64 = result.get("frames", [])
                speeds = result.get("speeds", [])

                if not frames_b64:
                    st.error("No frames received. Please check the backend log for issues.")
                else:
                    # Store results in session state for interactive display
                    st.session_state["frames_b64"] = frames_b64
                    st.session_state["speeds"] = speeds
                    st.session_state["num_frames"] = len(frames_b64)
                    st.session_state["current_frame"] = 0

# If tracking results are available, display the interactive viewer
if "frames_b64" in st.session_state and st.session_state.get("frames_b64"):
    st.write("**Step 2:** Use the slider to view tracking results frame by frame.")
    num_frames = st.session_state["num_frames"]
    # Frame slider
    frame_idx = st.slider("Frame index", 0, num_frames - 1, st.session_state.get("current_frame", 0))
    st.session_state["current_frame"] = frame_idx

    # Decode and display the current frame image with bounding box
    frame_data = base64.b64decode(st.session_state["frames_b64"][frame_idx])
    frame_image = Image.open(BytesIO(frame_data))
    st.image(frame_image, caption=f"Frame {frame_idx}", use_column_width=True)

    # Display current speed and a speed chart
    speeds = st.session_state.get("speeds", [])
    if speeds:
        current_speed = speeds[frame_idx] if frame_idx < len(speeds) else None
        st.write(f"**Speed:** {current_speed:.2f} pixels/second" if current_speed is not None else "**Speed:** N/A")
        st.line_chart(speeds, height=150)  # plot speed vs frame index