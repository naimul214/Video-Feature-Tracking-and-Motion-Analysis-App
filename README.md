# 🎥 Video Feature Tracking and Motion Analysis App

Deployed Live 👉 [https://motion-tracker-frontend-283944763297.us-central1.run.app](https://motion-tracker-frontend-283944763297.us-central1.run.app)

This project is a full-stack AI-powered web app built with **Streamlit** and **FastAPI**, containerized with Docker, and deployed on **Google Cloud Run**. It allows users to upload `.mp4` videos, select a region of interest (ROI), and analyze the motion of selected features using **Lucas-Kanade Optical Flow**.

---

## 🚀 Features

- 📤 Upload `.mp4` videos (max 200MB)
- ⏱️ Select custom time ranges from the video
- 🔍 Click to select a tracking area (ROI) on the video
- 🌀 Preprocessing options (Gaussian Blur, Histogram Equalization, Edge Detection)
- 📦 Lucas-Kanade-based motion tracking
- 📈 Frame-by-frame speed analysis (pixels/sec)
- 🖼️ Tracked frame preview with bounding boxes
- ☁️ Fully deployed on Google Cloud Run

---

## 🧠 Tech Stack

| Component      | Technology                    |
|----------------|-------------------------------|
| Frontend       | [Streamlit](https://streamlit.io) |
| Backend        | [FastAPI](https://fastapi.tiangolo.com) |
| Processing     | [OpenCV](https://opencv.org) (Lucas-Kanade Optical Flow) |
| Containerization | Docker |
| Deployment     | Google Cloud Run + Artifact Registry |
| Messaging      | REST API via JSON & base64 for images |

---

## 🗂️ Project Structure

Video-Feature-Tracking-and-Motion-Analysis-App/ ├── frontend/ │ ├── app.py │ ├── Dockerfile │ └── requirements.txt ├── backend/ │ ├── main.py │ ├── Dockerfile │ └── requirements.txt └── docker-compose.yml



---

## 🧪 How It Works

1. **User uploads a video** and selects a portion of it using time sliders.
2. **First frame is displayed**, and the user clicks a point to define the top-left corner of a region to track.
3. **ROI is tracked frame-by-frame** using Lucas-Kanade Optical Flow.
4. The app **computes speed** (pixels/sec) and draws bounding boxes on the frames.
5. Results are displayed interactively with sliders and graphs.

---

## 📦 Deployment (Cloud Run)

This app is containerized and deployed as two services:

| Service    | Port | Description         |
|------------|------|---------------------|
| Frontend   | 8501 | Streamlit app       |
| Backend    | 8000 | FastAPI tracker API |

Deployed using:
- Docker CLI
- Google Artifact Registry
- `gcloud run deploy`

---

## 🛠 Local Development

### Prerequisites:
- Python 3.10+
- Docker
- [Streamlit Image Coordinates Component](https://github.com/andfanilo/streamlit-image-coordinates)

### Running Locally:

```bash
# From project root
docker-compose up --build
# Access frontend at: http://localhost:8501

🧹 Cleanup
To delete deployed services from Google Cloud:

gcloud run services delete motion-tracker-frontend --region=us-central1
gcloud run services delete motion-tracker-backend --region=us-central1


🙋‍♂️ Author
Naimul Hassan
Student | AI Developer
Durham College
📧 nhridoy214@gmail.com
🔗 LinkedIn | GitHub

📄 License
MIT License – free to use, modify, and distribute.


