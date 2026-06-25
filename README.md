# Video Feature Tracking & Motion Analysis App

[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44--red?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C2D91?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![GCP](https://img.shields.io/badge/Google_Cloud_Run-Deployed-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

A lightweight, production-ready Edge-AI assistant and motion analysis tool that uses computer vision to track points of interest across video sequences. Built with a decoupled microservice architecture (FastAPI backend and Streamlit frontend), containerized with Docker, and deployed on Google Cloud Run.

Deployed Live 👉 [https://motion-tracker-frontend-283944763297.us-central1.run.app](https://motion-tracker-frontend-283944763297.us-central1.run.app)

---

## 🎯 The "Why" (Real-World Value)

In fields like physical therapy, sports analytics, and manufacturing automation, analyzing how features move across video frames is essential. This tool provides a zero-install, web-based interface for:
- Tracking specific joints or equipment in sports or physical rehabilitation.
- Analyzing velocity and motion trajectories of objects in manufacturing lines.
- Enhancing optical tracking reliability under adverse conditions (lighting shifts, noise) using real-time frame preprocessing filters.

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV (Lucas-Kanade Sparse Optical Flow, Shi-Tomasi Corner Detection, Canny Edge Detection, Gaussian Filtering, Histogram Equalization).
- **Backend API:** FastAPI (Async endpoints, Pydantic data schemas, Multipart Form handling).
- **Frontend Dashboard:** Streamlit, Streamlit Image Coordinates (interactive click-to-select interface).
- **Infrastructure & MLOps:** Docker, Docker Compose, Google Cloud Run, Google Artifact Registry.
- **Languages:** Python (typing, concurrent IO).

---

## 🧠 System Architecture & Workflow

The application leverages a decoupled frontend/backend microservice architecture designed to handle concurrent processing runs efficiently.

```mermaid
graph TD
    subgraph Client [Client Interface]
        A[Upload MP4 Video] --> B[Interactive ROI Selection]
        B --> C[Select Filters: Blur / Equalize / Edges]
    end

    subgraph Frontend [Streamlit Service (Port 8501)]
        C --> D[Extract Metadata & Start Frame]
        D --> E[POST request: /track]
    end

    subgraph Backend [FastAPI Service (Port 8000)]
        E --> F[Write isolated temp file]
        F --> G[Run Preprocessing Filters]
        G --> H[Shi-Tomasi Point Detection]
        H --> I[Lucas-Kanade Optical Flow Tracking]
        I --> J[Compute Velocity pixels/sec]
        J --> K[Base64 JSON Response Output]
    end

    Frontend -->|Send Video Bytes & ROI Parameters| Backend
    Backend -->|Return Annotated Frames & Speeds| Frontend

    K --> L[Interactive Frame-by-Frame Slider]
    K --> M[Plot Real-Time Speed Line Chart]

    %% Styling
    classDef feClass fill:#FF4B4B,stroke:#333,stroke-width:2px,color:#FFF;
    classDef beClass fill:#009688,stroke:#333,stroke-width:2px,color:#FFF;
    classDef clientClass fill:#262730,stroke:#333,stroke-width:2px,color:#FFF;
    
    class Frontend feClass;
    class Backend beClass;
    class Client,L,M clientClass;
```

---

## 🚀 Quickstart Guide

Ensure you have Docker and Docker Compose installed on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/naimul214/Video-Feature-Tracking-and-Motion-Analysis-App.git
cd Video-Feature-Tracking-and-Motion-Analysis-App
```

### 2. Run with Docker Compose
To build and run both the FastAPI backend and Streamlit frontend locally:
```bash
docker-compose up --build
```
- **Streamlit Frontend:** Available at [http://localhost:8501](http://localhost:8501)
- **FastAPI API Swagger Docs:** Available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Local Python Development (Without Docker)
If you prefer running without containers:
```bash
# Setup Backend
cd backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# In a new terminal, Setup Frontend
cd ../frontend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
export FASTAPI_URL="http://localhost:8000"  # set FASTAPI_URL=http://localhost:8000 on Windows CMD
streamlit run app.py --server.port=8501
```

---

## 📊 Results & Demo

*(Insert a GIF or Screenshot here showing the interface with an active tracking region and the speed line chart below it)*

### Performance Metrics
- **Algorithm:** Lucas-Kanade Sparse Optical Flow with Shi-Tomasi feature selection.
- **Inference Latency:** ~10-15ms per frame processing time on standard CPU hardware.
- **Robustness:** Incorporates Canny Edge Detection and Histogram Equalization options to handle dramatic illumination variations.

---

## ⚠️ Limitations & Future Work

- **Feature Loss Fallback:** If optical flow tracking points drift or are lost due to rapid motion or occlusion, the system currently falls back to the static center. Integrating a Kalman Filter or CSRT tracker would provide better occlusion resilience.
- **Scale Calibration:** Velocity is currently calculated in `pixels/second`. Adding a spatial calibration input (e.g., specifying that $X$ pixels equal $Y$ centimeters) would enable real-world physical metrics.
- **Batch Processing & Queueing:** Currently, processing is done synchronously on request. For long videos, this can time out. Implementing a task worker pattern (e.g., Celery + Redis) would allow asynchronous processing.
- **Dense Optical Flow Integration:** Add a toggle for Farneback Dense Optical Flow to visualize complete motion vectors across the entire frame rather than tracking a specific ROI.

---

## 🤝 Connect

- **Author:** Naimul Hassan
- **Education:** Honours Bachelor of Artificial Intelligence, Durham College (April 2026)
- **LinkedIn:** [linkedin.com/in/naimul214](https://linkedin.com/in/naimul214)
- **GitHub:** [github.com/naimul214](https://github.com/naimul214)
