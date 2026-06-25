FROM python:3.10-slim-bullseye

# Install system dependencies (if any)
# (Using opencv-python-headless, so minimal OS libs are needed; if issues, uncomment below)
# RUN apt-get update && apt-get install -y libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY app.py ./app.py

EXPOSE 8501
# Run the Streamlit app on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]