# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyMuPDF (fitz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and the model downloader script
COPY requirements.txt .
COPY download_models.py .

# Install Python packages from requirements.txt
# This step requires internet access during the build phase
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-CACHE MODELS ---
# Run the downloader script to fetch and cache models into the image layer.
# This is the key step that allows the container to run without network access.
RUN python download_models.py

# Copy the main application script into the container
COPY pdf_qa_processor.py .

# Create placeholder directories for mounting volumes
# This is good practice and avoids potential permission issues.
RUN mkdir /app/input && mkdir /app/output

# Set the entrypoint to run the main application script
ENTRYPOINT ["python", "pdf_qa_processor.py"]
