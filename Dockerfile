# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Fix apt sources to use HTTPS instead of HTTP (for Bookworm+)
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and model downloader script
COPY requirements.txt download_models.py ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so the container can run offline
RUN python download_models.py

# Copy the main application script
COPY pdf_qa_processor.py .

# Create directories for volume mounting
RUN mkdir -p /app/input /app/output

# Set the default command
ENTRYPOINT ["python", "pdf_qa_processor.py"]
