# download_models.py
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# This script is run during the Docker build process.
# It pre-downloads the required models so they are baked into the image.
# This allows the main application to run without internet access.

def main():
    """Downloads and caches the necessary models."""
    print("Downloading SentenceTransformer model: all-MiniLM-L6-v2")
    SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model downloaded.")

    print("Downloading text2text-generation pipeline model: google/flan-t5-base")
    pipeline("text2text-generation", model="google/flan-t5-base")
    print("text2text-generation model downloaded.")

    print("All models have been downloaded and cached successfully.")

if __name__ == "__main__":
    main()
