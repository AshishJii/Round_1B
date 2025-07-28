# download_models.py
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Download and save SentenceTransformer
    print("Downloading SentenceTransformer model: all-MiniLM-L6-v2")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    st_model.save(os.path.join(model_dir, "all-MiniLM-L6-v2"))
    print("SentenceTransformer model saved in ./models/all-MiniLM-L6-v2")

    # Download and save Hugging Face flan-t5-base model
    print("Downloading text2text-generation model: google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer.save_pretrained(os.path.join(model_dir, "flan-t5-base"))
    model.save_pretrained(os.path.join(model_dir, "flan-t5-base"))
    print("flan-t5-base model saved in ./models/flan-t5-base")

    print("All models downloaded and saved in ./models")

if __name__ == "__main__":
    main()
