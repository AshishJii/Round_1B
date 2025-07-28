# PDF Batch Processing and Q\&A Engine

This project provides a **containerized solution** to process batches of PDF documents. It:

* Extracts a **hierarchical structure** from each PDF.
* Uses a **sentence-transformer model** to find the most relevant sections based on a given task.
* Employs a **generative language model** to provide summarized answers.

> ✅ The entire process runs in a **sandboxed, offline Docker container**, designed for strict execution environments with resource limits.

---

## 🛠 Prerequisites

* **Docker:**
  You must have Docker installed and the Docker daemon running.
  [Get Docker](https://docs.docker.com/get-docker/)

---

## 📁 Directory Structure

Make sure your project directory is structured **exactly** as shown below:

```
/your_project_folder/
│
├── pdf_qa_processor.py         # Main Python script
├── download_models.py          # Helper script to pre-cache models
├── Dockerfile                  # Defines Docker container environment
├── requirements.txt            # Python dependencies list
│
├── input/                      # Input collections
│   ├── Collection_1/
│   │   ├── challenge1b_input.json
│   │   └── PDFs/
│   │       ├── doc1.pdf
│   │       └── ...
│   │
│   └── Collection_2/
│       ├── another_input.json
│       └── PDFs/
│           ├── docA.pdf
│           └── ...
│
└── output/                     # Output written here (created automatically)
```

---

## 🚀 How to Run

The process includes **two main steps**:

### 1. Build the Docker Image

This step will:

* Build the container image.
* Pre-cache all AI models (enabling offline execution).

Run the following in your project root:

```bash
docker build --platform linux/amd64 -t pdf-processor-final-app .
```

> 📌 The first time will take several minutes to download and cache the models.

---

### 2. Run the Docker Container

Once built, use the command below to run the container:

```bash
docker run --rm --network none --cpus="8" --memory="16g" \
    -v "$(pwd)/input:/app/input" \
    -v "$(pwd)/output:/app/output" \
    pdf-processor-final-app \
    /app/input /app/output
```

> 📂 On completion, results will be available in the `output` directory with:
>
> * One subfolder per collection
> * An `output.json` file inside each
