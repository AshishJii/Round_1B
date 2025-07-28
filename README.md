# PDF Semantic QA Engine

## 📄 Overview

**PDF Semantic QA Engine** is an intelligent, containerized PDF parsing and question-answering system. It extracts structured content from PDFs, semantically searches for user queries in document headings, and generates concise, context-aware answers by reading corresponding body text.

This tool is optimized for **offline use**, with all required transformer models pre-downloaded and cached during image build time via Docker.

---

## 🚀 Features

* **Hierarchical PDF Structure Extraction**
  Uses `PyMuPDF` to parse PDFs, extract headings (H1, H2) and their associated bodies using a combination of:

  * Font-size heuristics
  * NLP-based heading identification
  * Custom filters for noise reduction

* **Semantic Heading Retrieval**
  Ranks extracted headings based on semantic similarity to the query using:

  * [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from SentenceTransformers
  * Cosine similarity scoring between heading embeddings and query embedding

* **Context-Aware Answer Generation**
  Uses [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model locally to:

  * Generate a natural language answer using the body text of top-matching headings
  * Avoids metadata leakage, focusing only on semantic context

* **Offline Capability**

  * All transformer models are downloaded and stored at build time.
  * No internet access is required during runtime.
  * Suitable for air-gapped environments.

* **Batch Processing Support**

  * Accepts a directory of document collections
  * Each collection contains a structured input JSON and associated PDFs
  * Outputs enriched metadata, top relevant sections, and generated answers

---

## 🧠 How It Works

### 🔹 Step 1: Structure Extraction

Each PDF is parsed page-by-page. For each block:

* Font size and textual heuristics classify content as potential headings (`H1`, `H2`) or body.
* The document is converted into a hierarchical JSON format capturing:

  * Title
  * Heading levels
  * Page numbers
  * Cleaned text content

### 🔹 Step 2: Semantic Ranking

All extracted headings are embedded using MiniLM and compared with the query:

* Cosine similarity is computed
* Top-N (default 5) semantically closest headings are selected

### 🔹 Step 3: QA Generation

Each selected heading’s content is passed through a FLAN-T5 model with the input prompt and the result is a refined answer mapped to:
* Source document
* Section title
* Page number
* Rank

---

## 🧪 Input Format

Each collection must follow this structure:

```
input/
└── Collection_1/
    ├── input.json
    └── PDFs/
        ├── file1.pdf
        └── file2.pdf
```

### input.json

```json
{
  "persona": {
    "role": "User"
  },
  "job_to_be_done": {
    "task": "Explain the difference between supervised and unsupervised learning."
  },
  "documents": [
    { "filename": "ml_basics.pdf" },
    { "filename": "advanced_ai.pdf" }
  ]
}
```

---

## 📦 Output Format

Each output will be saved to:

```
output/
└── Collection_1/
    └── output.json
```

### output.json

```json
{
  "metadata": {
    "input_documents": ["ml_basics.pdf", "advanced_ai.pdf"],
    "persona": "User",
    "job_to_be_done": "Explain the difference between supervised and unsupervised learning."
  },
  "extracted_sections": [
    {
      "document": "ml_basics.pdf",
      "section_title": "Supervised vs Unsupervised Learning",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "ml_basics.pdf",
      "refined_text": "Supervised learning uses labeled data ...",
      "page_number": 5
    }
  ]
}
```

---

## 🐳 Docker Setup (Offline Execution)

### 🧱 Build Docker Image

```bash
docker build -t pdf-qa-engine .
```

This will:

* Install all Python dependencies
* Download and cache:

  * `all-MiniLM-L6-v2`
  * `flan-t5-base`

### ▶️ Run the Application

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-qa-engine
```

### Directory Bind Explanation

* `input/` → Mounts collections of PDFs and `input.json`
* `output/` → Output JSONs are written here

---

## 🧰 Requirements (if not using Docker)

* Python 3.12+
* `pip install -r requirements.txt`
* Dependencies:

  * `PyMuPDF`
  * `sentence-transformers`
  * `transformers`
  * `torch`

---

## 📂 Directory Layout

```
.
├── Dockerfile
├── pdf_qa_processor.py      # Main pipeline script
├── download_models.py       # Downloads required models for offline use
├── requirements.txt
├── input/                   # Mounted input folder
└── output/                  # Mounted output folder
```

---

## 🧠 Models Used

| Model Name            | Role                       | Location                    |
| --------------------- | -------------------------- | --------------------------- |
| `all-MiniLM-L6-v2`    | Semantic heading ranking   | `./models/all-MiniLM-L6-v2` |
| `google/flan-t5-base` | Text-to-text QA generation | `./models/flan-t5-base`     |

All models are stored locally under `./models/` to enable **offline inference**.

---
