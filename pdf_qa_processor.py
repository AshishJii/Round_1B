#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import sys
import os
import re
import argparse
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, logging, AutoModelForSeq2SeqLM, AutoTokenizer

# --- Suppress verbose warnings ---
logging.set_verbosity_error()

# --- Part 1: PDF Structure Extraction Functions ---

def is_heading_nlp(text):
    """
    Heuristic for heading detection:
    - No trailing punctuation
    - Short (<12 words)
    - Majority title-case words
    """
    txt = text.strip()
    if not txt or txt[-1] in ".?!":
        return False
    words = txt.split()
    if len(words) > 12:
        return False
    if not words:
        return False
    cap = sum(1 for w in words if w and w[0].isupper())
    return (cap / len(words)) >= 0.5

def is_valid_heading_text(txt, min_length=4, max_non_alnum_ratio=0.5):
    """
    - Require at least `min_length` characters.
    - Discard if too many non-alphanumeric symbols.
    """
    txt = txt.strip()
    if len(txt) < min_length:
        return False
    if not txt:
        return False
    non_alnum = sum(1 for c in txt if not (c.isalnum() or c.isspace()))
    if len(txt) > 0 and non_alnum / len(txt) > max_non_alnum_ratio:
        return False
    return True

def extract_pdf_structure(pdf_path):
    """
    Extracts the hierarchical structure (title, headings, content) from a single PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}", file=sys.stderr)
        return None

    blocks = []
    font_sizes = []

    for page_num, page in enumerate(doc, 1):
        try:
            page_blocks = page.get_text("dict")["blocks"]
        except Exception:
            continue
        for blk in page_blocks:
            if blk.get("type") != 0:
                continue
            block_text = ""
            max_font_size = 0
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        block_text += text + " "
                        size = span.get("size", 0)
                        font_sizes.append(size)
                        if size > max_font_size:
                            max_font_size = size
            if block_text.strip():
                blocks.append({
                    "page": page_num,
                    "text": block_text.strip(),
                    "font_size": max_font_size
                })

    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    heading_threshold = max(avg_font_size * 1.15, 12)

    title = None
    outline = []
    misc = []
    current_h1 = None
    current_h2 = None

    first_page_blocks = sorted(
        [b for b in blocks if b["page"] == 1],
        key=lambda x: x["font_size"],
        reverse=True
    )
    if first_page_blocks:
        title = first_page_blocks[0]["text"]

    for b in blocks:
        txt, page, fsize = b["text"], b["page"], b["font_size"]
        if txt == title:
            continue

        is_font_heading = fsize >= heading_threshold
        is_nlp_heading = is_heading_nlp(txt)

        if (is_font_heading or is_nlp_heading) and is_valid_heading_text(txt):
            level = "H1" if fsize > heading_threshold * 1.2 else "H2"
            node = {"level": level, "text": txt, "content": "", "page": page, "children": []}
            if level == "H1":
                outline.append(node)
                current_h1 = node
                current_h2 = None
            else:
                if current_h1:
                    current_h1["children"].append(node)
                else:
                    outline.append(node)
                current_h2 = node
            continue

        target = current_h2 or current_h1
        if target:
            target["content"] += (" " + txt) if target["content"] else txt
        else:
            misc.append({"page": page, "text": txt})

    result = {"title": title, "outline": outline}
    if misc:
        result["unstructured_content"] = misc
    return result


# --- Part 2: Question-Answering Functions ---

def extract_heading_data(data, parent_h1_text=None, source_file=None):
    """
    Recursively extracts heading/content pairs from the JSON structure.
    Each item is a dictionary: {'heading': '...', 'content': '...', 'source': '...', 'page': ...}.
    """
    structured_data = []

    if isinstance(data, dict) and 'merged_outline' in data:
        for item in data['merged_outline']:
            source_filename = item.get('text', 'Unknown File')
            structured_data.extend(extract_heading_data(item, source_file=source_filename))
        return structured_data

    if isinstance(data, dict) and data.get('level') == 'H0':
        source_filename = data.get('text', 'Unknown File')
        if 'children' in data:
            structured_data.extend(extract_heading_data(data['children'], source_file=source_filename))
        return structured_data

    if isinstance(data, list):
        for item in data:
            structured_data.extend(extract_heading_data(item, parent_h1_text, source_file))

    elif isinstance(data, dict):
        current_h1 = parent_h1_text
        heading_text = ""

        if 'text' not in data or 'content' not in data:
            if 'children' in data:
                structured_data.extend(extract_heading_data(data['children'], current_h1, source_file))
            return structured_data

        if data.get('level') == 'H1':
            heading_text = data['text']
            current_h1 = data['text']
        elif parent_h1_text:
            heading_text = f"{parent_h1_text} | {data['text']}"
        else:
            heading_text = data['text']

        structured_data.append({
            'heading': heading_text,
            'content': data['content'],
            'source': source_file,
            'page': data.get('page', 0)
        })

        if 'children' in data:
            structured_data.extend(extract_heading_data(data['children'], current_h1, source_file))

    return structured_data


def rank_top_headings(all_heading_data, question, top_n=5):
    """
    Stage 1: Ranks headings by semantic similarity and returns the top N results.
    """
    print("--- Stage 1: Ranking Headings by Semantic Similarity ---")
    try:
        model = SentenceTransformer('./models/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}", file=sys.stderr)
        return []

    headings = [item['heading'] for item in all_heading_data]
    if not headings:
        return []

    question_embedding = model.encode(question)
    heading_embeddings = model.encode(headings)
    cosine_scores = util.cos_sim(question_embedding, heading_embeddings)[0]

    results_with_scores = [{'data': all_heading_data[i], 'score': score.item()} for i, score in enumerate(cosine_scores)]
    sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)

    top_results_data = []
    for result in sorted_results:
        if len(top_results_data) >= top_n:
            break
        content = result['data'].get('content')
        if content and content.strip():
            top_results_data.append(result['data'])

    print(f"Identified top {len(top_results_data)} relevant headings with content.")
    return top_results_data

def generate_answers_from_content(top_headings, question):
    """
    Stage 2: Uses a generative model to create answers from the content.
    """
    print("\n--- Stage 2: Generating Refined Text from Content using FLAN-T5 ---")
    try:
        model_path = './models/flan-t5-base'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        # qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception as e:
        print(f"Error loading QA pipeline: {e}", file=sys.stderr)
        return []

    final_answers = []
    for item in top_headings:
        heading = item['heading']
        content = item['content']
        source = item.get('source', 'N/A')
        page = item.get('page', 0)
        print(f"Analyzing content from: '{heading}' (Source: {source})...")

        if not content or not isinstance(content, str):
            print("  -> No content to analyze.")
            continue

        prompt = f"""
        Answer the question based on the main content of the text provided below.
        Your answer must not include any metadata. Focus strictly on providing a comprehensive and accurate answer to the question itself.

        Context: "{content}"
        Question: "{question}"
        Answer:
        """

        results = qa_pipeline(prompt, max_length=512)
        generated_answer = results[0]['generated_text']

        final_answers.append({
            'document': source,
            'refined_text': generated_answer,
            'page_number': page
        })

    return final_answers


# --- Part 3: Main Orchestration Function ---

def process_single_collection(input_json_path, pdf_folder_path, output_json_path):
    """
    Runs the full analysis pipeline for a single collection.
    """
    # --- Step 1: Read Input JSON and Prepare for Processing ---
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing input JSON file '{input_json_path}': {e}", file=sys.stderr)
        return

    persona = input_data.get("persona", {}).get("role", "N/A")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "N/A")
    documents_to_process = input_data.get("documents", [])

    if not documents_to_process or job_to_be_done == "N/A":
        print(f"Skipping collection due to missing documents or job_to_be_done in {input_json_path}", file=sys.stderr)
        return

    # --- Step 2: Process PDF Folder ---
    merged_outline = []
    processed_filenames = []
    for doc_info in documents_to_process:
        filename = doc_info.get("filename")
        if not filename:
            continue
        pdf_path = os.path.join(pdf_folder_path, filename)
        if not os.path.exists(pdf_path):
            print(f"  - WARNING: Could not find {filename} in {pdf_folder_path}. Skipping.", file=sys.stderr)
            continue
        processed_filenames.append(filename)
        structure = extract_pdf_structure(pdf_path)
        if structure and structure.get("outline"):
            pdf_node = {
                "level": "H0", "text": filename,
                "children": structure["outline"],
                "title": structure.get("title", "No Title Found")
            }
            if "unstructured_content" in structure:
                pdf_node["unstructured_content"] = structure["unstructured_content"]
            merged_outline.append(pdf_node)

    internal_json_structure = {"merged_outline": merged_outline}

    # --- Step 3: Run Analysis ---
    all_data = extract_heading_data(internal_json_structure)
    if not all_data:
        print(f"Could not extract any heading/content data from PDFs in {pdf_folder_path}. Skipping.", file=sys.stderr)
        return

    top_headings = rank_top_headings(all_data, job_to_be_done, top_n=5)
    subsection_analysis = generate_answers_from_content(top_headings, job_to_be_done)

    # --- Step 4: Format and Write Final Output JSON ---
    extracted_sections = []
    for i, item in enumerate(top_headings):
        extracted_sections.append({
            "document": item.get('source', 'N/A'),
            "section_title": item.get('heading', 'N/A'),
            "importance_rank": i + 1,
            "page_number": item.get('page', 0)
        })

    final_output = {
        "metadata": {
            "input_documents": processed_filenames, "persona": persona, "job_to_be_done": job_to_be_done
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"--- Collection analysis complete. Output saved to {output_json_path} ---")
    except Exception as e:
        print(f"Error writing output JSON to '{output_json_path}': {e}", file=sys.stderr)


def main():
    """
    Main entry point. Iterates through collection folders.
    It automatically uses an 'input/' directory for collections and writes to an 'output/' directory.
    """
    # Define hardcoded root directories
    collections_root_dir = "input/"
    output_root_dir = "output/"

    # --- Pre-flight checks ---
    # Ensure the root input directory exists
    if not os.path.isdir(collections_root_dir):
        print(f"Error: Input directory '{collections_root_dir}' not found.", file=sys.stderr)
        print("Please create an 'input' directory and place your collection folders inside it.", file=sys.stderr)
        sys.exit(1)

    # Ensure the root output directory exists
    os.makedirs(output_root_dir, exist_ok=True)

    # --- Processing loop ---
    collection_names = sorted(os.listdir(collections_root_dir))
    if not collection_names:
        print(f"Warning: Input directory '{collections_root_dir}' is empty.", file=sys.stderr)
        return

    for collection_name in collection_names:
        collection_path = os.path.join(collections_root_dir, collection_name)
        if not os.path.isdir(collection_path):
            continue

        print(f"\n{'='*70}\nProcessing Collection: {collection_name}\n{'='*70}")

        # Define paths for the current collection
        # Assumes a structure like: input/Collection_1/input.json and input/Collection_1/PDFs/
        input_json_path = None
        for file in os.listdir(collection_path):
            if file.endswith('.json'):
                input_json_path = os.path.join(collection_path, file)
                break

        pdf_folder_path = os.path.join(collection_path, 'PDFs')
        output_collection_path = os.path.join(output_root_dir, collection_name)
        output_json_path = os.path.join(output_collection_path, 'output.json')

        if not input_json_path or not os.path.isdir(pdf_folder_path):
            print(f"Skipping '{collection_name}' as it does not contain an input JSON file or a 'PDFs' directory.", file=sys.stderr)
            continue

        process_single_collection(input_json_path, pdf_folder_path, output_json_path)

    print(f"\n{'='*70}\nAll collections processed.\n{'='*70}")


if __name__ == "__main__":
    main()
