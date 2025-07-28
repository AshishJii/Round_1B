#!/usr/bin/env python3
"""
challenge1b.py: Unified PDF outline extractor and QA for multiple documents.

Features:
- Extracts structured outlines (title, headings, nested sections) from multiple PDFs
- Detects underlined text via OCR to annotate headings
- Merges JSON outputs: top-level keys are PDF filenames, values are their outlines
- Supports semantic query over merged JSON: ranks relevant sections and generates answers

Usage:
  python challenge1b.py <pdf_or_dir>... --output merged.json [--question "Your question?"]
"""

import os
import sys
import json
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, logging

# Suppress transformer verbosity
logging.set_verbosity_error()


def detect_underlines_in_pdf(pdf_path, dpi=300, thresh_val=180):
    """
    Perform OCR-based detection of underlined words per page.
    Returns a dict: {page_num: set(of underlined words)}
    """
    underlined_by_page = {}
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting {pdf_path} to images: {e}", file=sys.stderr)
        return underlined_by_page

    for page_num, page in enumerate(pages, 1):
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        # detect horizontal line structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//2, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        underlined = set()
        n = len(data['level'])
        for i in range(n):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            for lx, ly, lw, lh in line_boxes:
                if abs(ly - (y + h)) < 5 and (x < lx + lw and x + w > lx):
                    text = data['text'][i].strip()
                    if text:
                        underlined.add(text)
                    break
        underlined_by_page[page_num] = underlined
    return underlined_by_page


def is_heading_nlp(text):
    txt = text.strip()
    if not txt or txt[-1] in ".?!":
        return False
    words = txt.split()
    if len(words) > 12:
        return False
    cap = sum(1 for w in words if w and w[0].isupper())
    return (cap / len(words)) >= 0.5


def is_valid_heading_text(txt, min_length=4, max_non_alnum_ratio=0.5):
    txt = txt.strip()
    if len(txt) < min_length:
        return False
    non_alnum = sum(1 for c in txt if not (c.isalnum() or c.isspace()))
    if non_alnum / len(txt) > max_non_alnum_ratio:
        return False
    return True


def extract_pdf_structure(pdf_path):
    """
    Extracts title, hierarchical headings, content, and underlined annotations from a PDF.
    Returns a dict: {title, outline: [...], unstructured: [...]}.
    """
    underlines = detect_underlines_in_pdf(pdf_path)
    doc = fitz.open(pdf_path)
    blocks = []
    font_sizes = []

    for page_num, page in enumerate(doc, 1):
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            text, max_size = "", 0
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if t:
                        text += t + " "
                        font_sizes.append(span.get("size", 0))
                        max_size = max(max_size, span.get("size", 0))
            if text.strip():
                blocks.append({"page": page_num, "text": text.strip(), "font_size": max_size})

    avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    threshold = avg_font + 1.5
    title, outline, misc = None, [], []
    current_h1, current_h2 = None, None

    for b in blocks:
        txt, page, fsize = b['text'], b['page'], b['font_size']
        # Title detection
        if page == 1 and title is None and len(txt.split()) > 4 and txt[-1] not in ".?!":
            title = txt
            continue
        # Heading detection
        if (fsize >= threshold or is_heading_nlp(txt)) and is_valid_heading_text(txt):
            level = "H1" if ":" in txt else "H2"
            node = {"level": level, "text": txt, "content": "", "page": page, "children": [],
                    "underlined": bool(set(txt.split()) & underlines.get(page, set()))}
            if level == "H1":
                outline.append(node)
                current_h1, current_h2 = node, None
            else:
                if current_h1:
                    current_h1['children'].append(node)
                else:
                    outline.append(node)
                current_h2 = node
            continue
        # Content
        parent = current_h2 or current_h1
        if parent:
            parent['content'] += (" " + txt) if parent['content'] else txt
        else:
            misc.append({"page": page, "text": txt})

    result = {"title": title, "outline": outline}
    if misc:
        result['unstructured'] = misc
    return result


def extract_heading_data(data, parent_h1=None):
    """Flatten outline data into list of {'heading', 'content'} dicts."""
    flat = []
    items = data.get('outline', data) if isinstance(data, dict) else data
    for node in items:
        lvl, text, content = node.get('level'), node.get('text'), node.get('content')
        if lvl == 'H1':
            current = text
        else:
            current = f"{parent_h1} | {text}" if parent_h1 else text
        if content and content.strip():
            flat.append({'heading': current, 'content': content})
        if node.get('children'):
            flat.extend(extract_heading_data(node['children'], current if lvl=='H1' else parent_h1))
    return flat


def rank_top_headings(all_data, question, top_n=5):
    """Rank headings by semantic similarity to question and return top_n items."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    headings = [d['heading'] for d in all_data]
    embeddings = model.encode(headings)
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, embeddings)[0]
    ranked = sorted([(all_data[i], scores[i].item()) for i in range(len(all_data))], key=lambda x: x[1], reverse=True)
    results = []
    for data, score in ranked:
        if len(results) >= top_n:
            break
        if data.get('content', '').strip():
            results.append(data)
    return results


def generate_answers_from_content(top_headings, question):
    """Generate answers from top headings using a generative QA model."""
    qa = pipeline('text2text-generation', model='google/flan-t5-base')
    answers = []
    for item in top_headings:
        prompt = f"""
Answer the question based on the text below. Do not include metadata.
Context:\n""" + item['content'] + f"\nQuestion: {question}\nAnswer:"""
        out = qa(prompt, max_length=512)[0]['generated_text']
        answers.append({'heading': item['heading'], 'answer': out.strip()})
    return answers


def query_merged_json(json_path, question, top_n=5):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_data = []
    for pdf, struct in data.items():
        flat = extract_heading_data(struct)
        for d in flat:
            d['heading'] = f"{pdf} | {d['heading']}"
        all_data.extend(flat)
    top = rank_top_headings(all_data, question, top_n)
    return generate_answers_from_content(top, question)


def main():
    parser = argparse.ArgumentParser(description="Extract and query outlines from multiple PDFs.")
    parser.add_argument('paths', nargs='+', help="PDF files or directories containing PDFs")
    parser.add_argument('--output', '-o', required=True, help="Path to merged output JSON")
    parser.add_argument('--question', '-q', help="Optional question for QA over merged JSON")
    args = parser.parse_args()

    pdfs = []
    for p in args.paths:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(p, f))
        elif p.lower().endswith('.pdf'):
            pdfs.append(p)

    merged = {}
    for pdf in pdfs:
        merged[os.path.basename(pdf)] = extract_pdf_structure(pdf)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged JSON saved to {args.output}")

    if args.question:
        answers = query_merged_json(args.output, args.question)
        print("\n--- QA Results ---")
        for a in answers:
            print(f"From: {a['heading']}\n{a['answer']}\n")

if __name__ == '__main__':
    main()
