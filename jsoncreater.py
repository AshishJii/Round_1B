#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import sys
import re

def is_heading_nlp(text):
    """
    Heuristic for heading detection:
    - No trailing punctuation
    - Short (<12 words)
    - Majority title‑case words
    """
    txt = text.strip()
    if not txt or txt[-1] in ".?!":
        return False
    words = txt.split()
    if len(words) > 12:
        return False
    cap = sum(1 for w in words if w and w[0].isupper())
    return (cap / len(words)) >= 0.5

def is_valid_heading_text(txt, min_length=4, max_non_alnum_ratio=0.5):
    """
    — Require at least `min_length` characters (excluding leading/trailing spaces).
    — Discard if too many non‑alphanumeric symbols.
    """
    txt = txt.strip()
    if len(txt) < min_length:
        return False
    # count everything that’s not a letter, digit, or space
    non_alnum = sum(1 for c in txt if not (c.isalnum() or c.isspace()))
    if non_alnum / len(txt) > max_non_alnum_ratio:
        return False
    return True

def extract_pdf_structure(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    font_sizes = []

    # 1) Extract all text blocks and track font sizes
    for page_num, page in enumerate(doc, 1):
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] != 0:
                continue
            block_text = ""
            max_font_size = 0
            for line in blk["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        block_text += text + " "
                        font_sizes.append(span["size"])
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
            if block_text.strip():
                blocks.append({
                    "page": page_num,
                    "text": block_text.strip(),
                    "font_size": max_font_size
                })

    # 2) Decide heading threshold dynamically
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    heading_threshold = avg_font_size + 1.5  # tweak as needed

    title = None
    outline = []
    misc = []
    current_h1 = None
    current_h2 = None

    for b in blocks:
        txt, page, fsize = b["text"], b["page"], b["font_size"]

        # — Title detection on first page
        if page == 1 and title is None \
           and len(txt.split()) > 4 \
           and txt[-1] not in ".?!":
            title = txt
            continue

        # — Heading detection (font or NLP) + validity check
        if (fsize >= heading_threshold or is_heading_nlp(txt)) \
           and is_valid_heading_text(txt):
            # H1 if contains colon, else H2
            level = "H1" if ":" in txt else "H2"
            node = {
                "level": level,
                "text": txt,
                "content": "",
                "page": page,
                "children": []
            }
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

        # — Otherwise, attach as content under the nearest heading
        target = current_h2 or current_h1
        if target:
            target["content"] += (" " + txt) if target["content"] else txt
        else:
            misc.append({"page": page, "text": txt})

    # 3) Build result
    result = {"title": title, "outline": outline}
    if misc:
        result["unstructured"] = misc
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_outline.py input.pdf output.json")
        sys.exit(1)
    structure = extract_pdf_structure(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    print(f"Saved outline to {sys.argv[2]}")

if __name__ == "__main__":
    main()
