# --- Required Installations ---
# pip install torch transformers sentence-transformers

import json
import argparse
import sys
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, logging

# Suppress verbose warnings from the transformers pipeline
logging.set_verbosity_error()

def extract_heading_data(data, parent_h1_text=None):
    """
    Recursively extracts heading/content pairs from the JSON.
    Each item is a dictionary: {'heading': '...', 'content': '...'}.
    """
    structured_data = []
    if isinstance(data, dict) and 'outline' in data:
        data = data['outline']

    if isinstance(data, list):
        for item in data:
            structured_data.extend(extract_heading_data(item, parent_h1_text))
    elif isinstance(data, dict):
        current_h1 = parent_h1_text
        heading_text = ""

        # Ensure there is text to process
        if 'text' not in data or 'content' not in data:
            if 'children' in data: # Still recurse even if parent has no content
                 structured_data.extend(extract_heading_data(data['children'], current_h1))
            return structured_data

        if data['level'] == 'H1':
            heading_text = data['text']
            current_h1 = data['text']
        elif parent_h1_text:
            heading_text = f"{parent_h1_text} | {data['text']}"
        else:
            heading_text = data['text']

        structured_data.append({'heading': heading_text, 'content': data['content']})

        if 'children' in data:
            structured_data.extend(extract_heading_data(data['children'], current_h1))

    return structured_data

def rank_top_headings(all_heading_data, question, top_n=5):
    """
    Stage 1: Ranks headings by semantic similarity and returns the top N
    results that have non-empty content.
    """
    print("--- Stage 1: Ranking Headings by Semantic Similarity ---")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}", file=sys.stderr)
        return []

    headings = [item['heading'] for item in all_heading_data]
    if not headings:
        return []

    question_embedding = model.encode(question)
    heading_embeddings = model.encode(headings)
    cosine_scores = util.cos_sim(question_embedding, heading_embeddings)[0]

    # Pair original data with scores
    results_with_scores = []
    for i, score in enumerate(cosine_scores):
        results_with_scores.append({'data': all_heading_data[i], 'score': score.item()})

    sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)

    # --- MODIFIED LOGIC ---
    # Filter the sorted list to find the top N headings that have content.
    top_results_data = []
    for result in sorted_results:
        # Stop once we have found enough valid headings.
        if len(top_results_data) >= top_n:
            break

        # Check if the content is present and not just whitespace.
        content = result['data'].get('content')
        if content and content.strip():
            top_results_data.append(result['data'])
    # ----------------------

    print(f"Identified top {len(top_results_data)} relevant headings with content.")
    return top_results_data

def generate_answers_from_content(top_headings, question):
    """
    Stage 2: Uses a GENERATIVE model to create long-form answers from the content.
    """
    print("\n--- Stage 2: Generating Answers from Content using FLAN-T5 ---")
    try:
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )
    except Exception as e:
        print(f"Error loading QA pipeline: {e}", file=sys.stderr)
        return []

    final_answers = []
    for i, item in enumerate(top_headings):
        heading = item['heading']
        content = item['content']
        print(f"Generating answer from: '{heading}'...")

        # This check is now mostly redundant due to the new logic in Stage 1,
        # but it remains as a safeguard.
        if not content or not isinstance(content, str):
            print("  -> No content to analyze.")
            continue

        # The final, generalized prompt
        prompt = f"""
        Answer the question based on the main content of the text provided below.

        Your answer must not include any metadata. This includes details about the author, publisher, publication date, contact information, or any references. Focus strictly on providing a comprehensive and accurate answer to the question itself.

        Context:
        "{content}"

        Question:
        "{question}"

        Answer:
        """

        results = qa_pipeline(prompt, max_length=512)
        generated_answer = results[0]['generated_text']

        final_answers.append({
            'heading': heading,
            'answer': generated_answer
        })

    return final_answers

def main():
    """
    Main function to orchestrate the two-stage question-answering process.
    """
    parser = argparse.ArgumentParser(
        description="A two-stage system to find answers from a JSON document.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("question", help="The question you want to ask.")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading file '{args.json_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)
    print(f"Processing file: {args.json_file}")
    print(f"Question: {args.question}")
    print("-" * 60)

    # Step 1: Get all heading/content data from the JSON
    all_data = extract_heading_data(json_data)

    # Step 2: Rank the headings and get the top 5 with non-empty content
    top_5_headings = rank_top_headings(all_data, args.question, top_n=5)

    # Step 3: Generate specific answers from the content of those top sections
    final_results = generate_answers_from_content(top_5_headings, args.question)

    # Step 4: Display the final results
    print("\n--- Final Answers ---")
    if final_results:
        for result in final_results:
            print(f"From Heading: '{result['heading']}'")
            print(f"  Answer: {result['answer']}\n")
    else:
        print("No answers could be generated from the top-ranked sections.")
    print("-" * 60)

if __name__ == "__main__":
    main()
