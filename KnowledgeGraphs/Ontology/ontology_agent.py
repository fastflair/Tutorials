#!/usr/bin/env python3
"""
ontology_agent.py

This script uses the OpenAI GPT-4-mini model to process an input text,
extract ontology suggestions and update a local ontology stored in "ontology.json".
Each ontology entry is a dictionary with the keys:
  - "concept": a unique term normalized to Title Case,
  - "description": the explanation of the concept, and
  - "parent": an optional parent concept.

Duplicate detection is performed by comparing normalized concept names and by
computing cosine similarity between text embeddings (using a free Hugging Face
model from the SentenceTransformer library). If a duplicate is found and the new
description is longer, it updates the existing entry. No user confirmation is needed;
this is fully automated.

Usage:
    python ontology_agent.py "<input text describing concepts and hierarchy>"
"""

import os
import sys
import json
import math
import pickle
import time
import torch
import networkx as nx

from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Constants
ONTOLOGY_FILE = "ontology.json"
KG_FILE = "knowledge_graph.pkl"
SIMILARITY_THRESHOLD = 0.95  # Adjust threshold as needed

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
retry_limit = 3

CONFIG = {
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "openai_model_large": "gpt-4o",
    "device": device,
    "dtype": dtype,
    "retry_limit": retry_limit,
}

#############################################
# OpenAI Prompt Response (with retry logic)
#############################################
def get_openai_prompt_response(
    prompt: str,
    config: dict,
    max_tokens: int = 6000,
    temperature: float = 0.33,
    openai_model: str = "",
):
    """
    Sends a prompt to OpenAI's API and retrieves the response with retry logic.
    """
    client = OpenAI(api_key=config["openai_api_key"])
    response = client.chat.completions.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "Act as a helpful assistant, ontologist, knowledge architect, and systems engineer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=openai_model or config["openai_model"],
        temperature=temperature,
    )

    retry_count = 0
    while retry_count < config["retry_limit"]:
        try:
            message_content = response.choices[0].message.content
            return message_content
        except Exception as e:
            print(f"Error occurred: {e}")
            retry_count += 1
            if retry_count == config["retry_limit"]:
                print("Retry limit reached. Moving to the next iteration.")
                return ""
            else:
                print(f"Retrying... (Attempt {retry_count}/{config['retry_limit']})")
                time.sleep(1)

#############################################
# Extract Nested Ontology Suggestions
#############################################
def get_suggested_ontology(input_text: str, config: dict) -> list:
    """
    Construct a prompt instructing the model to extract an ontology in nested JSON format.
    The output should be a JSON list where each object has exactly two keys:
       "concept" and "description", with subordinate concepts nested in "children".
    """
    prompt = (
        "Extract an ontology in NESTED JSON format from the following text. "
        "Return a JSON list where each element is an object with exactly two keys: "
        "'concept': a unique term in Title Case, and 'description': a concise description. "
        "If a concept has subordinate concepts, include them under the key 'children' as a nested list. "
        "Return only valid JSON without any additional keys. Text:\n" + input_text
    )
    response = get_openai_prompt_response(prompt, config, max_tokens=6000, temperature=0.33)
    response = response.replace("```json", "").replace("```", "")
    try:
        suggestions = json.loads(response)
        if isinstance(suggestions, list):
            return suggestions
        else:
            print("The returned ontology suggestion is not a list.")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to decode the ontology suggestions as JSON: {e}")
        print("Response received:")
        print(response)
        sys.exit(1)

#############################################
# Flatten Nested Ontology
#############################################
def flatten_ontology(nested_ontology: list, parent: str = None) -> list:
    """
    Recursively flattens a nested ontology structure.
    Each node is expected to have "concept" and "description", and optionally "children".
    The flattened entry includes a "parent" key.
    """
    flat_list = []
    for node in nested_ontology:
        entry = {
            "concept": node.get("concept", "").strip(),
            "description": node.get("description", "").strip(),
            "parent": parent
        }
        flat_list.append(entry)
        children = node.get("children", [])
        if children:
            flat_list.extend(flatten_ontology(children, parent=entry["concept"]))
    return flat_list

#############################################
# Extract Triples from Input Text
#############################################
def get_triples_from_input(input_text: str, config: dict) -> list:
    """
    Extract triples from the input text using the OpenAI model.
    Expected output is a JSON list of objects with keys "subject", "predicate", and "object".
    """
    prompt = (
        "Extract triples from the following text in valid JSON format. "
        "Each triple should be a JSON object with the keys: 'subject', 'predicate', and 'object'. "
        "Return a JSON list of such objects. Text:\n" + input_text
    )
    response = get_openai_prompt_response(prompt, config, max_tokens=2000, temperature=0.33)
    response = response.replace("```json", "").replace("```", "")
    try:
        triples = json.loads(response)
        if isinstance(triples, list):
            return triples
        else:
            print("The returned triple extraction is not a list.")
            return []
    except json.JSONDecodeError as e:
        print(f"Failed to decode triple extraction JSON: {e}")
        print("Response received:")
        print(response)
        return []

#############################################
# Ontology Management Utilities
#############################################
def load_ontology() -> list:
    """Load the ontology from ontology.json if it exists; otherwise, return an empty list."""
    if os.path.exists(ONTOLOGY_FILE):
        try:
            with open(ONTOLOGY_FILE, "r", encoding="utf-8") as f:
                ontology = json.load(f)
            if isinstance(ontology, list):
                return ontology
            else:
                print("Ontology file does not contain a list; starting with an empty ontology.")
        except json.JSONDecodeError:
            print("Failed to decode ontology.json; starting with an empty ontology.")
    return []

def save_ontology(ontology: list) -> None:
    """Save the hierarchical ontology to ontology.json."""
    with open(ONTOLOGY_FILE, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2)
    print(f"Ontology saved to '{ONTOLOGY_FILE}'.")

def get_embedding(text: str, model: SentenceTransformer):
    """Compute an embedding for a given text using the SentenceTransformer model."""
    return model.encode(text)

def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(a * a for a in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def normalize_concept(concept: str) -> str:
    """Normalize the concept name to Title Case."""
    return concept.strip().title()

#############################################
# Update Flat Ontology With New Suggestions
#############################################
def update_ontology(existing_ontology: list, new_suggestions_flat: list, model: SentenceTransformer) -> list:
    """
    Update the flat ontology with new suggestions.
    Duplicate detection uses normalized concept names and cosine similarity of description embeddings.
    If a duplicate is found and the new description is longer, update the description.
    """
    existing_entries = []
    for entry in existing_ontology:
        norm = normalize_concept(entry.get("concept", ""))
        entry["concept"] = norm
        desc = entry.get("description", "")
        emb = get_embedding(desc, model) if desc else None
        existing_entries.append((norm, desc, emb, entry))

    for suggestion in new_suggestions_flat:
        new_concept = normalize_concept(suggestion.get("concept", ""))
        new_desc = suggestion.get("description", "").strip()
        new_parent = suggestion.get("parent")
        new_emb = get_embedding(new_desc, model) if new_desc else None

        duplicate_found = False
        for norm, desc, emb, entry in existing_entries:
            if new_concept.lower() == norm.lower():
                duplicate_found = True
            elif new_emb is not None and emb is not None:
                if cosine_similarity(new_emb, emb) >= SIMILARITY_THRESHOLD:
                    duplicate_found = True
            if duplicate_found:
                if len(new_desc) > len(desc):
                    print(f"Updating description for concept '{new_concept}'.")
                    entry["description"] = new_desc
                break

        if not duplicate_found:
            new_entry = {
                "concept": new_concept,
                "description": new_desc,
                "parent": new_parent
            }
            existing_ontology.append(new_entry)
            existing_entries.append((new_concept, new_desc, new_emb, new_entry))
            print(f"Added new concept: '{new_concept}'.")
    return existing_ontology

#############################################
# Rebuild Nested Hierarchical Ontology
#############################################
def build_hierarchical_ontology(flat_ontology: list) -> list:
    """
    Transforms a flat ontology list (with "concept", "description", and "parent")
    into a nested tree structure where each node includes "concept", "description", and "children".
    """
    node_map = {}
    for entry in flat_ontology:
        concept = entry["concept"]
        node_map[concept] = {"concept": concept, "description": entry["description"], "children": []}

    root_nodes = []
    for entry in flat_ontology:
        concept = entry["concept"]
        parent = entry.get("parent")
        if parent and parent in node_map:
            node_map[parent]["children"].append(node_map[concept])
        else:
            root_nodes.append(node_map[concept])
    return root_nodes

#############################################
# Determine Node Type via LLM
#############################################
def determine_node_type_llm(node_name: str, context_text: str, ontology_list: list, config: dict) -> str:
    """
    Given a node name, context, and a flat ontology list (each entry with "concept" and "description"),
    this function uses the OpenAI model to decide which ontology concept best matches the node.
    It returns the matching concept (normalized to Title Case) or "Other" if none is found.
    """
    # Build a string of ontology definitions (each as "Concept: Description")
    ontology_str = "\n".join(
        [f"- {normalize_concept(entry['concept'])}: {entry['description']}" for entry in ontology_list]
    )
    prompt = (
        "Given the following ontology definitions:\n"
        f"{ontology_str}\n\n"
        f"Determine the most appropriate ontology concept (from the above list) for the node "
        f"with name \"{node_name}\" in the context of the triple: {context_text}. "
        "If none of the concepts fit, simply answer 'Other'. "
        "Only output a single phrase representing the matching concept."
    )
    response = get_openai_prompt_response(prompt, config, max_tokens=50, temperature=0.0)
    node_type = response.strip().strip(".")
    # Normalize the result
    return normalize_concept(node_type) if node_type and node_type.lower() != "other" else "Other"

#############################################
# Update Knowledge Graph with Triples
#############################################
def update_knowledge_graph(existing_graph: nx.Graph, triples: list, ontology_flat: list, config: dict) -> nx.Graph:
    """
    Update the NetworkX graph with new triples.
    For each triple (with keys "subject", "predicate", "object"):
      - Nodes are added with attributes "name" and "node_type".
      - The node type is assigned by calling determine_node_type_llm,
        passing the node name, context (the triple), and the ontology.
      - An edge is added with the predicate as the "relation" attribute.
    """
    for triple in triples:
        subj = triple.get("subject", "").strip()
        pred = triple.get("predicate", "").strip()
        obj = triple.get("object", "").strip()
        if not subj or not pred or not obj:
            continue

        subj_norm = normalize_concept(subj)
        obj_norm = normalize_concept(obj)
        context_text = f"subject: {subj}, predicate: {pred}, object: {obj}"
        subj_type = determine_node_type_llm(subj, context_text, ontology_flat, config)
        obj_type = determine_node_type_llm(obj, context_text, ontology_flat, config)

        if subj_norm not in existing_graph:
            existing_graph.add_node(subj_norm, name=subj_norm, node_type=subj_type)
        if obj_norm not in existing_graph:
            existing_graph.add_node(obj_norm, name=obj_norm, node_type=obj_type)
        existing_graph.add_edge(subj_norm, obj_norm, relation=pred)
        print(f"Added edge: ({subj_norm}, {subj_type}) -[{pred}]-> ({obj_norm}, {obj_type})")
    return existing_graph

#############################################
# Main Function
#############################################
def main():
    if len(sys.argv) < 2:
        print("Usage: python ontology_agent.py '<input text>'")
        sys.exit(1)

    input_text = sys.argv[1]
    print("Processing input text for ontology extraction...")

    # Load the SentenceTransformer model for local embedding computation.
    print("Loading the SentenceTransformer model...")
    hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # ---- Ontology Extraction and Update ----
    new_nested_ontology = get_suggested_ontology(input_text, CONFIG)
    print("New nested ontology suggestions received:")
    print(json.dumps(new_nested_ontology, indent=2))
    new_flat_ontology = flatten_ontology(new_nested_ontology)
    existing_flat_ontology = load_ontology()  # Flat ontology from file
    updated_flat_ontology = update_ontology(existing_flat_ontology, new_flat_ontology, hf_model)
    hierarchical_ontology = build_hierarchical_ontology(updated_flat_ontology)
    save_ontology(hierarchical_ontology)
    print("Ontology update complete.")

    # ---- Triple Extraction and Knowledge Graph Update ----
    print("Extracting triples for the knowledge graph...")
    triples = get_triples_from_input(input_text, CONFIG)
    print("Extracted triples:")
    print(json.dumps(triples, indent=2))

    # Load existing knowledge graph if available; otherwise, create a new graph.
    if os.path.exists(KG_FILE):
        with open(KG_FILE, "rb") as f:
            kg = pickle.load(f)
        print(f"Loaded existing knowledge graph from '{KG_FILE}'.")
    else:
        kg = nx.Graph()
        print("Creating a new knowledge graph.")

    # Update the knowledge graph using the updated ontology and LLM-based node type extraction.
    kg = update_knowledge_graph(kg, triples, updated_flat_ontology, CONFIG)

    # Save the knowledge graph using pickle.
    with open(KG_FILE, "wb") as f:
        pickle.dump(kg, f)
    print(f"Knowledge graph saved to '{KG_FILE}'.")

if __name__ == "__main__":
    main()