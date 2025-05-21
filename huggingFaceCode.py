import os
import re
import nltk
import pdfplumber
import networkx as nx
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from transformers import pipeline

# Download sentence tokenizer if needed
nltk.download("punkt")

# Hugging Face Pipelines
ner_pipeline = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")
pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")

# Utility: Clean up OCR and symbols
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)           # Remove excessive whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    return text.strip()

# Utility: Remove subwords like '##tion'
def filter_tokens(tokens):
    return [
        t for t in tokens
        if not t['word'].startswith('##') and is_clean_word(t['word'])
    ]

# Utility: Filter junk tokens
def is_clean_word(word):
    return word.isalpha() and len(word) > 2

# Step 1: Extract all text from PDFs in a folder
def extract_text_from_folder(folder_path):
    full_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += clean_text(page_text) + "\n"
    return full_text

# Step 2: Extract subject-verb-object triples (basic version)
def extract_svo_triples(pos_results):
    nouns = [t['word'] for t in pos_results if t['entity_group'] in ['NOUN', 'PROPN']]
    verbs = [t['word'] for t in pos_results if t['entity_group'] == 'VERB']
    triples = []
    if len(nouns) >= 2 and verbs:
        triples.append((nouns[0], verbs[0], nouns[1]))
    return triples

# Step 3: Build the graph
def build_graph(text, min_entity_freq=2):
    G = nx.DiGraph()
    entity_freq = {}
    sentences = sent_tokenize(text)

    for sentence in sentences:
        ner_raw = ner_pipeline(sentence)
        pos_raw = pos_pipeline(sentence)
        ner_results = filter_tokens(ner_raw)
        pos_results = filter_tokens(pos_raw)

        # Count entities
        for ent in ner_results:
            entity = ent['word']
            entity_freq[entity] = entity_freq.get(entity, 0) + 1

        # Add subject-verb-object edges
        triples = extract_svo_triples(pos_results)
        for subj, verb, obj in triples:
            if subj != obj and is_clean_word(subj) and is_clean_word(obj) and is_clean_word(verb):
                G.add_edge(subj, obj, label=verb)

    # Prune graph: keep only frequent/connected nodes
    high_freq = {n for n, f in entity_freq.items() if f >= min_entity_freq}
    G = G.subgraph([n for n in G.nodes if n in high_freq or G.degree(n) > 1]).copy()

    return G

# Step 4: Visualize
def plot_graph(G, output_file="cognitive_map_cleaned.png"):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5)
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Cleaned Cognitive Map", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

# Step 5: Run it all
if __name__ == "__main__":
    folder_path = "D:/NLP/UCD_Test/PDFs"  # 🔁 Replace this
    all_text = extract_text_from_folder(folder_path)
    graph = build_graph(all_text)
    plot_graph(graph)
