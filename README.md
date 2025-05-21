# nlp-cognitive-mapping

OVERVIEW
-----------
This project builds a **cognitive map** from PDF documents using NLP techniques 
like Named Entity Recognition (NER) and Part-of-Speech (POS) tagging via Hugging Face.

It extracts concepts and relationships (e.g., who does what to whom), visualizing 
them as a network graph.

KEY FEATURES
---------------
- Extracts text from multiple PDFs
- Uses Hugging Face for NER and POS tagging
- Cleans and filters noisy tokens (e.g., subwords like '##tion')
- Builds a subject–verb–object (SVO) graph
- Visualizes relationships using NetworkX


SETUP
---------
1. Install dependencies:

   pip install transformers nltk pdfplumber matplotlib networkx

2. Download NLTK data:

   python -c "import nltk; nltk.download('punkt')"

3. Place PDFs in the `pdfs/` folder.

4. Run:

   python cognitive_map.py

HOW IT WORKS
----------------
- Text is extracted from all PDFs
- Hugging Face applies NER and POS tagging
- Subwords (##) and junk words are filtered
- SVO triples are identified and turned into a graph
- The graph is saved as `cognitive_map.png`

FILTERING
------------
- Removes non-alphabetic and short tokens (<3 letters)
- Removes subword fragments (e.g., '##ed')
- Ignores low-frequency entities

OUTPUT
---------
A clean cognitive map showing how concepts/entities relate across your PDFs.






