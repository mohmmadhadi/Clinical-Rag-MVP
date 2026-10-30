# Clinical RAG MVP
### Evidence-Grounded Question Answering over Synthetic Clinical Notes

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about clinical notes using semantic search, LLM-powered reasoning, and verifiable passage-level citations.

The project demonstrates how to build a clinical-style question-answering pipeline that prioritizes **grounded responses, source traceability, and controlled retrieval** rather than relying on an LLM's parametric knowledge.

## Overview

Clinical documentation contains valuable information, but extracting specific details from unstructured notes can be time-consuming.

This project implements an end-to-end RAG pipeline that allows users to ask natural-language questions about a collection of synthetic clinical notes. The system retrieves relevant passages, generates answers based exclusively on the retrieved evidence, and attaches citations to factual claims.

It also includes an agentic retrieval workflow that rewrites queries, handles unsuccessful retrieval attempts, and retries with a broader set of passages when necessary.

**Key features**

- **Semantic Retrieval:** Dense vector search using Sentence Transformers and cosine similarity.
- **Agentic Query Rewriting:** Expands clinical abbreviations and reformulates questions to improve retrieval.
- **Adaptive Retrieval:** Retries unanswered questions using the original query and a larger candidate set.
- **Evidence-Grounded Generation:** Restricts LLM responses to the retrieved clinical passages.
- **Citation Verification:** Checks whether cited passage IDs belong to the retrieved evidence.
- **Interactive Interface:** A Gradio application for asking questions and inspecting retrieved passages.
- **Rate Limiting:** Limits requests to 15 questions per hour per client IP.

## Architecture

The application follows a modular RAG architecture:

```text
             Synthetic Clinical Notes
                       |
                       v
                 Data Ingestion
                       |
                       v
                 Text Chunking
              (180 tokens, 40 overlap)
                       |
                       v
              Embedding Generation
             (BGE-small-en-v1.5)
                       |
                       v
                Vector Index
                       |
                       |
User Question --------+
       |
       v
  Query Rewriting
   (Agent Mode)
       |
       v
  Semantic Retrieval
   (Cosine Similarity)
       |
       v
  Retrieved Passages
       |
       v
   LLM Generation
       |
       v
  Citation Verification
       |
       v
 Evidence-Grounded Answer
       |
       v
    Gradio Interface
```

### Agentic Retrieval Workflow

When agent mode is enabled, the system follows an additional retrieval strategy:

1. **Query rewriting:** The LLM converts the original question into a standalone search query and expands clinical abbreviations (e.g., SOB → shortness of breath).
2. **Initial retrieval:** The rewritten query is used to retrieve the most relevant passages.
3. **Answer generation:** The LLM generates an answer using only the retrieved passages.
4. **Citation verification:** Every cited passage ID is checked against the retrieved results.
5. **Adaptive retry:** If the answer is `NOT_FOUND`, the system retrieves additional passages using the original question and retries generation with a merged evidence set.

The system also supports a standard RAG mode without query rewriting or adaptive retry, allowing comparison between the two approaches.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Integration | OpenAI Python SDK |
| LLM Provider | OpenRouter |
| Default LLM | NVIDIA Nemotron 3 Ultra |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Operations | NumPy |
| Embedding Framework | Sentence Transformers |
| Dataset | Hugging Face Datasets |
| User Interface | Gradio |

## Dataset

The project uses the [Asclepius Synthetic Clinical Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) dataset.

The ingestion pipeline:

- Streams clinical notes directly from Hugging Face.
- Scans up to 20,000 records.
- Removes duplicate and empty notes.
- Randomly selects up to 500 unique notes using a fixed random seed.
- Stores the resulting dataset in JSONL format.

The resulting notes are divided into overlapping chunks to improve retrieval granularity and preserve relevant context.

**Note:** The dataset contains synthetic clinical text. No real patient records are intentionally used in this demonstration.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mohmmadhadi/Clinical-Rag-MVP.git
cd Clinical-Rag-MVP
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the LLM API

The project uses OpenRouter through the OpenAI-compatible API.

Obtain an API key from [OpenRouter](https://openrouter.ai/) and configure the environment variable:

Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_api_key"
```

Linux / macOS:

```bash
export OPENROUTER_API_KEY="your_api_key"
```

You can optionally configure a different model or API endpoint:

```bash
LLM_MODEL="your-model"
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_API_KEY="your_api_key"
```

`OPENROUTER_API_KEY` and `LLM_API_KEY` are alternative ways to provide the API key.

### 5. Prepare the dataset (optional)

If you want to regenerate the clinical notes:

```bash
python ingest.py
```

This creates:

```text
data/
└── notes.jsonl
```

### 6. Build the vector index (optional)

After preparing the dataset:

```bash
python build_index.py
```

This generates the embedding vectors and chunk metadata:

```text
index/
├── vectors.npy
└── chunks.jsonl
```

If a prebuilt index is already available in the repository, you can skip the ingestion and indexing steps.

### 7. Launch the application

```bash
python app.py
```

Open the local Gradio URL displayed in your terminal (typically `http://127.0.0.1:7860`).

## Using the Application

The interface provides three main controls:

- **Question:** Enter a natural-language question about the clinical notes.
- **Agent Mode:** Enable or disable query rewriting and adaptive retrieval.
- **Passages to Retrieve:** Select the number of passages to retrieve (3–10).

The application displays:

1. The generated answer.
2. The retrieval and reasoning trace.
3. The retrieved clinical passages, including their similarity scores.

Passages cited in the answer are visually highlighted, making it easier to inspect the evidence supporting the response.

### Example Questions

```text
Which patients were treated for pneumonia and with what?

Which patient had SOB and what was given?

Who was diagnosed with a stroke?
```

If the retrieved evidence does not contain the requested information, the system returns a `NOT_FOUND` response rather than attempting to answer from external knowledge.

## Project Structure

```text
Clinical-Rag-MVP/
│
├── clinrag/
│   ├── chunking.py       # Clinical note chunking
│   ├── retrieval.py      # Embedding generation and vector search
│   └── rag.py            # LLM integration and RAG pipeline
│
├── data/
│   └── notes.jsonl       # Processed synthetic clinical notes
│
├── index/
│   ├── vectors.npy       # Dense embedding vectors
│   └── chunks.jsonl      # Indexed text chunks and metadata
│
├── app.py                # Gradio application
├── ingest.py             # Dataset ingestion
├── build_index.py        # Vector index construction
├── requirements.txt      # Python dependencies
└── README.md
```

## Technical Details

**Chunking**

Clinical notes are divided into overlapping chunks using the following configuration:

- Chunk size: 180 tokens (whitespace-delimited units).
- Overlap: 40 tokens.
- Chunk identifiers: Unique IDs in the format `note_id#chunk_number`.

**Embeddings and Retrieval**

The system uses `BAAI/bge-small-en-v1.5` to generate normalized dense embeddings.

Similarity is calculated using the dot product of normalized vectors, equivalent to cosine similarity.

The default retrieval configuration returns six passages.

**Generation and Citation Validation**

The LLM is instructed to:

- Use only the supplied clinical passages.
- Cite supporting passage IDs after factual statements.
- Return `NOT_FOUND` when the evidence is insufficient.
- Avoid providing medical advice.

A lightweight validation function checks generated citation IDs against the retrieved passages and flags unverified citations.

## Limitations

This project is an MVP and has several limitations:

- The dataset is synthetic and does not represent real-world clinical documentation.
- Retrieval relies on dense vector similarity without a dedicated reranking model.
- Citation validation checks passage-ID membership, not whether a citation semantically supports the associated claim.
- The system does not perform clinical validation or guarantee medical accuracy.
- The in-memory rate limiter is intended for a lightweight demonstration, not production deployment.
- Model availability, response quality, and latency depend on the configured LLM provider.

## Future Improvements

Potential directions for further development include:

- Hybrid retrieval combining dense and sparse search.
- Cross-encoder reranking for improved retrieval precision.
- More comprehensive citation and factual consistency evaluation.
- Automated benchmarking across different retrieval configurations.
- Retrieval evaluation using Recall@K, Precision@K, and MRR.
- Answer evaluation using faithfulness and answer relevance metrics.
- Docker-based deployment and production-ready API endpoints.

## Disclaimer

**This project is intended for educational and research purposes only.**
