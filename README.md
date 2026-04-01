# Je-Suis-Coach-AI

Cloud-based technical interview simulator for **quantitative finance roles** such as quant analyst or structurer.

The application generates interview questions from indexed PDF study material, evaluates answers, tracks progress, and runs entirely with cloud LLM providers.

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq Cloud / HuggingFace Inference |
| Embeddings | HuggingFace Inference API |
| Retrieval | LangChain + ChromaDB |
| Database | SQLite |
| Charts | Plotly |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Spyo0/InterviewBot.git
cd InterviewBot
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a `.env` file at the project root and add your provider credentials.

Minimal example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

HF_API_TOKEN=your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VALIDATION_THRESHOLD=0.70
MAX_PDFS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

You only need the credentials for the provider you actually use.

### 5. Run the app

```bash
streamlit run app.py
```

## LLM Providers

The application runs fully in the cloud. No local model installation is required.

| Provider | Models | Required credential |
|---|---|---|
| **Groq** | Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B | [console.groq.com](https://console.groq.com) |
| **HuggingFace** | Mistral 7B, Llama 3 8B | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Features

### Interview Mode

- Choose a topic.
- Choose a difficulty level: `Auto`, `Fundamental`, `Intermediate`, or `Advanced`.
- Automatically retrieve the most relevant PDF chapters for the selected topic.
- Use a configurable stress timer: `Off`, `1 min`, `2 min`, `3 min`, or `5 min`.
- Adapt difficulty progressively when `Auto` mode is enabled.
- Get immediate scoring, feedback, correction, and response time.
- Render mathematical expressions with native LaTeX support.

### Exam Mode

- Run a 10-question exam session.
- Hide feedback until the end of the exam.
- Show a full correction summary at the end.
- Track average score, validated answers, and average response time.

### Dashboard

- View a topic mastery matrix.
- Track response time trends over time.
- Review recent answer history.

### PDF Support

- Upload up to 5 PDF files.
- Split PDFs automatically by chapters and pages.
- Index the content in ChromaDB for retrieval.
- Use the most relevant indexed chapters automatically based on the selected topic.
- Display a visual extract from the source page only when the question actually depends on a figure, chart, diagram, or similar visual support.

## Environment Variables

```env
# Provider selection
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# HuggingFace
HF_API_TOKEN=your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Retrieval and embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VALIDATION_THRESHOLD=0.70
MAX_PDFS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Covered Topics

- Stochastic calculus
- Probabilities
- Derivatives pricing
- Black-Scholes
- Greeks
- Implied volatility
- Monte Carlo
- Logical brainteasers
- Mental math and approximations

## Project Structure

```text
app.py          # Streamlit interface (Interview, Exam, PDF, Dashboard)
engine.py       # RAG pipeline, LLM providers, question generation, evaluation
database.py     # SQLite persistence for sessions, scores, and mastery
processor.py    # PDF parsing and page/chapter extraction
data/           # PDF storage, ChromaDB data, SQLite database
```

## Requirements

- Python 3.11+
- A Groq API key or a HuggingFace API token

