# Je-Suis-Coach-AI

Simulateur d'entretien technique **Finance de Marche (Quant / Structureur)**, 100% cloud et gratuit.

## Stack

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| LLM | Groq Cloud / HuggingFace Inference |
| RAG | LangChain + ChromaDB |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Base de donnees | SQLite |
| Graphiques | Plotly |

## Installation

```bash
# 1. Cloner le projet
git clone https://github.com/Spyo0/InterviewBot.git && cd InterviewBot

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Configurer
cp .env.example .env
# Ajouter votre cle API Groq ou HuggingFace dans .env

# 4. Lancer
streamlit run app.py
```

## Providers LLM

Le moteur tourne entierement en cloud — aucune installation locale de modele requise.

| Provider | Modeles | Cle requise |
|---|---|---|
| **Groq** (recommande) | Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B | [console.groq.com](https://console.groq.com) |
| **HuggingFace** | Mistral 7B, Llama 3 8B | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Configuration (.env)

```env
# Provider : "groq" ou "huggingface"
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# HuggingFace
HF_API_TOKEN=your_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# RAG
VALIDATION_THRESHOLD=0.70
MAX_PDFS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Utilisation

### 1. Importer des PDF
Onglet **PDF** > deposez vos supports (max 5) > cliquez **Indexer**.

### 2. Simuler un entretien
Onglet **Entretien** > choisissez un theme > **Nouvelle question** > repondez > **Soumettre**.

Seuil de validation : **70%** (configurable dans `.env`).

### 3. Suivre sa progression
Onglet **Dashboard** > matrice de maitrise, temps de reponse, historique.

## Themes couverts

- Calcul stochastique (Ito, Brownien)
- Probabilites
- Pricing de produits derives (Black-Scholes, Grecques)
- Volatilite implicite
- Monte Carlo
- Brainteasers logiques
- Calcul mental / Approximations

## Architecture

```
app.py          # Interface Streamlit
engine.py       # Logique RAG (LangChain + ChromaDB + providers cloud)
database.py     # Gestion SQLite
processor.py    # Parsing des PDF
data/           # Stockage PDF + ChromaDB + SQLite
```

## Prerequis

- Python 3.11+
- Une cle API Groq (gratuite) ou un token HuggingFace
