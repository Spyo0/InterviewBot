# Je-Suis-Coach-AI

Simulateur d'entretien technique **Finance de Marche (Quant / Structureur)**, 100% cloud et gratuit.

## Stack

| Composant | Technologie |
|---|---|
| Interface | Streamlit (dark theme) |
| LLM | Groq Cloud / HuggingFace Inference |
| Embeddings | HuggingFace Inference API (0 RAM locale) |
| RAG | LangChain + ChromaDB |
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

## Fonctionnalites

### Mode Entretien
- Choix du theme et du chapitre
- Timer de stress configurable (Off / 1 / 2 / 3 / 5 min) avec compte a rebours JS en temps reel
- Difficulte progressive : les questions s'adaptent au score moyen de la session
- Feedback immediat avec score, correction et temps de reponse
- Rendu LaTeX natif pour toutes les formules mathematiques

### Mode Examen
- 10 questions d'affilee sans feedback intermediaire
- Correction detaillee revelee uniquement a la fin
- Score global, nombre de questions validees, temps moyen

### Dashboard
- Matrice de maitrise par theme (Non aborde / En cours / Maitrise)
- Graphique d'evolution des temps de reponse
- Historique complet des sessions

### Gestion PDF
- Import de supports PDF (max 5)
- Decoupe automatique par chapitre
- Indexation vectorielle pour le RAG

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

# RAG & Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VALIDATION_THRESHOLD=0.70
MAX_PDFS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

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
app.py          # Interface Streamlit (Entretien, Examen, PDF, Dashboard)
engine.py       # Logique RAG + providers cloud + difficulte progressive
database.py     # Gestion SQLite (scores, sessions, maitrise)
processor.py    # Parsing des PDF (PyMuPDF)
data/           # Stockage PDF + ChromaDB + SQLite
```

## Prerequis

- Python 3.11+
- Une cle API Groq (gratuite) ou un token HuggingFace
