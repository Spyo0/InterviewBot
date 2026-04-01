# Je-Suis-Coach-AI

Simulateur d'entretien technique **Finance de Marche (Quant / Structureur)**, 100% local et gratuit.

## Stack

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| LLM | Ollama (Llama3 / Mistral) |
| RAG | LangChain + ChromaDB |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Base de donnees | SQLite |
| Graphiques | Plotly |

## Installation

```bash
# 1. Cloner le projet
git clone <repo-url> && cd BotEntretien

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Installer et lancer Ollama
# https://ollama.com
ollama pull llama3

# 4. Configurer (optionnel)
cp .env.example .env
# Editer .env selon vos besoins

# 5. Lancer
streamlit run app.py
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

## Configuration (.env)

```env
OLLAMA_MODEL=llama3           # Modele par defaut
OLLAMA_BASE_URL=http://localhost:11434
VALIDATION_THRESHOLD=0.70     # Seuil de validation
MAX_PDFS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Architecture

```
app.py          # Interface Streamlit
engine.py       # Logique RAG (Ollama + LangChain + ChromaDB)
database.py     # Gestion SQLite
processor.py    # Parsing des PDF
data/           # Stockage PDF + ChromaDB + SQLite
```

## Prerequis

- Python 3.11+
- Ollama installe et actif
- Apple Silicon recommande (M1/M2/M3/M4)
