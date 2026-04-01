"""Logique RAG : LangChain + ChromaDB + providers cloud (Groq / HuggingFace)."""

import os
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from processor import process_pdf, list_pdfs

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "0.70"))

# Providers cloud
PROVIDERS = {
    "groq": {
        "label": "Groq Cloud",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "huggingface": {
        "label": "HuggingFace Inference",
        "models": ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3-8B-Instruct"],
    },
}


def build_llm(provider: str, model_name: str):
    """Construit le LLM selon le provider choisi."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante dans .env")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.7)

    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        api_token = os.getenv("HF_API_TOKEN", "")
        if not api_token:
            raise ValueError("HF_API_TOKEN manquant dans .env")
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_token,
            temperature=0.7,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=llm)

    else:
        raise ValueError(f"Provider inconnu : {provider}. Utilisez 'groq' ou 'huggingface'.")

SYSTEM_PROMPT = """Tu es un coach d'entretien spécialisé en Finance Quantitative (Quant/Structureur).
Tu poses des questions techniques sur : calcul stochastique, probabilités, pricing de produits dérivés, brainteasers logiques.

Règles :
- Pose UNE question à la fois, claire et précise.
- Adapte la difficulté au niveau montré par les réponses précédentes.
- Pour les calculs mentaux, demande des approximations (ex: √0.8, log(1.05)).
- Cite toujours la source (livre/page) quand tu t'appuies sur le contexte fourni.
- Évalue les réponses avec rigueur mais bienveillance.

{context}"""

EVAL_PROMPT = """Tu es un évaluateur d'entretien Quant. Évalue la réponse suivante.

Question : {question}
Réponse de l'utilisateur : {answer}
Contexte de référence : {context}

Donne :
1. Un score entre 0.0 et 1.0 (précision et complétude de la réponse)
2. Un feedback constructif en français
3. La réponse correcte si l'utilisateur s'est trompé

Format EXACT de réponse :
SCORE: [nombre entre 0.0 et 1.0]
FEEDBACK: [ton feedback]
CORRECTION: [correction si nécessaire, sinon "Correct"]"""


class JeSuisCoachEngine:
    def __init__(self, provider: str = "groq", model_name: str = "llama-3.3-70b-versatile"):
        self.provider = provider
        self.model_name = model_name
        self.llm = build_llm(provider, model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._ensure_collection()

    def _ensure_collection(self):
        self.collection = self.chroma_client.get_or_create_collection(
            name="quant_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdfs(self) -> int:
        """Indexe tous les PDF du dossier data/. Retourne le nombre de chunks indexés."""
        pdfs = list_pdfs()
        total = 0

        for pdf_name in pdfs:
            docs = process_pdf(pdf_name)
            if not docs:
                continue

            ids = [f"{pdf_name}_{i}" for i in range(len(docs))]
            texts = [d["text"] for d in docs]
            metadatas = [d["metadata"] for d in docs]

            # Vérifier les IDs existants pour éviter les doublons
            existing = set()
            try:
                result = self.collection.get(ids=ids)
                existing = set(result["ids"]) if result["ids"] else set()
            except Exception:
                pass

            new_ids = [id_ for id_ in ids if id_ not in existing]
            if not new_ids:
                continue

            new_indices = [ids.index(id_) for id_ in new_ids]
            new_texts = [texts[i] for i in new_indices]
            new_metadatas = [metadatas[i] for i in new_indices]

            embeddings = self.embeddings.embed_documents(new_texts)
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metadatas,
                embeddings=embeddings,
            )
            total += len(new_ids)

        return total

    def search_context(self, query: str, n_results: int = 3, chapter_filter: str = None) -> str:
        """Recherche le contexte pertinent dans ChromaDB."""
        if self.collection.count() == 0:
            return ""

        where_filter = None
        if chapter_filter:
            where_filter = {"chapter": chapter_filter}

        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            where=where_filter if where_filter else None,
        )

        if not results["documents"][0]:
            return ""

        context_parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            ref = f"[{meta.get('source', '?')} - {meta.get('chapter', '?')} - p.{meta.get('start_page', '?')}]"
            context_parts.append(f"{ref}\n{doc}")

        return "\n\n---\n\n".join(context_parts)

    def generate_question(self, topic: str, chapter: str = None, history: str = "") -> str:
        """Génère une question d'entretien."""
        context = self.search_context(topic, chapter_filter=chapter)
        context_block = f"\nContexte des livres de référence :\n{context}" if context else ""

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Génère une question d'entretien sur le thème : {topic}
{history_block}
Pose une seule question technique précise."""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context_block,
            "topic": topic,
            "history_block": f"\nQuestions déjà posées dans cette session :\n{history}" if history else "",
        })

    def evaluate_answer(self, question: str, answer: str, topic: str) -> dict:
        """Évalue la réponse de l'utilisateur."""
        context = self.search_context(question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un évaluateur rigoureux d'entretiens Quant."),
            ("human", EVAL_PROMPT),
        ])

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context if context else "Pas de contexte de référence disponible.",
        })

        return self._parse_evaluation(raw)

    def _parse_evaluation(self, raw: str) -> dict:
        """Parse la réponse d'évaluation du LLM."""
        result = {"score": 0.5, "feedback": raw, "correction": ""}

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    result["score"] = max(0.0, min(1.0, float(score_str)))
                except ValueError:
                    pass
            elif line.startswith("FEEDBACK:"):
                result["feedback"] = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("CORRECTION:"):
                result["correction"] = line.replace("CORRECTION:", "").strip()

        return result

    def get_available_chapters(self) -> list[str]:
        """Retourne la liste des chapitres indexés."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        chapters = set()
        for meta in results["metadatas"]:
            if "chapter" in meta:
                chapters.add(meta["chapter"])
        return sorted(chapters)

    def get_indexed_sources(self) -> list[str]:
        """Retourne la liste des sources indexées."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
