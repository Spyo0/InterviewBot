"""Logique RAG : LangChain + ChromaDB + providers cloud (Groq / HuggingFace)."""

import os
import re
from typing import Optional
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from processor import get_page_preview, process_pdf, list_pdfs

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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

DIFFICULTY_INSTRUCTIONS = {
    "Fondamental": "\nNiveau de difficulté : FONDAMENTAL. Pose une question de base pour consolider les acquis.",
    "Intermédiaire": "\nNiveau de difficulté : INTERMÉDIAIRE. Pose une question standard d'entretien.",
    "Élevé": "\nNiveau de difficulté : ÉLEVÉ. Pose une question avancée, piège ou multi-étapes.",
}
VISUAL_KEYWORDS = (
    "figure",
    "schéma",
    "schema",
    "graphique",
    "graph",
    "courbe",
    "diagramme",
    "illustr",
    "visuel",
    "tableau",
)
INLINE_SOURCE_PATTERNS = (
    "source :",
    "source:",
    "référence :",
    "référence:",
    "reference :",
    "reference:",
    "réponse attendue :",
    "réponse attendue:",
    "expected answer:",
)


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
- Appuie-toi en priorité sur le contexte issu des PDF indexés quand il est fourni.
- Si la page source contient une figure, un schéma ou un graphe utile, tu peux t'appuyer dessus et y faire explicitement référence.
- Ne donne JAMAIS la réponse, la définition complète, une correction, ni une "réponse attendue" dans l'énoncé.
- N'inclus JAMAIS de ligne "Source", "Référence" ou une page/livre dans le texte de la question : l'interface affiche déjà cette information séparément.
- Si le contexte contient une définition explicite, transforme-la en question de compréhension, de reformulation ou d'application sans recopier cette définition.
- Pour les calculs mentaux, demande des approximations (ex: √0.8, log(1.05)).
- Appuie-toi toujours sur une source identifiable du contexte fourni, sans l'écrire dans le texte de la question.
- Évalue les réponses avec rigueur mais bienveillance.
- Formate TOUTES les équations et expressions mathématiques en LaTeX inline ($...$) ou display ($$...$$).
  Exemples : $dS_t = \\mu S_t dt + \\sigma S_t dW_t$, $\\mathbb{{E}}[X]$, $\\frac{{\\partial V}}{{\\partial t}}$.
- Ne jamais écrire de formule en texte brut. Toujours utiliser la notation LaTeX.

{context}"""

EVAL_PROMPT = """Tu es un évaluateur d'entretien Quant. Évalue la réponse suivante.

Question : {question}
Réponse de l'utilisateur : {answer}
Contexte de référence : {context}

Consignes d'évaluation :
- Vérifie la rigueur mathématique : chaque terme doit être présent (ex: ne pas oublier $dt$, $dW_t$ dans le Lemme d'Itô).
- Pénalise fortement les erreurs de signe, les termes manquants et les confusions de notation.
- Pour les démonstrations, vérifie que chaque étape est justifiée.
- Formate toutes les équations en LaTeX ($...$  ou $$...$$).

Donne :
1. Un score entre 0.0 et 1.0 (précision et complétude de la réponse)
2. Un feedback constructif en français avec les erreurs précises identifiées
3. La réponse correcte complète si l'utilisateur s'est trompé

Format EXACT de réponse :
SCORE: [nombre entre 0.0 et 1.0]
FEEDBACK: [ton feedback détaillé]
CORRECTION: [correction complète en LaTeX si nécessaire, sinon "Correct"]"""


class JeSuisCoachEngine:
    def __init__(self, provider: str = "groq", model_name: str = "llama-3.3-70b-versatile"):
        self.provider = provider
        self.model_name = model_name
        self.llm = build_llm(provider, model_name)
        hf_token = os.getenv("HF_API_TOKEN", "")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=hf_token if hf_token else None,
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._ensure_collection()

    def _ensure_collection(self):
        self.collection = self.chroma_client.get_or_create_collection(
            name="quant_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdfs(self) -> int:
        """Ré-indexe tous les PDF du dossier data/. Retourne le nombre de chunks indexés."""
        pdfs = list_pdfs()
        total = 0

        for pdf_name in pdfs:
            docs = process_pdf(pdf_name)
            if not docs:
                continue

            try:
                self.collection.delete(where={"source": pdf_name})
            except Exception:
                pass

            ids = [
                f"{pdf_name}_p{doc['metadata'].get('page', doc['metadata'].get('start_page', 0))}_c{doc['metadata']['chunk_index']}"
                for doc in docs
            ]
            texts = [d["text"] for d in docs]
            metadatas = [d["metadata"] for d in docs]

            embeddings = self.embeddings.embed_documents(texts)
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total += len(ids)

        return total

    @staticmethod
    def _format_source_ref(metadata: dict) -> str:
        source = metadata.get("source", "?")
        chapter = metadata.get("chapter", "?")
        page = metadata.get("page", metadata.get("start_page", "?"))
        return f"{source} - {chapter} - p.{page}"

    def _query_context_matches(
        self,
        query: str,
        n_results: int = 3,
        chapter_filter: Optional[str] = None,
    ) -> list[dict]:
        """Retourne les chunks les plus pertinents avec leur metadata."""
        if self.collection.count() == 0:
            return []

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
            return []

        matches = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            matches.append({
                "text": doc,
                "metadata": meta,
                "source_ref": self._format_source_ref(meta),
            })
        return matches

    def _build_context_block(self, matches: list[dict]) -> str:
        """Construit le bloc de contexte à partir des chunks sélectionnés."""
        if not matches:
            return ""

        context_parts = []
        for match in matches:
            meta = match["metadata"]
            ref = f"[{self._format_source_ref(meta)}]"
            if meta.get("has_visuals"):
                ref += " [Support visuel disponible]"
            context_parts.append(f"{ref}\n{match['text']}")

        return "\n\n---\n\n".join(context_parts)

    def search_context(self, query: str, n_results: int = 3, chapter_filter: str = None) -> str:
        """Recherche le contexte pertinent dans ChromaDB."""
        matches = self._query_context_matches(query, n_results=n_results, chapter_filter=chapter_filter)
        return self._build_context_block(matches)

    @staticmethod
    def _resolve_difficulty(avg_score: Optional[float], difficulty_level: Optional[str]) -> tuple[str, str]:
        """Détermine le niveau de difficulté effectif et son instruction associée."""
        if difficulty_level and difficulty_level in DIFFICULTY_INSTRUCTIONS:
            return difficulty_level, DIFFICULTY_INSTRUCTIONS[difficulty_level]

        if avg_score is None:
            return "Intermédiaire", DIFFICULTY_INSTRUCTIONS["Intermédiaire"]
        if avg_score >= 0.80:
            return "Élevé", DIFFICULTY_INSTRUCTIONS["Élevé"]
        if avg_score >= 0.50:
            return "Intermédiaire", DIFFICULTY_INSTRUCTIONS["Intermédiaire"]
        return "Fondamental", DIFFICULTY_INSTRUCTIONS["Fondamental"]

    def select_relevant_chapters(
        self,
        topic: str,
        excluded_sources: Optional[list[str]] = None,
        n_results: int = 3,
    ) -> list[dict]:
        """Sélectionne les chapitres PDF les plus pertinents pour un thème donné."""
        matches = self._query_context_matches(topic, n_results=max(6, n_results * 4))
        if not matches:
            return []

        excluded = set(excluded_sources or [])
        preferred_matches = []
        fallback_matches = []
        seen_refs = set()

        for match in matches:
            source_ref = match["source_ref"]
            if source_ref in seen_refs:
                continue
            seen_refs.add(source_ref)

            if source_ref in excluded:
                fallback_matches.append(match)
            else:
                preferred_matches.append(match)

        selected_matches = preferred_matches[:n_results]
        if len(selected_matches) < n_results:
            selected_matches.extend(
                fallback_matches[: n_results - len(selected_matches)]
            )

        return selected_matches

    @staticmethod
    def _question_needs_visual(question: str) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in VISUAL_KEYWORDS)

    def _resolve_visual_support(self, matches: list[dict], question: str) -> dict:
        """Retourne une preview de page quand un support visuel est disponible ou nécessaire."""
        question_needs_visual = self._question_needs_visual(question)

        for match in matches:
            metadata = match["metadata"]
            page_number = metadata.get("page") or metadata.get("start_page")
            source = metadata.get("source", "")
            if not source or not page_number:
                continue

            preview = get_page_preview(
                pdf_name=source,
                page_number=page_number,
                only_if_visual=not question_needs_visual,
            )
            if not preview:
                continue

            return {
                "image_path": preview["path"],
                "image_page": preview["page"],
                "image_has_visuals": preview["has_visuals"],
                "image_caption": f"Support visuel extrait de {source} - page {preview['page']}",
            }

        return {
            "image_path": None,
            "image_page": None,
            "image_has_visuals": False,
            "image_caption": "",
        }

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Nettoie les sections interdites dans l'énoncé généré."""
        cleaned_lines = []
        for line in question.splitlines():
            stripped_line = line.strip()
            stripped_lower = stripped_line.lower()
            if any(stripped_lower.startswith(pattern) for pattern in INLINE_SOURCE_PATTERNS):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(
            r"(?is)\b(réponse attendue|expected answer)\b.*$",
            "",
            cleaned,
        ).strip()
        return cleaned

    @staticmethod
    def _question_has_answer_leak(question: str) -> bool:
        normalized_question = question.lower()
        if any(pattern in normalized_question for pattern in INLINE_SOURCE_PATTERNS):
            return True

        leak_patterns = (
            r"(?i)\bla [a-zà-ÿ0-9_()Δ$\\ ]+ est\b",
            r"(?i)\ble [a-zà-ÿ0-9_()Δ$\\ ]+ est\b",
            r"(?i)\bc'?est[- ]à[- ]dire\b",
        )
        return any(re.search(pattern, question) for pattern in leak_patterns)

    def _repair_question(self, topic: str, question: str) -> str:
        """Réécrit un énoncé qui contient déjà la réponse ou une source inline."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Réécris la question d'entretien suivante.

Contraintes :
- Garde une seule question.
- Ne donne jamais la réponse.
- N'écris jamais "Réponse attendue", "Source" ou "Référence".
- Ne recopie pas la définition du cours : transforme-la en vraie question d'entretien.
- Ne mentionne ni livre, ni chapitre, ni page dans le texte.

Thème : {topic}
Question à réécrire :
{question}"""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        repaired_question = chain.invoke({
            "context": "",
            "topic": topic,
            "question": question,
        })
        return self._normalize_question(repaired_question)

    def generate_question(
        self,
        topic: str,
        history: str = "",
        avg_score: float = None,
        excluded_sources: Optional[list[str]] = None,
        difficulty_level: Optional[str] = None,
    ) -> dict:
        """Génère une question d'entretien à partir des chapitres PDF pertinents."""
        selected_matches = self.select_relevant_chapters(
            topic=topic,
            excluded_sources=excluded_sources,
        )
        if not selected_matches:
            raise ValueError(
                "Aucun chapitre PDF indexé n'est disponible pour générer une question."
            )

        context = self._build_context_block(selected_matches)
        context_block = f"\nContexte des livres de référence :\n{context}" if context else ""
        primary_match = selected_matches[0]
        resolved_difficulty, difficulty_block = self._resolve_difficulty(
            avg_score=avg_score,
            difficulty_level=difficulty_level,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Génère une question d'entretien sur le thème : {topic}
Appuie-toi prioritairement sur les extraits PDF fournis ci-dessus.
Choisis l'angle le plus pertinent parmi les chapitres remontés pour ce thème.
{difficulty_block}
{history_block}
Pose une seule question technique précise, sans donner la réponse ni écrire la source dans le texte."""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        raw_question = chain.invoke({
            "context": context_block,
            "topic": topic,
            "difficulty_block": difficulty_block,
            "history_block": f"\nQuestions déjà posées dans cette session :\n{history}" if history else "",
        })
        question = self._normalize_question(raw_question)

        if self._question_has_answer_leak(question):
            repaired_question = self._repair_question(topic=topic, question=question)
            if repaired_question:
                question = repaired_question

        source_refs = [match["source_ref"] for match in selected_matches]
        visual_support = self._resolve_visual_support(selected_matches, question)
        return {
            "question": question,
            "context": context,
            "source_ref": source_refs[0],
            "source_refs": source_refs,
            "chapter": primary_match["metadata"].get("chapter", ""),
            "source": primary_match["metadata"].get("source", ""),
            "page": primary_match["metadata"].get("page", primary_match["metadata"].get("start_page")),
            "difficulty": resolved_difficulty,
            **visual_support,
        }

    def evaluate_answer(self, question: str, answer: str, topic: str, context: str = "") -> dict:
        """Évalue la réponse de l'utilisateur."""
        reference_context = context or self.search_context(question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un évaluateur rigoureux d'entretiens Quant."),
            ("human", EVAL_PROMPT),
        ])

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "context": reference_context if reference_context else "Pas de contexte de référence disponible.",
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
