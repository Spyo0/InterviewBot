"""Logique RAG : LangChain + ChromaDB + providers cloud (Groq / HuggingFace)."""

import json
import logging
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

from processor import get_page_preview, process_pdf, list_pdfs

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
MASTERY_SHEETS_DIR = os.path.join(DATA_DIR, "mastery_sheets")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "0.70"))
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))
SOURCE_SIMILARITY_LIMIT = float(os.getenv("SOURCE_SIMILARITY_LIMIT", "0.82"))

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
QUESTION_TYPES_BY_DIFFICULTY = {
    "Fondamental": ["definition", "application", "mental_math"],
    "Intermédiaire": ["application", "definition", "proof", "mental_math"],
    "Élevé": ["application", "proof", "visual_interpretation", "mental_math"],
}
QUESTION_TYPE_LABELS = {
    "definition": "Compréhension / définition",
    "application": "Application pratique",
    "mental_math": "Calcul mental / approximation",
    "proof": "Raisonnement / démonstration",
    "visual_interpretation": "Interprétation de support visuel",
}
QUESTION_TYPE_INSTRUCTIONS = {
    "definition": (
        "Pose une question de compréhension ou de reformulation. "
        "Ne demande pas de réciter une définition mot pour mot : exige une explication utile en entretien."
    ),
    "application": (
        "Pose une question d'application du concept à un cas simple ou réaliste. "
        "L'utilisateur doit expliquer comment utiliser l'idée, pas seulement la nommer."
    ),
    "mental_math": (
        "Pose un calcul mental ou une approximation rapide, en gardant une difficulté raisonnable "
        "et sans exiger de contexte affiché si ce n'est pas indispensable."
    ),
    "proof": (
        "Pose une question de raisonnement rigoureux ou de mini-démonstration. "
        "La réponse attendue doit nécessiter des étapes, pas une phrase isolée."
    ),
    "visual_interpretation": (
        "Pose une question qui dépend explicitement d'une figure, d'un tableau, d'un schéma ou d'un graphe présent dans le support."
    ),
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
CONTEXT_DISPLAY_KEYWORDS = (
    "extrait",
    "figure",
    "schéma",
    "schema",
    "graphe",
    "graphique",
    "diagramme",
    "tableau",
    "illustr",
    "comme montré",
    "comme illustre",
    "comme illustré",
    "comme vu",
    "selon le schéma",
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
TYPE_PREFIX_PATTERNS = (
    "type :",
    "type:",
    "question :",
    "question:",
)


class APIError(Exception):
    """Erreur remontée quand le provider LLM ou embeddings est injoignable."""


def _is_retryable(exc: BaseException) -> bool:
    """Retourne True pour les erreurs transitoires (rate limit, timeout, 5xx)."""
    msg = str(exc).lower()
    retryable_keywords = ("rate limit", "429", "503", "502", "timeout", "connection", "overloaded")
    return any(keyword in msg for keyword in retryable_keywords)


_llm_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
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
- N'évoque pas explicitement un extrait de livre ou un support PDF si la question peut se suffire à elle-même.
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
4. Une liste courte des points corrects si la réponse contient des éléments valables
5. Une liste courte des erreurs ou manques prioritaires
6. La source la plus utile du contexte si elle est identifiable

Retourne EXCLUSIVEMENT un objet JSON valide au format suivant :
{{
  "score": 0.0,
  "feedback": "feedback détaillé en français",
  "correction": "correction complète en LaTeX si nécessaire, sinon 'Correct'",
  "strengths": ["point fort 1", "point fort 2"],
  "mistakes": ["erreur 1", "erreur 2"],
  "source_used": "source utile ou chaîne vide"
}}"""


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

    def _search_mastery_sheets(self, query: str, n_results: int = 2) -> list[dict]:
        """Recherche prioritaire dans les Fiches de Maîtrise par correspondance de mots-clés."""
        if not os.path.exists(MASTERY_SHEETS_DIR):
            return []

        query_words = set(re.sub(r"[^\w\s]", " ", query.lower()).split())
        stop_words = {"de", "du", "le", "la", "les", "un", "une", "des", "en", "et", "ou", "est",
                      "que", "qui", "par", "sur", "pour", "dans", "avec", "il", "si", "au", "aux"}
        query_words -= stop_words

        candidates = []
        for fname in os.listdir(MASTERY_SHEETS_DIR):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(MASTERY_SHEETS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue

            content_lower = content.lower()
            score = sum(1 for w in query_words if w in content_lower)
            if score > 0:
                sheet_name = fname.replace(".md", "").replace("_", " ").title()
                candidates.append({
                    "text": content,
                    "score": score,
                    "metadata": {
                        "source": fname,
                        "doc_type": "mastery_sheet",
                        "chapter": sheet_name,
                        "page": "—",
                    },
                    "source_ref": f"📋 Fiche de Maîtrise : {sheet_name}",
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:n_results]

    def _build_mastery_context_block(self, sheets: list[dict]) -> str:
        """Construit le bloc de contexte pour les fiches de maîtrise."""
        if not sheets:
            return ""
        parts = []
        for sheet in sheets:
            ref = f"[{sheet['source_ref']}]"
            parts.append(f"{ref}\n{sheet['text']}")
        return "\n\n---\n\n".join(parts)

    def search_context(self, query: str, n_results: int = 3, chapter_filter: str = None) -> str:
        """Recherche le contexte : Fiches de Maîtrise en priorité, puis PDFs."""
        # Priorité 1 : Fiches de maîtrise (savoir structuré + limites + pièges)
        mastery_sheets = self._search_mastery_sheets(query, n_results=1)
        mastery_block = self._build_mastery_context_block(mastery_sheets)

        # Priorité 2 : Chunks PDF (source brute indexée)
        pdf_matches = self._query_context_matches(query, n_results=n_results, chapter_filter=chapter_filter)
        pdf_block = self._build_context_block(pdf_matches)

        blocks = [b for b in [mastery_block, pdf_block] if b]
        return "\n\n---\n\n".join(blocks)

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

    @staticmethod
    def _question_needs_context_display(question: str) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in CONTEXT_DISPLAY_KEYWORDS)

    @staticmethod
    def _normalize_similarity_text(text: str) -> str:
        normalized = re.sub(r"\$.*?\$", " ", text)
        normalized = re.sub(r"[^a-z0-9à-ÿ]+", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _select_question_type(
        self,
        topic: str,
        difficulty: str,
        selected_matches: list[dict],
        recent_question_types: Optional[list[str]] = None,
    ) -> str:
        """Choisit un type de question cohérent avec la difficulté et varie la session."""
        available_types = list(QUESTION_TYPES_BY_DIFFICULTY[difficulty])
        normalized_topic = topic.lower()
        if "calcul mental" not in normalized_topic and "approximation" not in normalized_topic:
            available_types = [
                question_type for question_type in available_types
                if question_type != "mental_math"
            ]

        has_visual_support = any(match["metadata"].get("has_visuals") for match in selected_matches)
        if not has_visual_support:
            available_types = [
                question_type for question_type in available_types
                if question_type != "visual_interpretation"
            ]

        recent_counts = Counter(recent_question_types or [])
        base_priority = {question_type: index for index, question_type in enumerate(available_types)}
        available_types.sort(key=lambda question_type: (recent_counts[question_type], base_priority[question_type]))
        return available_types[0] if available_types else "application"

    @staticmethod
    def _parse_generated_question(raw: str, fallback_question_type: str) -> dict:
        """Parse une génération structurée TYPE/QUESTION tout en gardant un fallback robuste."""
        type_match = re.search(r"(?im)^TYPE:\s*(.+)$", raw)
        question_match = re.search(r"(?ims)^QUESTION:\s*(.+)$", raw)

        question_type = fallback_question_type
        if type_match:
            raw_question_type = type_match.group(1).strip().lower()
            if raw_question_type in QUESTION_TYPE_LABELS:
                question_type = raw_question_type

        question = question_match.group(1).strip() if question_match else raw.strip()
        return {
            "question_type": question_type,
            "question": question,
        }

    def _max_source_similarity(self, question: str, matches: list[dict]) -> float:
        """Mesure si la question recopie trop directement le support source."""
        normalized_question = self._normalize_similarity_text(question)
        if not normalized_question:
            return 0.0

        candidates = []
        for match in matches:
            raw_text = match["text"]
            candidates.append(raw_text)
            candidates.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", raw_text)
                if len(sentence.strip()) >= 40
            )

        best_similarity = 0.0
        for candidate in candidates:
            normalized_candidate = self._normalize_similarity_text(candidate)
            if not normalized_candidate:
                continue
            similarity = SequenceMatcher(None, normalized_question, normalized_candidate).ratio()
            best_similarity = max(best_similarity, similarity)

        return best_similarity

    def _validate_generated_question(
        self,
        question: str,
        question_type: str,
        matches: list[dict],
        recent_questions: Optional[list[str]] = None,
    ) -> dict:
        """Valide un énoncé avant affichage pour éviter les questions faibles ou trop copiées."""
        issues = []
        validation_score = 1.0
        word_count = len(question.split())

        if self._question_has_answer_leak(question):
            issues.append("answer_leak")
            validation_score -= 0.70

        normalized_question = question.lower()
        if any(pattern in normalized_question for pattern in INLINE_SOURCE_PATTERNS):
            issues.append("inline_source")
            validation_score -= 0.50

        min_word_count = 7 if question_type == "mental_math" else 10
        if word_count < min_word_count:
            issues.append("too_short")
            validation_score -= 0.20

        source_similarity = self._max_source_similarity(question, matches)
        if source_similarity >= SOURCE_SIMILARITY_LIMIT:
            issues.append("too_close_to_source")
            validation_score -= 0.35

        if recent_questions:
            normalized_recent_questions = {
                self._normalize_similarity_text(previous_question)
                for previous_question in recent_questions[-5:]
            }
            if self._normalize_similarity_text(question) in normalized_recent_questions:
                issues.append("duplicate_question")
                validation_score -= 0.25

        if question_type == "visual_interpretation" and not any(
            match["metadata"].get("has_visuals") for match in matches
        ):
            issues.append("missing_visual_support")
            validation_score -= 0.30

        return {
            "is_valid": validation_score >= VALIDATION_THRESHOLD and not issues,
            "score": max(0.0, validation_score),
            "issues": issues,
            "source_similarity": source_similarity,
        }

    @staticmethod
    def _build_retry_guidance(validation: dict) -> str:
        """Construit une consigne de régénération à partir des défauts constatés."""
        if not validation["issues"]:
            return ""

        issue_messages = {
            "answer_leak": "- La question précédente contenait déjà un élément de réponse. Reformule sans donner la solution.",
            "inline_source": "- La question précédente réintégrait la source dans l'énoncé. N'affiche aucune référence de support dans le texte.",
            "too_short": "- La question précédente était trop courte. Pose une vraie question d'entretien avec assez de matière pour être évaluée.",
            "too_close_to_source": "- La question précédente copiait trop directement le support. Reformule-la sous un angle d'entretien, de compréhension ou d'application.",
            "duplicate_question": "- La question précédente répétait trop ce qui a déjà été demandé. Choisis un autre angle.",
            "missing_visual_support": "- La question précédente demandait une interprétation visuelle sans support exploitable. Choisis un autre type de question.",
        }
        retry_lines = [issue_messages[issue] for issue in validation["issues"] if issue in issue_messages]
        return "\n".join(retry_lines)

    def _resolve_visual_support(self, matches: list[dict], question: str) -> dict:
        """Retourne une preview de page quand un support visuel est disponible ou nécessaire."""
        question_needs_visual = self._question_needs_visual(question)
        if not question_needs_visual:
            return {
                "image_path": None,
                "image_page": None,
                "image_has_visuals": False,
                "image_caption": "",
            }

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
            if any(stripped_lower.startswith(pattern) for pattern in INLINE_SOURCE_PATTERNS + TYPE_PREFIX_PATTERNS):
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
        try:
            repaired_question = _llm_retry(chain.invoke)({
                "context": "",
                "topic": topic,
                "question": question,
            })
        except Exception:
            return question
        return self._normalize_question(repaired_question)

    def generate_question(
        self,
        topic: str,
        history: str = "",
        avg_score: float = None,
        excluded_sources: Optional[list[str]] = None,
        difficulty_level: Optional[str] = None,
        recent_questions: Optional[list[str]] = None,
        recent_question_types: Optional[list[str]] = None,
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
        selected_question_type = self._select_question_type(
            topic=topic,
            difficulty=resolved_difficulty,
            selected_matches=selected_matches,
            recent_question_types=recent_question_types,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Génère une question d'entretien sur le thème : {topic}
Appuie-toi prioritairement sur les extraits PDF fournis ci-dessus.
Choisis l'angle le plus pertinent parmi les chapitres remontés pour ce thème.
Type de question imposé : {question_type_label}
Consigne spécifique au type :
{question_type_instruction}
{difficulty_block}
{history_block}
{retry_guidance}
Format EXACT :
TYPE: {question_type}
QUESTION: [une seule question technique précise, sans donner la réponse ni écrire la source dans le texte]"""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        best_candidate = None
        retry_guidance = ""
        question_type = selected_question_type
        question = ""

        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                raw_question = _llm_retry(chain.invoke)({
                    "context": context_block,
                    "topic": topic,
                    "question_type": question_type,
                    "question_type_label": QUESTION_TYPE_LABELS[question_type],
                    "question_type_instruction": QUESTION_TYPE_INSTRUCTIONS[question_type],
                    "difficulty_block": difficulty_block,
                    "history_block": f"\nQuestions déjà posées dans cette session :\n{history}" if history else "",
                    "retry_guidance": retry_guidance,
                })
            except Exception as exc:
                raise APIError(
                    f"Le provider LLM ({self.provider}) est injoignable après 3 tentatives. "
                    f"Vérifiez votre clé API et votre connexion.\nDétail : {exc}"
                ) from exc

            parsed_candidate = self._parse_generated_question(raw_question, fallback_question_type=question_type)
            question_type = parsed_candidate["question_type"]
            question = self._normalize_question(parsed_candidate["question"])

            if self._question_has_answer_leak(question):
                repaired_question = self._repair_question(topic=topic, question=question)
                if repaired_question:
                    question = repaired_question

            validation = self._validate_generated_question(
                question=question,
                question_type=question_type,
                matches=selected_matches,
                recent_questions=recent_questions,
            )
            candidate = {
                "question": question,
                "question_type": question_type,
                "validation": validation,
            }
            if best_candidate is None or validation["score"] > best_candidate["validation"]["score"]:
                best_candidate = candidate

            if validation["is_valid"]:
                break

            retry_guidance = self._build_retry_guidance(validation)
            if "missing_visual_support" in validation["issues"]:
                question_type = self._select_question_type(
                    topic=topic,
                    difficulty=resolved_difficulty,
                    selected_matches=selected_matches,
                    recent_question_types=(recent_question_types or []) + [question_type],
                )
        else:
            if best_candidate is not None:
                question = best_candidate["question"]
                question_type = best_candidate["question_type"]

        source_refs = [match["source_ref"] for match in selected_matches]
        should_display_context = self._question_needs_context_display(question)
        visual_support = self._resolve_visual_support(selected_matches, question)
        return {
            "question": question,
            "question_type": question_type,
            "question_type_label": QUESTION_TYPE_LABELS.get(question_type, "Application pratique"),
            "context": context,
            "source_ref": source_refs[0],
            "display_source_ref": source_refs[0] if should_display_context else "",
            "source_refs": source_refs,
            "chapter": primary_match["metadata"].get("chapter", ""),
            "source": primary_match["metadata"].get("source", ""),
            "page": primary_match["metadata"].get("page", primary_match["metadata"].get("start_page")),
            "difficulty": resolved_difficulty,
            "show_context_support": should_display_context,
            **visual_support,
        }

    def generate_followup(
        self,
        topic: str,
        conversation: list[dict],
        context: str = "",
    ) -> str:
        """Génère une question de relance basée sur l'échange précédent.

        `conversation` est une liste de dicts {"role": "question"|"answer", "content": str}.
        """
        history_lines = []
        for turn in conversation[-4:]:
            prefix = "Q:" if turn["role"] == "question" else "R:"
            history_lines.append(f"{prefix} {turn['content']}")
        history_text = "\n".join(history_lines)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Tu suis un entretien en cours sur le thème : {topic}

Voici les derniers échanges :
{history}

Pose UNE question de relance courte et précise :
- Si la réponse était correcte : approfondis ou déplace-toi vers un cas limite.
- Si la réponse était incomplète : cible le point manquant précis sans répéter la question.
- Si la réponse était incorrecte : reformule pour guider sans donner la réponse.

Ne répète pas la question précédente. Pose une seule question de suivi."""),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": f"\nContexte de référence :\n{context}" if context else "",
            "topic": topic,
            "history": history_text,
        }).strip()

    def evaluate_answer(self, question: str, answer: str, topic: str, context: str = "") -> dict:
        """Évalue la réponse de l'utilisateur."""
        reference_context = context or self.search_context(question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un évaluateur rigoureux d'entretiens Quant."),
            ("human", EVAL_PROMPT),
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = _llm_retry(chain.invoke)({
                "question": question,
                "answer": answer,
                "context": reference_context if reference_context else "Pas de contexte de référence disponible.",
            })
        except Exception as exc:
            raise APIError(
                f"Le provider LLM ({self.provider}) est injoignable lors de l'évaluation. "
                f"Vérifiez votre clé API et votre connexion.\nDétail : {exc}"
            ) from exc

        return self._parse_structured_evaluation(raw)

    @staticmethod
    def _extract_json_payload(raw: str) -> str:
        """Extrait un JSON même si le modèle l'encadre dans un bloc Markdown."""
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw

        json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        return json_match.group(1) if json_match else raw

    def _parse_structured_evaluation(self, raw: str) -> dict:
        """Parse une évaluation JSON et retombe sur un fallback robuste si besoin."""
        default_result = {
            "score": 0.5,
            "feedback": raw.strip(),
            "correction": "",
            "strengths": [],
            "mistakes": [],
            "source_used": "",
        }

        try:
            parsed = json.loads(self._extract_json_payload(raw))
        except json.JSONDecodeError:
            return self._fallback_parse_evaluation(raw, default_result)

        if not isinstance(parsed, dict):
            return default_result

        score = parsed.get("score", default_result["score"])
        try:
            parsed_score = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            parsed_score = default_result["score"]

        return {
            "score": parsed_score,
            "feedback": str(parsed.get("feedback") or default_result["feedback"]).strip(),
            "correction": str(parsed.get("correction") or "").strip(),
            "strengths": self._coerce_string_list(parsed.get("strengths")),
            "mistakes": self._coerce_string_list(parsed.get("mistakes")),
            "source_used": str(parsed.get("source_used") or "").strip(),
        }

    @staticmethod
    def _coerce_string_list(value) -> list[str]:
        """Normalise une valeur vers une liste de chaînes exploitables."""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _fallback_parse_evaluation(self, raw: str, default_result: dict) -> dict:
        """Fallback de parsing pour les modèles qui ne renvoient pas un JSON propre."""
        result = default_result.copy()

        for line in raw.split("\n"):
            stripped_line = line.strip()
            upper_line = stripped_line.upper()
            if upper_line.startswith("SCORE:"):
                try:
                    score_str = stripped_line.split(":", 1)[1].strip()
                    result["score"] = max(0.0, min(1.0, float(score_str)))
                except (IndexError, ValueError):
                    pass
            elif upper_line.startswith("FEEDBACK:"):
                result["feedback"] = stripped_line.split(":", 1)[1].strip()
            elif upper_line.startswith("CORRECTION:"):
                result["correction"] = stripped_line.split(":", 1)[1].strip()
            elif upper_line.startswith("SOURCE_USED:"):
                result["source_used"] = stripped_line.split(":", 1)[1].strip()

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
