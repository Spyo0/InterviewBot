"""Je-Suis-Coach-AI : Simulateur d'entretien"""

import os
import time
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from engine import JeSuisCoachEngine, PROVIDERS
from database import (
    create_session,
    save_answer,
    update_mastery,
    get_all_mastery,
    get_response_times,
    get_answer_history,
)
from processor import list_pdfs

load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Je-Suis-Coach-AI",
    page_icon="Q",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Dark theme CSS — inspired by Linear / Vercel / Raycast
# Palette: near-black bg, soft white text, indigo accent, subtle borders
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Import Inter font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .block-container {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    header[data-testid="stHeader"] {
        background: rgba(10,10,10,0.8) !important;
        backdrop-filter: blur(12px);
    }

    /* ── Hide defaults ── */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ── Header branding ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.3rem;
    }
    .app-header .logo {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1rem; color: #fff;
    }
    .app-header .title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f5f5f5;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.82rem;
        color: #666;
        margin-bottom: 1.8rem;
        letter-spacing: 0.01em;
    }

    /* ── Tabs (Linear style) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #1e1e1e;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.65rem 1.3rem;
        font-weight: 500;
        font-size: 0.85rem;
        color: #555;
        border-bottom: 2px solid transparent;
        transition: color 0.15s, border-color 0.15s;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #999; }
    .stTabs [aria-selected="true"] {
        color: #e5e5e5 !important;
        border-bottom: 2px solid #818cf8 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #818cf8 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease !important;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        background: #6366f1 !important;
        box-shadow: 0 0 20px rgba(129,140,248,0.25) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #888 !important;
        border: 1px solid #2a2a2a !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #444 !important;
        color: #ccc !important;
        box-shadow: none !important;
    }

    /* ── Inputs & Select ── */
    .stTextArea textarea, .stTextInput input {
        background: #111 !important;
        border: 1px solid #222 !important;
        border-radius: 8px !important;
        color: #e5e5e5 !important;
        font-size: 0.9rem !important;
        transition: border-color 0.15s;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px rgba(129,140,248,0.3) !important;
    }
    .stTextArea textarea::placeholder { color: #444 !important; }

    .stSelectbox > div > div {
        background: #111 !important;
        border: 1px solid #222 !important;
        border-radius: 8px !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 8px !important;
        font-size: 0.85rem;
        color: #999 !important;
    }
    .streamlit-expanderContent {
        background: #0e0e0e !important;
        border: 1px solid #1e1e1e !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #111;
        border: 1px dashed #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #818cf8;
    }

    /* ── Cards (metric) ── */
    .metric-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color 0.15s;
    }
    .metric-card:hover { border-color: #2a2a2a; }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f5f5f5;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* ── Question block ── */
    .question-block {
        background: #111;
        border: 1px solid #1e1e1e;
        border-left: 3px solid #818cf8;
        padding: 1.3rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d4d4d4;
    }

    /* ── Score badges ── */
    .score-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-pass { background: rgba(34,197,94,0.15); color: #4ade80; }
    .score-mid  { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .score-fail { background: rgba(248,113,113,0.15); color: #f87171; }

    /* ── Mastery rows ── */
    .mastery-row {
        display: flex;
        align-items: center;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #141414;
        border-radius: 6px;
        transition: background 0.1s;
    }
    .mastery-row:hover { background: #111; }
    .mastery-topic {
        flex: 1;
        font-size: 0.88rem;
        color: #ccc;
        font-weight: 500;
    }
    .mastery-badge {
        font-size: 0.72rem;
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-mastered { background: rgba(34,197,94,0.12); color: #4ade80; }
    .badge-progress { background: rgba(251,191,36,0.12); color: #fbbf24; }
    .badge-new      { background: #141414; color: #444; }
    .mastery-detail {
        font-size: 0.72rem;
        color: #444;
        margin-left: 0.8rem;
        min-width: 110px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        margin-top: 0.5rem;
    }

    /* ── Dividers ── */
    .soft-divider {
        height: 1px;
        background: #1a1a1a;
        margin: 1.8rem 0;
    }

    /* ── Feedback text ── */
    .feedback-block {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-top: 0.8rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #b0b0b0;
    }
    .feedback-block strong { color: #e5e5e5; }

    /* ── PDF file list ── */
    .pdf-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.3rem;
        font-size: 0.88rem;
        color: #ccc;
    }
    .pdf-item:hover { background: #111; }
    .pdf-dot {
        width: 6px; height: 6px;
        background: #818cf8;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border: 1px solid #1e1e1e !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Timer ── */
    .timer-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 10px;
        margin: 0.8rem 0;
    }
    .timer-display {
        font-family: 'Inter', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e5e5;
        font-variant-numeric: tabular-nums;
    }
    .timer-display.warning { color: #fbbf24; }
    .timer-display.danger { color: #f87171; animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: 0.5; } }
    .timer-label {
        font-size: 0.7rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Exam mode ── */
    .exam-progress {
        display: flex;
        gap: 0.3rem;
        margin: 0.8rem 0;
    }
    .exam-dot {
        width: 28px; height: 4px;
        border-radius: 2px;
        background: #222;
    }
    .exam-dot.done { background: #818cf8; }
    .exam-dot.current { background: #4ade80; }

    /* ── Toast / alerts ── */
    .stAlert { border-radius: 8px !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


TOPICS = [
    "Calcul stochastique",
    "Probabilites",
    "Pricing de produits derives",
    "Brainteasers logiques",
    "Processus de Poisson",
    "Mouvement brownien",
    "Lemme d'Ito",
    "Black-Scholes",
    "Grecques (Greeks)",
    "Volatilite implicite",
    "Monte Carlo",
    "Calcul mental / Approximations",
]


# --- Init Engine ---
@st.cache_resource
def load_engine(provider: str, model: str):
    return JeSuisCoachEngine(provider=provider, model_name=model)


def init_session_state():
    defaults = {
        "engine": None,
        "session_id": None,
        "current_question": None,
        "current_question_topic": None,
        "current_question_context": "",
        "current_question_source": "",
        "question_start_time": None,
        "session_history": [],
        "questions_asked": [],
        "question_sources": [],
        # Mode examen
        "exam_mode": False,
        "exam_questions": [],
        "exam_question_meta": [],
        "exam_answers": [],
        "exam_index": 0,
        "exam_topic": None,
        "exam_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# --- Header ---
st.markdown(
    '<div class="app-header">'
    '<div class="logo">Q</div>'
    '<span class="title">Je-Suis-Coach-AI</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Simulateur d\'entretien — Finance quantitative</div>',
    unsafe_allow_html=True,
)

# --- Provider & Model selector ---
with st.expander("Parametres", expanded=False):
    default_provider = os.getenv("LLM_PROVIDER", "groq")
    provider_keys = list(PROVIDERS.keys())
    provider_labels = [PROVIDERS[k]["label"] for k in provider_keys]
    default_p_idx = provider_keys.index(default_provider) if default_provider in provider_keys else 0

    provider_choice = st.selectbox(
        "Provider", provider_labels, index=default_p_idx,
        help="Groq = ultra-rapide, HuggingFace = large choix de modeles"
    )
    provider_key = provider_keys[provider_labels.index(provider_choice)]

    model_options = PROVIDERS[provider_key]["models"]
    env_defaults = {
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "huggingface": os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    }
    default_model = env_defaults.get(provider_key, model_options[0])
    default_m_idx = model_options.index(default_model) if default_model in model_options else 0
    model_choice = st.selectbox("Modele", model_options, index=default_m_idx)

    if provider_key == "groq":
        st.caption("Groq : inference ultra-rapide, 0 RAM locale. Cle API requise dans .env")
    else:
        st.caption("HuggingFace : inference cloud gratuite. Token API requis dans .env")

# Load engine
engine_key = f"{provider_key}:{model_choice}"
if st.session_state["engine"] is None or st.session_state.get("_engine_key") != engine_key:
    with st.spinner("Chargement du moteur..."):
        try:
            st.session_state["engine"] = load_engine(provider_key, model_choice)
            st.session_state["_engine_key"] = engine_key
        except ValueError as e:
            st.error(str(e))
            st.stop()

engine: JeSuisCoachEngine = st.session_state["engine"]


# --- Helper: compute session avg score ---
def _session_avg_score() -> float | None:
    hist = st.session_state["session_history"]
    if not hist:
        return None
    return sum(h["score"] for h in hist) / len(hist)


# --- Helper: JS timer (no rerun) ---
def render_timer(duration_s: int = 180):
    """Inject a JS countdown timer that runs client-side without Streamlit reruns."""
    st.markdown(f"""
    <div class="timer-container">
        <span class="timer-label">Temps restant</span>
        <span class="timer-display" id="countdown-timer">{duration_s // 60}:{duration_s % 60:02d}</span>
    </div>
    <script>
        (function() {{
            var total = {duration_s};
            var el = document.getElementById('countdown-timer');
            if (!el) return;
            var iv = setInterval(function() {{
                total--;
                if (total <= 0) {{ clearInterval(iv); el.textContent = "0:00"; el.className = "timer-display danger"; return; }}
                var m = Math.floor(total / 60);
                var s = total % 60;
                el.textContent = m + ":" + (s < 10 ? "0" : "") + s;
                if (total <= 30) el.className = "timer-display danger";
                else if (total <= 60) el.className = "timer-display warning";
                else el.className = "timer-display";
            }}, 1000);
        }})();
    </script>
    """, unsafe_allow_html=True)


# --- Tabs ---
tab_interview, tab_exam, tab_pdf, tab_dashboard = st.tabs(["Entretien", "Examen", "PDF", "Dashboard"])


# ===================== TAB: ENTRETIEN =====================
with tab_interview:
    has_indexed_chapters = bool(engine.get_available_chapters())

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.selectbox("Theme", TOPICS, label_visibility="collapsed",
                             help="Choisissez un theme")
    with col2:
        timer_duration = st.selectbox("Timer", [0, 60, 120, 180, 300],
                                      format_func=lambda x: "Off" if x == 0 else f"{x//60}min",
                                      index=3, label_visibility="collapsed",
                                      help="Compte a rebours")

    st.caption("Le theme est choisi ici, puis le bot selectionne automatiquement les chapitres PDF les plus pertinents.")
    if not has_indexed_chapters:
        st.info("Indexez au moins un PDF pour generer des questions a partir de vos supports.")

    if st.button(
        "Nouvelle question",
        use_container_width=True,
        type="primary",
        disabled=not has_indexed_chapters,
    ):
        history_text = "\n".join(
            f"- {question_item}" for question_item in st.session_state["questions_asked"][-5:]
        )

        avg = _session_avg_score()

        try:
            with st.spinner("Generation..."):
                question_payload = engine.generate_question(
                    topic=topic,
                    history=history_text,
                    avg_score=avg,
                    excluded_sources=st.session_state["question_sources"],
                )
        except ValueError as exc:
            st.warning(str(exc))
        else:
            if st.session_state["session_id"] is None:
                st.session_state["session_id"] = create_session(topic)

            st.session_state["current_question"] = question_payload["question"]
            st.session_state["current_question_topic"] = topic
            st.session_state["current_question_context"] = question_payload["context"]
            st.session_state["current_question_source"] = question_payload["source_ref"]
            st.session_state["question_start_time"] = time.time()
            st.session_state["questions_asked"].append(question_payload["question"])
            st.session_state["question_sources"].append(question_payload["source_ref"])

    # Display question
    if st.session_state["current_question"]:
        # Timer
        if timer_duration > 0:
            render_timer(timer_duration)

        st.markdown(
            f'<div class="question-block">{st.session_state["current_question"]}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state["current_question_source"]:
            st.caption(f"Source PDF utilisee : {st.session_state['current_question_source']}")

        answer = st.text_area("Reponse", height=120, label_visibility="collapsed",
                              placeholder="Ecrivez votre reponse ici...")

        if st.button("Soumettre", use_container_width=True):
            if not answer.strip():
                st.warning("Entrez une reponse.")
            else:
                elapsed = time.time() - st.session_state["question_start_time"]

                with st.spinner("Evaluation..."):
                    evaluation = engine.evaluate_answer(
                        st.session_state["current_question"],
                        answer,
                        st.session_state["current_question_topic"] or topic,
                        context=st.session_state["current_question_context"],
                    )

                score = evaluation["score"]
                feedback = evaluation["feedback"]
                correction = evaluation["correction"]

                save_answer(
                    st.session_state["session_id"],
                    st.session_state["current_question"],
                    answer,
                    score,
                    elapsed,
                    feedback,
                    source_ref=st.session_state["current_question_source"],
                )
                update_mastery(
                    st.session_state["current_question_topic"] or topic,
                    score,
                    elapsed,
                )

                st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

                score_pct = int(score * 100)
                if score >= 0.70:
                    badge_class = "score-pass"
                elif score >= 0.50:
                    badge_class = "score-mid"
                else:
                    badge_class = "score-fail"

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="value"><span class="score-badge {badge_class}">{score_pct}%</span></div>'
                        f'<div class="label">Score</div></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="value">{elapsed:.1f}s</div>'
                        f'<div class="label">Temps de reponse</div></div>',
                        unsafe_allow_html=True,
                    )

                # Feedback
                fb_html = f'<div class="feedback-block"><strong>Feedback</strong><br>{feedback}'
                if correction and correction.lower() != "correct":
                    fb_html += f'<br><br><strong>Correction</strong><br>{correction}'
                fb_html += '</div>'
                st.markdown(fb_html, unsafe_allow_html=True)

                st.session_state["session_history"].append({
                    "question": st.session_state["current_question"],
                    "score": score,
                    "time": elapsed,
                    "source_ref": st.session_state["current_question_source"],
                })

    # Session history
    if st.session_state["session_history"]:
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historique de session</div>', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state["session_history"], 1):
            score_pct = int(item["score"] * 100)
            if item["score"] >= 0.70:
                badge_class = "score-pass"
            elif item["score"] >= 0.50:
                badge_class = "score-mid"
            else:
                badge_class = "score-fail"
            with st.expander(f"Q{i}  ·  {score_pct}%  ·  {item['time']:.1f}s"):
                st.write(item["question"])
                if item.get("source_ref"):
                    st.caption(f"Source : {item['source_ref']}")

    # Reset
    if st.session_state["session_id"] is not None:
        st.markdown("")
        if st.button("Reinitialiser la session", type="secondary"):
            st.session_state["session_id"] = None
            st.session_state["current_question"] = None
            st.session_state["current_question_topic"] = None
            st.session_state["current_question_context"] = ""
            st.session_state["current_question_source"] = ""
            st.session_state["session_history"] = []
            st.session_state["questions_asked"] = []
            st.session_state["question_sources"] = []
            st.rerun()


# ===================== TAB: EXAMEN =====================
EXAM_COUNT = 10

with tab_exam:
    if not st.session_state["exam_mode"] and st.session_state["exam_results"] is None:
        has_indexed_chapters = bool(engine.get_available_chapters())

        # Config exam
        st.markdown('<div class="section-title">Mode examen</div>', unsafe_allow_html=True)
        st.caption(f"{EXAM_COUNT} questions d'affilee · Pas de feedback entre les questions · Correction a la fin")
        st.caption("Le theme guide la recherche, puis les chapitres PDF pertinents sont selectionnes automatiquement.")

        exam_topic = st.selectbox("Theme de l'examen", TOPICS, key="exam_topic_select")
        if not has_indexed_chapters:
            st.info("Indexez au moins un PDF avant de lancer l'examen.")

        if st.button(
            "Lancer l'examen",
            use_container_width=True,
            type="primary",
            disabled=not has_indexed_chapters,
        ):
            try:
                with st.spinner("Preparation de l'examen..."):
                    question_payload = engine.generate_question(topic=exam_topic)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                st.session_state["exam_mode"] = True
                st.session_state["exam_topic"] = exam_topic
                st.session_state["exam_questions"] = [question_payload["question"]]
                st.session_state["exam_question_meta"] = [question_payload]
                st.session_state["exam_answers"] = []
                st.session_state["exam_index"] = 0
                st.session_state["exam_results"] = None
                st.session_state["session_id"] = create_session(exam_topic, "auto_pdf")
                st.session_state["question_start_time"] = time.time()
                st.rerun()

    elif st.session_state["exam_mode"]:
        idx = st.session_state["exam_index"]
        total = EXAM_COUNT

        # Progress bar
        dots_html = '<div class="exam-progress">'
        for i in range(total):
            if i < idx:
                dots_html += '<div class="exam-dot done"></div>'
            elif i == idx:
                dots_html += '<div class="exam-dot current"></div>'
            else:
                dots_html += '<div class="exam-dot"></div>'
        dots_html += '</div>'
        st.markdown(dots_html, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">Question {idx + 1} / {total}</div>', unsafe_allow_html=True)

        # Timer
        render_timer(180)

        # Question
        current_q = st.session_state["exam_questions"][idx]
        current_question_meta = st.session_state["exam_question_meta"][idx]
        st.markdown(
            f'<div class="question-block">{current_q}</div>',
            unsafe_allow_html=True,
        )
        if current_question_meta.get("source_ref"):
            st.caption(f"Source PDF utilisee : {current_question_meta['source_ref']}")

        answer = st.text_area("Reponse", height=120, label_visibility="collapsed",
                              placeholder="Ecrivez votre reponse...", key=f"exam_answer_{idx}")

        if st.button("Question suivante" if idx < total - 1 else "Terminer l'examen",
                      use_container_width=True, type="primary"):
            if not answer.strip():
                st.warning("Entrez une reponse.")
            else:
                elapsed = time.time() - st.session_state["question_start_time"]
                st.session_state["exam_answers"].append({
                    "answer": answer,
                    "time": elapsed,
                })

                if idx < total - 1:
                    # Generate next question
                    try:
                        with st.spinner("Question suivante..."):
                            question_payload = engine.generate_question(
                                topic=st.session_state["exam_topic"],
                                history="\n".join(
                                    f"- {question_item}" for question_item in st.session_state["exam_questions"][-5:]
                                ),
                                excluded_sources=[
                                    item["source_ref"] for item in st.session_state["exam_question_meta"]
                                ],
                            )
                    except ValueError as exc:
                        st.warning(str(exc))
                    else:
                        st.session_state["exam_questions"].append(question_payload["question"])
                        st.session_state["exam_question_meta"].append(question_payload)
                        st.session_state["exam_index"] = idx + 1
                        st.session_state["question_start_time"] = time.time()
                        st.rerun()
                else:
                    # End exam — evaluate all answers
                    st.session_state["exam_mode"] = False
                    results = []
                    with st.spinner("Correction de l'examen..."):
                        for i, (q, a, meta) in enumerate(zip(
                            st.session_state["exam_questions"],
                            st.session_state["exam_answers"],
                            st.session_state["exam_question_meta"],
                        )):
                            evaluation = engine.evaluate_answer(
                                q,
                                a["answer"],
                                st.session_state["exam_topic"],
                                context=meta.get("context", ""),
                            )
                            elapsed = a["time"]
                            save_answer(
                                st.session_state["session_id"],
                                q,
                                a["answer"],
                                evaluation["score"],
                                elapsed,
                                evaluation["feedback"],
                                source_ref=meta.get("source_ref", ""),
                            )
                            update_mastery(st.session_state["exam_topic"], evaluation["score"], elapsed)
                            results.append({
                                "question": q,
                                "answer": a["answer"],
                                "time": elapsed,
                                "source_ref": meta.get("source_ref", ""),
                                **evaluation,
                            })
                    st.session_state["exam_results"] = results
                    st.rerun()

    elif st.session_state["exam_results"] is not None:
        # Display results
        results = st.session_state["exam_results"]
        avg = sum(r["score"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        passed = sum(1 for r in results if r["score"] >= 0.70)

        st.markdown('<div class="section-title">Resultats de l\'examen</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            badge = "score-pass" if avg >= 0.70 else ("score-mid" if avg >= 0.50 else "score-fail")
            st.markdown(
                f'<div class="metric-card"><div class="value">'
                f'<span class="score-badge {badge}">{int(avg*100)}%</span>'
                f'</div><div class="label">Score moyen</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="value">{passed}/{len(results)}</div>'
                f'<div class="label">Questions validees</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="value">{avg_time:.0f}s</div>'
                f'<div class="label">Temps moyen</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Correction detaillee</div>', unsafe_allow_html=True)

        for i, r in enumerate(results, 1):
            score_pct = int(r["score"] * 100)
            if r["score"] >= 0.70:
                badge_class = "score-pass"
            elif r["score"] >= 0.50:
                badge_class = "score-mid"
            else:
                badge_class = "score-fail"
            with st.expander(f"Q{i}  ·  {score_pct}%  ·  {r['time']:.0f}s"):
                st.markdown(f"**Question :** {r['question']}")
                if r.get("source_ref"):
                    st.markdown(f"**Source :** {r['source_ref']}")
                st.markdown(f"**Reponse :** {r['answer']}")
                st.markdown(f"**Feedback :** {r['feedback']}")
                if r["correction"] and r["correction"].lower() != "correct":
                    st.markdown(f"**Correction :** {r['correction']}")

        if st.button("Nouvel examen", use_container_width=True, type="primary"):
            st.session_state["exam_mode"] = False
            st.session_state["exam_questions"] = []
            st.session_state["exam_question_meta"] = []
            st.session_state["exam_answers"] = []
            st.session_state["exam_index"] = 0
            st.session_state["exam_results"] = None
            st.session_state["exam_topic"] = None
            st.session_state["session_id"] = None
            st.rerun()


# ===================== TAB: PDF =====================
with tab_pdf:
    st.markdown('<div class="section-title">Importer des PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        existing = list_pdfs()
        for f in uploaded:
            if len(existing) >= 5:
                st.warning("Maximum 5 PDF atteint.")
                break
            if f.name not in existing:
                path = f"data/{f.name}"
                with open(path, "wb") as out:
                    out.write(f.read())
                existing.append(f.name)
                st.success(f"{f.name} ajoute.")

    # List
    pdfs = list_pdfs()
    if pdfs:
        st.markdown('<div class="section-title">Fichiers disponibles</div>', unsafe_allow_html=True)
        for pdf in pdfs:
            st.markdown(
                f'<div class="pdf-item"><span class="pdf-dot"></span>{pdf}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Aucun PDF. Deposez vos fichiers ci-dessus.")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    if st.button("Indexer les PDF", type="primary", use_container_width=True):
        with st.spinner("Indexation en cours..."):
            n = engine.index_pdfs()
        st.success(f"{n} chunks indexes.")

    chapters = engine.get_available_chapters()
    if chapters:
        st.markdown('<div class="section-title">Chapitres indexes</div>', unsafe_allow_html=True)
        st.caption("Ces chapitres sont ensuite selectionnes automatiquement en fonction du theme choisi.")
        for ch in chapters:
            st.markdown(
                f'<div class="pdf-item"><span class="pdf-dot"></span>{ch}</div>',
                unsafe_allow_html=True,
            )


# ===================== TAB: DASHBOARD =====================
with tab_dashboard:
    # Summary cards
    mastery_data = get_all_mastery()
    mastery_map = {m["topic"]: m for m in mastery_data} if mastery_data else {}
    times_data = get_response_times()

    total_q = len(times_data) if times_data else 0
    mastered = sum(1 for m in mastery_data if m["status"] == "Maitrise") if mastery_data else 0
    avg_score = sum(m["best_score"] for m in mastery_data) / len(mastery_data) * 100 if mastery_data else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="value">{total_q}</div>'
            f'<div class="label">Questions</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="value">{mastered}/{len(TOPICS)}</div>'
            f'<div class="label">Maitrises</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="value">{avg_score:.0f}%</div>'
            f'<div class="label">Score moyen</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # Mastery matrix
    st.markdown('<div class="section-title">Maitrise par theme</div>', unsafe_allow_html=True)

    for t in TOPICS:
        m = mastery_map.get(t)
        if m:
            status = m["status"]
            best = int(m["best_score"] * 100)
            avg_t = m["avg_time_s"]
            attempts = m["attempts"]
            if status == "Maitrise":
                badge = '<span class="mastery-badge badge-mastered">Maitrise</span>'
            elif status == "En cours":
                badge = '<span class="mastery-badge badge-progress">En cours</span>'
            else:
                badge = '<span class="mastery-badge badge-new">—</span>'
            detail = f'{best}% · {avg_t:.0f}s · {attempts}x'
        else:
            badge = '<span class="mastery-badge badge-new">—</span>'
            detail = ""

        st.markdown(
            f'<div class="mastery-row">'
            f'<span class="mastery-topic">{t}</span>'
            f'{badge}'
            f'<span class="mastery-detail">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # Response time chart
    st.markdown('<div class="section-title">Temps de reponse</div>', unsafe_allow_html=True)

    if times_data:
        df_times = pd.DataFrame(times_data)
        df_times["created_at"] = pd.to_datetime(df_times["created_at"])

        fig = px.scatter(
            df_times, x="created_at", y="response_time_s",
            color="score",
            color_continuous_scale=["#f87171", "#fbbf24", "#4ade80"],
            labels={"created_at": "", "response_time_s": "Temps (s)", "score": "Score"},
        )
        fig.add_trace(go.Scatter(
            x=df_times["created_at"],
            y=df_times["response_time_s"].rolling(5, min_periods=1).mean(),
            mode="lines", name="Tendance",
            line=dict(color="#818cf8", width=2, dash="dot"),
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            showlegend=False,
            font=dict(color="#666", size=11),
            xaxis=dict(showgrid=False, color="#444"),
            yaxis=dict(gridcolor="#1a1a1a", color="#444"),
            coloraxis_colorbar=dict(
                tickfont=dict(color="#555"),
                title=dict(font=dict(color="#555")),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas encore de donnees. Lancez une simulation.")

    # Recent history
    history = get_answer_history(15)
    if history:
        st.markdown('<div class="section-title">Historique recent</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(history)[
            ["session_topic", "source_ref", "question", "score", "response_time_s", "created_at"]
        ]
        df_hist.columns = ["Theme", "Source", "Question", "Score", "Temps", "Date"]
        df_hist["Score"] = (df_hist["Score"] * 100).astype(int).astype(str) + "%"
        df_hist["Temps"] = df_hist["Temps"].round(1).astype(str) + "s"
        df_hist["Source"] = df_hist["Source"].fillna("").replace("", "—")
        df_hist["Source"] = df_hist["Source"].apply(
            lambda value: value if value == "—" or len(value) <= 55 else value[:55] + "..."
        )
        df_hist["Question"] = df_hist["Question"].apply(
            lambda value: value if len(value) <= 70 else value[:70] + "..."
        )
        st.dataframe(df_hist, use_container_width=True, hide_index=True, height=300)
