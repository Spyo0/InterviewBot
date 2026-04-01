"""Je-Suis-Coach-AI : Simulateur d'entretien"""

import os
import time
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from engine import JeSuisCoachEngine
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
def load_engine(model: str):
    return JeSuisCoachEngine(model_name=model)


def init_session_state():
    defaults = {
        "engine": None,
        "session_id": None,
        "current_question": None,
        "question_start_time": None,
        "session_history": [],
        "questions_asked": [],
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

# --- Model selector ---
with st.expander("Parametres", expanded=False):
    default_model = os.getenv("OLLAMA_MODEL", "llama3")
    model_options = ["llama3", "mistral", "llama3.1"]
    default_index = model_options.index(default_model) if default_model in model_options else 0
    model_choice = st.selectbox("Modele Ollama", model_options, index=default_index, label_visibility="collapsed")

# Load engine
if st.session_state["engine"] is None or st.session_state.get("_model") != model_choice:
    with st.spinner("Chargement du moteur..."):
        st.session_state["engine"] = load_engine(model_choice)
        st.session_state["_model"] = model_choice

engine: JeSuisCoachEngine = st.session_state["engine"]


# --- Tabs ---
tab_interview, tab_pdf, tab_dashboard = st.tabs(["Entretien", "PDF", "Dashboard"])


# ===================== TAB: ENTRETIEN =====================
with tab_interview:
    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox("Theme", TOPICS, label_visibility="collapsed",
                             help="Choisissez un theme")
    with col2:
        chapters = engine.get_available_chapters()
        chapter = st.selectbox("Chapitre", ["Tous"] + chapters,
                               label_visibility="collapsed",
                               help="Filtrer par chapitre")
        if chapter == "Tous":
            chapter = None

    if st.button("Nouvelle question", use_container_width=True, type="primary"):
        if st.session_state["session_id"] is None:
            st.session_state["session_id"] = create_session(topic, chapter or "")

        history_text = "\n".join(
            f"- {q}" for q in st.session_state["questions_asked"][-5:]
        )

        with st.spinner("Generation..."):
            question = engine.generate_question(topic, chapter, history_text)

        st.session_state["current_question"] = question
        st.session_state["question_start_time"] = time.time()
        st.session_state["questions_asked"].append(question)

    # Display question
    if st.session_state["current_question"]:
        st.markdown(
            f'<div class="question-block">{st.session_state["current_question"]}</div>',
            unsafe_allow_html=True,
        )

        answer = st.text_area("Reponse", height=120, label_visibility="collapsed",
                              placeholder="Ecrivez votre reponse ici...")

        if st.button("Soumettre", use_container_width=True):
            if not answer.strip():
                st.warning("Entrez une reponse.")
            else:
                elapsed = time.time() - st.session_state["question_start_time"]

                with st.spinner("Evaluation..."):
                    evaluation = engine.evaluate_answer(
                        st.session_state["current_question"], answer, topic
                    )

                score = evaluation["score"]
                feedback = evaluation["feedback"]
                correction = evaluation["correction"]

                save_answer(
                    st.session_state["session_id"],
                    st.session_state["current_question"],
                    answer, score, elapsed, feedback,
                )
                update_mastery(topic, score, elapsed)

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

    # Reset
    if st.session_state["session_id"] is not None:
        st.markdown("")
        if st.button("Reinitialiser la session", type="secondary"):
            st.session_state["session_id"] = None
            st.session_state["current_question"] = None
            st.session_state["session_history"] = []
            st.session_state["questions_asked"] = []
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
            ["session_topic", "question", "score", "response_time_s", "created_at"]
        ]
        df_hist.columns = ["Theme", "Question", "Score", "Temps", "Date"]
        df_hist["Score"] = (df_hist["Score"] * 100).astype(int).astype(str) + "%"
        df_hist["Temps"] = df_hist["Temps"].round(1).astype(str) + "s"
        df_hist["Question"] = df_hist["Question"].str[:70] + "..."
        st.dataframe(df_hist, use_container_width=True, hide_index=True, height=300)
