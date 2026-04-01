"""Gestion SQLite : scores, temps de réponse, progression."""

import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "quant_coach.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crée les tables si elles n'existent pas."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            topic TEXT,
            chapter TEXT
        );

        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            user_answer TEXT NOT NULL,
            score REAL NOT NULL,
            response_time_s REAL NOT NULL,
            feedback TEXT,
            source_ref TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS mastery (
            topic TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'Non abordé',
            best_score REAL DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            avg_time_s REAL DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()


def create_session(topic: str, chapter: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (started_at, topic, chapter) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), topic, chapter),
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def save_answer(
    session_id: int,
    question: str,
    user_answer: str,
    score: float,
    response_time_s: float,
    feedback: str = "",
    source_ref: str = "",
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO answers
           (session_id, question, user_answer, score, response_time_s, feedback, source_ref, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, question, user_answer, score, response_time_s, feedback, source_ref, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def update_mastery(topic: str, score: float, time_s: float) -> None:
    conn = get_connection()
    cur = conn.cursor()

    row = cur.execute("SELECT * FROM mastery WHERE topic = ?", (topic,)).fetchone()

    if row is None:
        status = _score_to_status(score)
        cur.execute(
            "INSERT INTO mastery (topic, status, best_score, attempts, avg_time_s) VALUES (?, ?, ?, 1, ?)",
            (topic, status, score, time_s),
        )
    else:
        attempts = row["attempts"] + 1
        best = max(row["best_score"], score)
        avg_t = (row["avg_time_s"] * row["attempts"] + time_s) / attempts
        status = _score_to_status(best)
        cur.execute(
            "UPDATE mastery SET status=?, best_score=?, attempts=?, avg_time_s=? WHERE topic=?",
            (status, best, attempts, avg_t, topic),
        )
    conn.commit()
    conn.close()


def _score_to_status(score: float) -> str:
    if score >= 0.70:
        return "Maîtrisé"
    elif score > 0:
        return "En cours"
    return "Non abordé"


def get_all_mastery() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM mastery ORDER BY topic").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_answer_history(limit: int = 200) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT a.*, s.topic as session_topic, s.chapter
           FROM answers a JOIN sessions s ON a.session_id = s.id
           ORDER BY a.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_response_times() -> list[dict]:
    """Retourne l'historique des temps de réponse pour le graphique d'évolution."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT created_at, response_time_s, score FROM answers ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()
