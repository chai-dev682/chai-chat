"""
SQLite persistence for Conversation Response sessions.

Stores sessions as JSON blobs so the full chat history (including image data)
survives page reloads. Images are stored as base64 strings within the JSON,
which keeps the schema simple at the cost of DB size.
"""

import json
import sqlite3
from os.path import join, dirname, abspath

DB_PATH = join(dirname(dirname(abspath(__file__))), "conv_sessions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            context_json TEXT NOT NULL,
            chat_history_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def load_all_sessions() -> dict:
    """Load all sessions from DB, returning {id: {label, context, chat_history}}."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, label, context_json, chat_history_json FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()

    sessions = {}
    for row in rows:
        sid, label, context_json, chat_history_json = row
        sessions[sid] = {
            "label": label,
            "context": json.loads(context_json),
            "chat_history": json.loads(chat_history_json),
        }
    return sessions


def save_session(sid: str, label: str, context: dict, chat_history: list):
    """Insert or update a single session."""
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO sessions (id, label, context_json, chat_history_json, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            label = excluded.label,
            context_json = excluded.context_json,
            chat_history_json = excluded.chat_history_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (sid, label, json.dumps(context), json.dumps(chat_history)),
    )
    conn.commit()
    conn.close()


def rename_session(sid: str, new_label: str):
    """Rename a session's label."""
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET label = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (new_label, sid),
    )
    conn.commit()
    conn.close()


def delete_session(sid: str):
    """Delete a session by id."""
    conn = _get_conn()
    conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    conn.commit()
    conn.close()
