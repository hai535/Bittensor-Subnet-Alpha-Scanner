import sqlite3
import json
import time
import threading

DB_PATH = "/root/claude-chat/chat_history.db"

_local = threading.local()


def get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            created_at REAL,
            updated_at REAL,
            user TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    """)
    # Add user column if missing (migration)
    try:
        conn.execute("SELECT user FROM sessions LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE sessions ADD COLUMN user TEXT DEFAULT ''")
    # Assign existing sessions without user to 'shamless'
    conn.execute("UPDATE sessions SET user = 'shamless' WHERE user = '' OR user IS NULL")
    conn.commit()


def create_session(session_id, title="", user=""):
    conn = get_conn()
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at, user) VALUES (?, ?, ?, ?, ?)",
        (session_id, title, now, now, user),
    )
    conn.commit()


def update_session_title(session_id, title):
    conn = get_conn()
    conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, time.time(), session_id))
    conn.commit()


def add_message(session_id, role, content, user=""):
    conn = get_conn()
    now = time.time()
    # Ensure session exists
    create_session(session_id, user=user)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()

    # Auto-set title from first user message
    row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row and not row["title"] and role == "user":
        title = content[:50] + ("..." if len(content) > 50 else "")
        update_session_title(session_id, title)


def get_messages(session_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]


def list_sessions(user="", limit=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE user = ? ORDER BY updated_at DESC LIMIT ?",
        (user, limit),
    ).fetchall()
    return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"], "updated_at": r["updated_at"]} for r in rows]


def delete_session(session_id):
    conn = get_conn()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()


def clear_user_sessions(user):
    conn = get_conn()
    session_ids = conn.execute("SELECT id FROM sessions WHERE user = ?", (user,)).fetchall()
    for row in session_ids:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (row["id"],))
    conn.execute("DELETE FROM sessions WHERE user = ?", (user,))
    conn.commit()


def rename_user(old_user, new_user):
    conn = get_conn()
    conn.execute("UPDATE sessions SET user = ? WHERE user = ?", (new_user, old_user))
    conn.commit()
