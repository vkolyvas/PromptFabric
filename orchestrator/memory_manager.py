import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.settings import settings


class MemoryManager:
    """SQLite-based conversation memory"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session"""
        if not session_id:
            session_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now)
        )

        conn.commit()
        conn.close()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )

        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )

        conn.commit()
        conn.close()

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages from session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT role, content, created_at FROM messages
               WHERE session_id = ? ORDER BY created_at DESC LIMIT ?""",
            (session_id, limit)
        )

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "created_at": row[2]
            })

        conn.close()
        return list(reversed(messages))

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get formatted message history for LLM"""
        messages = self.get_messages(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def delete_session(self, session_id: str):
        """Delete a session and its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        conn.commit()
        conn.close()


memory_manager = MemoryManager()
