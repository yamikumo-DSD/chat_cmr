import session_states as session
import sqlite3 # sqlite3 is built-in module (not in requirements.txt)
from contextlib import closing
from global_settings import SESSION_LOG_FILE 

db_path = SESSION_LOG_FILE

def create_log_db() -> None:
    import os
    if os.path.isfile(db_path):
        raise FileExistsError(f"{db_path} already exists.")
    
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE logs
(time_stamp text, session_id text, title text, states BLOB)""")
        conn.commit()

def register_log_db(states: session.States) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        history = states.context.history()
        title = history[0]["content"][:20] if len(history) > 0 else ""
        title = title.replace("'", "''") # To escape single quote "`", use "``" instead.
        c.execute(
            "INSERT INTO logs VALUES (?, ?, ?, ?)",
            (states.time_stamp, states.session_id, title, sqlite3.Binary(states.save_bytes()), )
        )
        conn.commit()
        
def remove_log_db(session_id: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM logs WHERE session_id = ?", (session_id,))
        conn.commit()

def load_log_db(return_states_binaries: bool = False) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        db = []
        
        if return_states_binaries: command = "SELECT time_stamp, session_id, title, states FROM logs"
        else: command = "SELECT time_stamp, session_id, title FROM logs"
            
        for item in c.execute(command):
            db.append(item)
            
        return db
        
def find_log_db(session_id: str) -> tuple|None:
    """
    Returns:
        tuple: Corresponding tuple (time_stamp, session_id, title, states_binaries)
    """
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM logs WHERE session_id = ?", (session_id,))
        return c.fetchone()

def update_log_db(states: session.States) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE logs SET states = ? WHERE session_id = ?",
            (
                sqlite3.Binary(states.save_bytes()), 
                states.session_id,
            )
        )
        conn.commit()