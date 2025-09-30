# db.py
import sqlite3, pandas as pd, streamlit as st
from passlib.hash import bcrypt

DB_PATH = "app.db"

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_db():
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolios(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS holdings(
                portfolio_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                PRIMARY KEY(portfolio_id, ticker),
                FOREIGN KEY(portfolio_id) REFERENCES portfolios(id)
            )
        """)
        con.commit()

# ---------- auth ----------
def signup(email, password):
    email = (email or "").strip().lower()
    if not email or not password: return False, "Email and password required."
    try:
        with _conn() as con:
            con.execute("INSERT INTO users(email, password_hash) VALUES (?,?)",
                        (email, bcrypt.hash(password)))
        st.session_state.user = {"email": email, "id": get_user_id(email)}
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Email already registered."

def login(email, password):
    email = (email or "").strip().lower()
    with _conn() as con:
        row = con.execute("SELECT id, password_hash FROM users WHERE email=?", (email,)).fetchone()
    if not row: return False, "Invalid credentials."
    uid, hashed = row
    if not bcrypt.verify(password, hashed): return False, "Invalid credentials."
    st.session_state.user = {"email": email, "id": uid}
    return True, "Signed in."

def logout():
    st.session_state.pop("user", None)

def get_user_id(email):
    with _conn() as con:
        r = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    return r[0] if r else None

def get_current_user_from_state():
    return st.session_state.get("user")

# ---------- portfolios ----------
def upsert_portfolio(user_id: int, name: str, holdings_df: pd.DataFrame):
    name = (name or "").strip()
    if not name: raise ValueError("Portfolio name required.")
    if holdings_df.empty: raise ValueError("Holdings are empty.")
    holdings_df = holdings_df.copy()
    holdings_df["ticker"] = holdings_df["ticker"].astype(str).str.upper()
    holdings_df["shares"] = pd.to_numeric(holdings_df["shares"], errors="coerce").fillna(0.0)

    with _conn() as con:
        cur = con.cursor()
        cur.execute("INSERT OR IGNORE INTO portfolios(user_id, name) VALUES (?,?)", (user_id, name))
        pid = cur.execute("SELECT id FROM portfolios WHERE user_id=? AND name=?", (user_id, name)).fetchone()[0]
        cur.execute("DELETE FROM holdings WHERE portfolio_id=?", (pid,))
        cur.executemany(
            "INSERT INTO holdings(portfolio_id, ticker, shares) VALUES (?,?,?)",
            [(pid, r["ticker"], float(r["shares"])) for _, r in holdings_df.iterrows()]
        )
        con.commit()

def list_portfolios(user_id: int):
    with _conn() as con:
        rows = con.execute("SELECT id, name FROM portfolios WHERE user_id=? ORDER BY name", (user_id,)).fetchall()
    return rows

def load_holdings(portfolio_id: int) -> pd.DataFrame:
    with _conn() as con:
        rows = con.execute("SELECT ticker, shares FROM holdings WHERE portfolio_id=?", (portfolio_id,)).fetchall()
    return pd.DataFrame(rows, columns=["ticker","shares"])

def delete_portfolio(user_id: int, name: str):
    with _conn() as con:
        cur = con.cursor()
        pid = cur.execute("SELECT id FROM portfolios WHERE user_id=? AND name=?", (user_id, name)).fetchone()
        if not pid: return
        pid = pid[0]
        cur.execute("DELETE FROM holdings WHERE portfolio_id=?", (pid,))
        cur.execute("DELETE FROM portfolios WHERE id=?", (pid,))
        con.commit()
