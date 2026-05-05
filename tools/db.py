import sqlite3
import hashlib
import os
import secrets
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "promptee.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_prompt TEXT NOT NULL,
            optimized_prompt TEXT NOT NULL,
            raw_score TEXT NOT NULL,
            optimized_score TEXT NOT NULL,
            improvement_score REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def create_user(username: str, password: str):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        salt = secrets.token_hex(16)
        pwd_hash = hash_password(password, salt)
        c.execute('INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)', (username, pwd_hash, salt))
        conn.commit()
        return True, "User created successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def verify_user(username: str, password: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, password_hash, salt FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user:
        pwd_hash = hash_password(password, user['salt'])
        if pwd_hash == user['password_hash']:
            return True, user['id']
    return False, None

def save_optimization_history(user_id: int, raw_prompt: str, optimized_prompt: str, raw_score: dict, optimized_score: dict, improvement_score: float):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO optimization_history (user_id, raw_prompt, optimized_prompt, raw_score, optimized_score, improvement_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, raw_prompt, optimized_prompt, json.dumps(raw_score), json.dumps(optimized_score), improvement_score))
    conn.commit()
    conn.close()

def get_user_history(user_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT timestamp, raw_prompt, optimized_prompt, raw_score, optimized_score, improvement_score
        FROM optimization_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'timestamp': row['timestamp'],
            'raw_prompt': row['raw_prompt'],
            'optimized_prompt': row['optimized_prompt'],
            'raw_score': json.loads(row['raw_score']),
            'optimized_score': json.loads(row['optimized_score']),
            'improvement_score': row['improvement_score']
        })
    return history
