import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radar_data.db")


def _conexao():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conexao() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo TEXT NOT NULL,
                data_hora TEXT NOT NULL,
                dados TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                perfil TEXT NOT NULL,
                data_hora TEXT NOT NULL,
                score_medio REAL,
                dy_ponderado REAL,
                total_ativos INTEGER,
                ativos TEXT NOT NULL,
                alocacao TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_tipo ON snapshots(tipo);
            CREATE INDEX IF NOT EXISTS idx_snapshots_data ON snapshots(data_hora);
            CREATE INDEX IF NOT EXISTS idx_portfolio_perfil ON portfolio_history(perfil);
            CREATE INDEX IF NOT EXISTS idx_portfolio_data ON portfolio_history(data_hora);
        """)


def salvar_snapshot(tipo, dados):
    init_db()
    with _conexao() as conn:
        conn.execute(
            "INSERT INTO snapshots (tipo, data_hora, dados) VALUES (?, ?, ?)",
            (tipo, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(dados, ensure_ascii=False)),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def listar_snapshots(tipo=None, limite=50):
    init_db()
    with _conexao() as conn:
        if tipo:
            rows = conn.execute(
                "SELECT id, tipo, data_hora FROM snapshots WHERE tipo = ? ORDER BY data_hora DESC LIMIT ?",
                (tipo, limite),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, tipo, data_hora FROM snapshots ORDER BY data_hora DESC LIMIT ?",
                (limite,),
            ).fetchall()
        return [{"id": r[0], "tipo": r[1], "data_hora": r[2]} for r in rows]


def carregar_snapshot(snapshot_id):
    init_db()
    with _conexao() as conn:
        row = conn.execute(
            "SELECT dados FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
    return None


def salvar_portfolio(perfil, score_medio, dy_ponderado, total_ativos, ativos, alocacao):
    init_db()
    with _conexao() as conn:
        conn.execute(
            """INSERT INTO portfolio_history
               (perfil, data_hora, score_medio, dy_ponderado, total_ativos, ativos, alocacao)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (perfil, time.strftime("%Y-%m-%d %H:%M:%S"), score_medio, dy_ponderado,
             total_ativos, json.dumps(ativos, ensure_ascii=False),
             json.dumps(alocacao, ensure_ascii=False)),
        )


def historico_portfolio(perfil, limite=30):
    init_db()
    with _conexao() as conn:
        rows = conn.execute(
            """SELECT id, data_hora, score_medio, dy_ponderado, total_ativos, ativos, alocacao
               FROM portfolio_history WHERE perfil = ?
               ORDER BY data_hora DESC LIMIT ?""",
            (perfil, limite),
        ).fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "data_hora": r[1],
                "score_medio": r[2],
                "dy_ponderado": r[3],
                "total_ativos": r[4],
                "ativos": json.loads(r[5]),
                "alocacao": json.loads(r[6]),
            })
        return results
