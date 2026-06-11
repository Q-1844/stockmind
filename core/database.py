"""
StockMind · 股票心智 — 数据库层
基于 SQLite + FTS5 的持久化存储，支持全文检索
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from config import DB_PATH, SKILLS_DIR


# ═══════════════════════════════════════════════════════════════
# 数据库管理器（兼容性别名，实际操作通过模块级函数调用）
# ═══════════════════════════════════════════════════════════════

class DatabaseManager:
    """数据库管理器占位类，保持与其他模块的接口兼容"""
    pass


# ═══════════════════════════════════════════════════════════════
# 数据库连接管理
# ═══════════════════════════════════════════════════════════════

def _ensure_data_dir() -> None:
    """确保 data 目录存在"""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


@contextmanager
def get_connection():
    """获取数据库连接的上下文管理器"""
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════
# 数据库初始化
# ═══════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
-- 决策记录表
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('buy', 'sell', 'hold')),
    reason TEXT NOT NULL DEFAULT '',
    price_at_decision REAL,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0 AND 1),
    dimensions_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 反思记录表
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    check_date TEXT NOT NULL,
    price_after REAL,
    was_correct INTEGER CHECK(was_correct IN (0, 1) OR was_correct IS NULL),
    profit_pct REAL,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (decision_id) REFERENCES decisions(id) ON DELETE CASCADE
);

-- 技能库表
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    usage_count INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 关注列表
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    notes TEXT NOT NULL DEFAULT ''
);

-- 日线快照表
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    UNIQUE(symbol, date)
);

-- 技能全文检索虚拟表（FTS5）
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    name,
    category,
    content,
    tags_json,
    content='skills',
    content_rowid='id'
);

-- 触发器：保持 FTS 索引同步
CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, name, category, content, tags_json)
    VALUES (new.id, new.name, new.category, new.content, new.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, category, content, tags_json)
    VALUES ('delete', old.id, old.name, old.category, old.content, old.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, category, content, tags_json)
    VALUES ('delete', old.id, old.name, old.category, old.content, old.tags_json);
    INSERT INTO skills_fts(rowid, name, category, content, tags_json)
    VALUES (new.id, new.name, new.category, new.content, new.tags_json);
END;

-- 常用索引
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);
CREATE INDEX IF NOT EXISTS idx_reflections_decision_id ON reflections(decision_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date ON daily_snapshots(symbol, date);
"""


def init_db() -> None:
    """初始化数据库，创建所有表和索引"""
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)


# ═══════════════════════════════════════════════════════════════
# Decisions 决策操作
# ═══════════════════════════════════════════════════════════════

def add_decision(
    symbol: str,
    date: str,
    action: str,
    reason: str,
    price_at_decision: Optional[float] = None,
    confidence: float = 0.5,
    dimensions: Optional[dict] = None,
) -> int:
    """添加一条决策记录，返回记录 ID"""
    if action not in ("buy", "sell", "hold"):
        raise ValueError(f"action 必须是 buy/sell/hold，收到: {action}")
    dimensions_json = json.dumps(dimensions or {}, ensure_ascii=False)
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO decisions (symbol, date, action, reason, price_at_decision, confidence, dimensions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, date, action, reason, price_at_decision, confidence, dimensions_json),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_decisions(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """获取决策列表（按时间倒序）"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def get_decisions_by_symbol(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """获取某只股票的决策历史"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_decisions(days: int = 7) -> list[dict[str, Any]]:
    """获取最近 N 天的决策"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE date >= date('now', ?) ORDER BY date DESC",
            (f"-{days} days",),
        ).fetchall()
        return [dict(r) for r in rows]


def get_decision_stats() -> dict[str, Any]:
    """获取决策统计信息"""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        by_action = conn.execute(
            "SELECT action, COUNT(*) as cnt FROM decisions GROUP BY action"
        ).fetchall()
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM decisions"
        ).fetchone()[0]
        return {
            "total": total,
            "by_action": {r["action"]: r["cnt"] for r in by_action},
            "avg_confidence": round(avg_confidence or 0, 3),
        }


# ═══════════════════════════════════════════════════════════════
# Reflections 反思操作
# ═══════════════════════════════════════════════════════════════

def add_reflection(
    decision_id: int,
    check_date: str,
    price_after: Optional[float] = None,
    was_correct: Optional[int] = None,
    profit_pct: Optional[float] = None,
    notes: str = "",
) -> int:
    """添加一条反思记录"""
    if was_correct is not None and was_correct not in (0, 1):
        raise ValueError(f"was_correct 必须是 0/1/None，收到: {was_correct}")
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO reflections (decision_id, check_date, price_after, was_correct, profit_pct, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (decision_id, check_date, price_after, was_correct, profit_pct, notes),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_reflections_by_decision(decision_id: int) -> list[dict[str, Any]]:
    """获取某条决策的所有反思"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reflections WHERE decision_id = ? ORDER BY check_date",
            (decision_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_accuracy_stats(symbol: Optional[str] = None) -> dict[str, Any]:
    """获取准确率统计，可按股票筛选"""
    with get_connection() as conn:
        if symbol:
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                          AVG(profit_pct) as avg_profit
                   FROM reflections r
                   JOIN decisions d ON r.decision_id = d.id
                   WHERE d.symbol = ? AND r.was_correct IS NOT NULL""",
                (symbol,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                          AVG(profit_pct) as avg_profit
                   FROM reflections WHERE was_correct IS NOT NULL"""
            ).fetchone()
        total = row["total"]
        correct = row["correct"] or 0
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 3) if total > 0 else 0.0,
            "avg_profit_pct": round(row["avg_profit"] or 0, 4),
        }


def get_consecutive_errors(symbol: Optional[str] = None) -> int:
    """获取连续错误次数（从最近一次反思倒推）"""
    with get_connection() as conn:
        if symbol:
            rows = conn.execute(
                """SELECT r.was_correct FROM reflections r
                   JOIN decisions d ON r.decision_id = d.id
                   WHERE d.symbol = ? AND r.was_correct IS NOT NULL
                   ORDER BY r.check_date DESC""",
                (symbol,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT was_correct FROM reflections
                   WHERE was_correct IS NOT NULL
                   ORDER BY check_date DESC"""
            ).fetchall()
        count = 0
        for r in rows:
            if r["was_correct"] == 0:
                count += 1
            else:
                break
        return count


# ═══════════════════════════════════════════════════════════════
# Skills 技能操作
# ═══════════════════════════════════════════════════════════════

def add_skill(
    name: str,
    category: str = "",
    content: str = "",
    tags: Optional[list[str]] = None,
) -> int:
    """添加一条技能，返回 ID"""
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO skills (name, category, content, tags_json)
               VALUES (?, ?, ?, ?)""",
            (name, category, content, tags_json),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def search_skills(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """使用 FTS5 全文检索技能"""
    with get_connection() as conn:
        # 安全转义 FTS5 查询中的特殊字符
        safe_query = query.replace('"', '""')
        rows = conn.execute(
            """SELECT s.* FROM skills s
               JOIN skills_fts f ON s.id = f.rowid
               WHERE skills_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (f'"{safe_query}"', limit),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.pop("tags_json", "[]"))
            return_dimensions = d.pop("dimensions_json", None)  # skills 无此字段，防御性处理
            results.append(d)
        return results


def update_skill_usage(skill_id: int) -> None:
    """增加技能使用计数"""
    with get_connection() as conn:
        conn.execute(
            "UPDATE skills SET usage_count = usage_count + 1, updated_at = datetime('now') WHERE id = ?",
            (skill_id,),
        )


def update_skill_success_rate(skill_id: int, success: bool) -> None:
    """更新技能成功率（增量计算）"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT usage_count, success_rate FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            return
        count = row["usage_count"]
        old_rate = row["success_rate"]
        # 增量更新: new_rate = old_rate + (1 - old_rate) / count  (成功)
        #           new_rate = old_rate - old_rate / count         (失败)
        if success:
            new_rate = old_rate + (1.0 - old_rate) / max(count, 1)
        else:
            new_rate = old_rate - old_rate / max(count, 1)
        new_rate = max(0.0, min(1.0, new_rate))
        conn.execute(
            "UPDATE skills SET success_rate = ?, updated_at = datetime('now') WHERE id = ?",
            (round(new_rate, 4), skill_id),
        )


def get_all_skills(category: Optional[str] = None) -> list[dict[str, Any]]:
    """获取所有技能，可按分类筛选"""
    with get_connection() as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM skills WHERE category = ? ORDER BY usage_count DESC",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM skills ORDER BY usage_count DESC"
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.pop("tags_json", "[]"))
            results.append(d)
        return results


def delete_skill(skill_id: int) -> bool:
    """删除技能，返回是否成功"""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        return cursor.rowcount > 0


# ═══════════════════════════════════════════════════════════════
# Watchlist 关注列表操作
# ═══════════════════════════════════════════════════════════════

def add_to_watchlist(symbol: str, name: str = "", notes: str = "") -> int:
    """添加股票到关注列表"""
    with get_connection() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO watchlist (symbol, name, notes) VALUES (?, ?, ?)",
                (symbol.upper(), name, notes),
            )
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError:
            # 已存在则忽略
            return -1


def remove_from_watchlist(symbol: str) -> bool:
    """从关注列表移除"""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),)
        )
        return cursor.rowcount > 0


def get_watchlist() -> list[dict[str, Any]]:
    """获取完整关注列表"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
# DailySnapshots 日线快照操作
# ═══════════════════════════════════════════════════════════════

def add_snapshot(
    symbol: str,
    date: str,
    open: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: Optional[float] = None,
    volume: Optional[float] = None,
) -> int:
    """添加日线快照（INSERT OR REPLACE）"""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO daily_snapshots (symbol, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, date, open, high, low, close, volume),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def get_snapshots(symbol: str, limit: int = 30) -> list[dict[str, Any]]:
    """获取某只股票的日线快照"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_snapshots WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_snapshot(symbol: str) -> Optional[dict[str, Any]]:
    """获取某只股票的最新日线快照"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM daily_snapshots WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None


# ═══════════════════════════════════════════════════════════════
# 辅助方法
# ═══════════════════════════════════════════════════════════════

def get_agent_performance_summary() -> dict[str, Any]:
    """获取 Agent 整体表现摘要：总准确率、按股票准确率、近期趋势"""
    with get_connection() as conn:
        # 总体准确率
        overall = get_accuracy_stats()

        # 按股票准确率
        symbols = conn.execute(
            """SELECT DISTINCT d.symbol FROM decisions d
               JOIN reflections r ON r.decision_id = d.id
               WHERE r.was_correct IS NOT NULL"""
        ).fetchall()
        by_symbol = {}
        for s in symbols:
            by_symbol[s["symbol"]] = get_accuracy_stats(symbol=s["symbol"])

        # 近期趋势（最近 7 条反思的正确率）
        recent_rows = conn.execute(
            """SELECT was_correct FROM reflections
               WHERE was_correct IS NOT NULL
               ORDER BY check_date DESC LIMIT 7"""
        ).fetchall()
        if recent_rows:
            recent_correct = sum(1 for r in recent_rows if r["was_correct"] == 1)
            recent_trend = round(recent_correct / len(recent_rows), 3)
        else:
            recent_trend = 0.0

        return {
            "overall": overall,
            "by_symbol": by_symbol,
            "recent_trend": recent_trend,
            "consecutive_errors": get_consecutive_errors(),
        }


def get_decision_chain(symbol: str) -> list[dict[str, Any]]:
    """获取某只股票的完整决策链（决策 + 对应反思）"""
    with get_connection() as conn:
        decisions = conn.execute(
            "SELECT * FROM decisions WHERE symbol = ? ORDER BY date",
            (symbol,),
        ).fetchall()
        chain = []
        for d in decisions:
            reflections = conn.execute(
                "SELECT * FROM reflections WHERE decision_id = ? ORDER BY check_date",
                (d["id"],),
            ).fetchall()
            chain.append({
                "decision": dict(d),
                "reflections": [dict(r) for r in reflections],
            })
        return chain


def reset_all() -> None:
    """清空所有数据（仅用于测试）"""
    with get_connection() as conn:
        for table in ("reflections", "decisions", "skills", "skills_fts",
                       "watchlist", "daily_snapshots"):
            conn.execute(f"DELETE FROM {table}")
        # 重置自增 ID
        for table in ("reflections", "decisions", "skills",
                       "watchlist", "daily_snapshots"):
            conn.execute(f"DELETE FROM sqlite_sequence WHERE name = '{table}'")


# ═══════════════════════════════════════════════════════════════
# 模块加载时自动初始化
# ═══════════════════════════════════════════════════════════════
init_db()
