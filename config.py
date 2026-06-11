"""
StockMind · 股票心智 — 越用越强的股票分析 Agent
配置文件
"""
import os

# ═══════════════════════════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════════════════════════
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax")  # minimax / openai / deepseek

LLM_CONFIG = {
    "minimax": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimax.chat/v1",
        "model": "MiniMax-Text-01",
        "max_tokens": 2048,
        "temperature": 0.3,
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "max_tokens": 2048,
        "temperature": 0.3,
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "max_tokens": 2048,
        "temperature": 0.3,
    },
}

# ═══════════════════════════════════════════════════════════════
# 数据源配置
# ═══════════════════════════════════════════════════════════════
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance")  # yfinance / alpha_vantage

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
# 数据库配置
# ═══════════════════════════════════════════════════════════════
DB_PATH = os.getenv("DB_PATH", "data/stockmind.db")

# ═══════════════════════════════════════════════════════════════
# 技能库配置
# ═══════════════════════════════════════════════════════════════
SKILLS_DIR = os.getenv("SKILLS_DIR", "skills")
SKILLS_INDEX_FILE = os.path.join(SKILLS_DIR, "_index.json")

# ═══════════════════════════════════════════════════════════════
# Agent 行为配置
# ═══════════════════════════════════════════════════════════════
# 反思间隔（天）
REFLECTION_INTERVAL_DAYS = 1

# 历史决策检索数量
HISTORY_LIMIT = 10

# 技能检索数量
SKILL_RETRIEVAL_LIMIT = 5

# 信心阈值：低于此值不给出建议
CONFIDENCE_THRESHOLD = 0.3

# 最大连续错误次数（超过后自动降级为"观望"）
MAX_CONSECUTIVE_ERRORS = 3

# ═══════════════════════════════════════════════════════════════
# 分析维度
# ═══════════════════════════════════════════════════════════════
ANALYSIS_DIMENSIONS = [
    "fundamental",    # 基本面分析
    "technical",      # 技术面分析
    "sentiment",      # 情绪面分析
    "news",           # 新闻面分析
    "macro",          # 宏观面分析
]

# ═══════════════════════════════════════════════════════════════
# 默认关注列表
# ═══════════════════════════════════════════════════════════════
DEFAULT_WATCHLIST = [
    "AAPL",   # 苹果
    "TSLA",   # 特斯拉
    "NVDA",   # 英伟达
    "MSFT",   # 微软
    "GOOGL",  # 谷歌
]

# ═══════════════════════════════════════════════════════════════
# Web 界面配置
# ═══════════════════════════════════════════════════════════════
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
