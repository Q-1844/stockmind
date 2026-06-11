"""
StockMind 交互式测试脚本
直接复制粘贴到终端运行

使用方法：
1. 确保 API key 已设置（见下方）
2. 复制整个文件保存为 test_interactive.py
3. python3 test_interactive.py
"""
import os
import sys
import json

# ═══════════════════════════════════════════════════════════════
# 第一步：配置 API
# ═══════════════════════════════════════════════════════════════
os.environ['LLM_PROVIDER'] = 'minimax'
os.environ['MINIMAX_API_KEY'] = 'tp-cgt4i1mb3elsi1zsa7olev1zia9xqbdkha0d4c85pvtqqmwf'
os.environ['MINIMAX_BASE_URL'] = 'https://token-plan-cn.xiaomimimo.com/v1'
os.environ['MINIMAX_MODEL'] = 'mimo-v2.5-pro'

sys.path.insert(0, '.')

print("=" * 70)
print("🧠 StockMind · 股票心智 - 交互式测试")
print("=" * 70)
print()
print("可用测试命令：")
print("  1  → 测试 LLM 连接")
print("  2  → 测试股票数据获取")
print("  3  → 测试数据库")
print("  4  → 测试单次分析")
print("  5  → 测试反思机制")
print("  6  → 测试技能库")
print("  7  → 批量分析（多只股票）")
print("  8  → 查看 Agent 表现报告")
print("  9  → 启动 Web 仪表盘")
print("  all → 跑全部测试")
print("  q  → 退出")
print()

# ═══════════════════════════════════════════════════════════════
# 测试函数
# ═══════════════════════════════════════════════════════════════

def test_llm():
    """测试 1：LLM 连接"""
    print("\n" + "─" * 60)
    print("🧪 测试 1: Mimo LLM 连接")
    print("─" * 60)
    from core.llm_client import LLMClient
    client = LLMClient('minimax')
    try:
        response = client.chat([
            {"role": "system", "content": "你是一个股票分析师，回答简洁。"},
            {"role": "user", "content": "用一句话评价 AAPL 当前是否值得买入？"}
        ])
        print(f"✅ LLM 正常")
        print(f"\n回复:\n{response}\n")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_data_fetcher(symbol="AAPL"):
    """测试 2：股票数据获取"""
    print("\n" + "─" * 60)
    print(f"🧪 测试 2: 获取 {symbol} 数据")
    print("─" * 60)
    from core.data_fetcher import StockDataFetcher
    fetcher = StockDataFetcher()
    try:
        info = fetcher.fetch_stock_info(symbol)
        print(f"✅ 获取成功")
        print(f"   名称: {info.get('name', 'N/A')}")
        print(f"   行业: {info.get('sector', 'N/A')}")
        print(f"   当前价: ${info.get('current_price', 0):.2f}")
        print(f"   PE: {info.get('pe_ratio', 'N/A')}")
        print(f"   市值: {info.get('market_cap', 0):,}")
        print(f"   52周高: ${info.get('52_week_high', 0):.2f}")
        print(f"   52周低: ${info.get('52_week_low', 0):.2f}")

        # 技术指标
        tech = fetcher.fetch_technical_indicators(symbol)
        if 'error' not in tech:
            print(f"\n   📊 技术指标:")
            print(f"      SMA20: ${tech.get('sma_20', 0):.2f}")
            print(f"      SMA50: ${tech.get('sma_50', 0):.2f}")
            print(f"      RSI:   {tech.get('rsi_14', 0):.1f}")
            print(f"      MACD:  {tech.get('macd', 0):.2f}")

        # 新闻
        news = fetcher.fetch_news(symbol, limit=3)
        if news:
            print(f"\n   📰 最近新闻:")
            for n in news:
                print(f"      • {n.get('title', '')[:60]}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_database():
    """测试 3：数据库"""
    print("\n" + "─" * 60)
    print("🧪 测试 3: 数据库 + FTS5 全文检索")
    print("─" * 60)
    from core.database import get_connection, DatabaseManager
    try:
        with get_connection() as conn:
            # 创建测试关注
            conn.execute(
                'INSERT OR IGNORE INTO watchlist (symbol, name) VALUES (?, ?)',
                ('AAPL', 'Apple Inc.')
            )
            conn.execute(
                'INSERT OR IGNORE INTO watchlist (symbol, name) VALUES (?, ?)',
                ('TSLA', 'Tesla Inc.')
            )
            conn.commit()

            # 插入测试技能
            conn.execute(
                'INSERT INTO skills (name, category, content, tags_json) '
                'VALUES (?, ?, ?, ?)',
                ('低PE估值策略', 'fundamental', '当PE低于行业平均30%时考虑买入。这是价值投资的核心策略。', '["PE","估值","价值投资"]')
            )
            conn.execute(
                'INSERT INTO skills (name, category, content, tags_json) '
                'VALUES (?, ?, ?, ?)',
                ('MACD金叉策略', 'technical', 'MACD从下穿向上穿时形成金叉，是买入信号。', '["MACD","金叉","技术面"]')
            )
            conn.commit()

            # 查询关注列表
            cur = conn.execute('SELECT symbol, name FROM watchlist')
            print(f"✅ 关注列表: {cur.fetchall()}")

            # FTS5 全文搜索
            cur = conn.execute(
                "SELECT name, category FROM skills_fts WHERE skills_fts MATCH ?",
                ('估值',)
            )
            results = cur.fetchall()
            print(f"✅ FTS5 搜索 '估值': 找到 {len(results)} 条")
            for r in results:
                print(f"   • {r[0]} ({r[1]})")

            # 测试 DatabaseManager
            db = DatabaseManager()
            wl = db.get_watchlist()
            print(f"✅ DatabaseManager.get_watchlist(): {len(wl)} 只股票")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze(symbol="AAPL"):
    """测试 4：单次分析"""
    print("\n" + "─" * 60)
    print(f"🧪 测试 4: 完整分析 {symbol}")
    print("─" * 60)
    from main import StockMindAgent
    agent = StockMindAgent()
    try:
        result = agent.analyze(symbol)
        print(f"✅ 分析完成")
        print(f"\n   股票: {result.get('symbol', symbol)}")
        print(f"   价格: ${result.get('price', 0):.2f}")
        print(f"   建议: {result.get('action', 'N/A').upper()}")
        print(f"   信心: {result.get('confidence', 0):.1%}")
        print(f"   风险: {result.get('risk_level', 'N/A')}")
        print(f"   理由: {result.get('reason', 'N/A')}")

        # 五维度分析
        dims = result.get('dimensions', {})
        if dims:
            print(f"\n   📊 五维度评分:")
            for name, data in dims.items():
                if isinstance(data, dict):
                    score = data.get('score', 0)
                    emoji = "🟢" if score > 0.3 else ("🔴" if score < -0.3 else "🟡")
                    print(f"      {emoji} {name:12s}: {score:+.2f}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reflect():
    """测试 5：反思机制"""
    print("\n" + "─" * 60)
    print("🧪 测试 5: 反思历史决策")
    print("─" * 60)
    from main import StockMindAgent
    agent = StockMindAgent()
    try:
        results = agent.reflect(days_ago=1)
        print(f"✅ 反思完成，处理 {len(results)} 条决策")
        for r in results[:5]:
            sym = r.get('symbol', 'N/A')
            correct = r.get('was_correct', None)
            pct = r.get('profit_pct', 0)
            status = "✓" if correct == 1 else ("✗" if correct == 0 else "?")
            print(f"   {status} {sym}: 收益 {pct:+.2%}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skills():
    """测试 6：技能库"""
    print("\n" + "─" * 60)
    print("🧪 测试 6: 技能库")
    print("─" * 60)
    from main import StockMindAgent
    agent = StockMindAgent()
    try:
        print("\n   📚 现有技能:")
        agent.skills_list()

        print("\n   🔍 搜索 'PE':")
        result = agent.skills_search("PE")
        if result:
            for s in result:
                print(f"      • {s.get('name', 'N/A')}: {s.get('content', '')[:60]}...")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_batch(symbols=None):
    """测试 7：批量分析"""
    if symbols is None:
        symbols = ["AAPL", "TSLA", "NVDA"]
    print("\n" + "─" * 60)
    print(f"🧪 测试 7: 批量分析 {symbols}")
    print("─" * 60)
    from main import StockMindAgent
    agent = StockMindAgent()
    for sym in symbols:
        print(f"\n   📊 {sym}:")
        try:
            result = agent.analyze(sym)
            print(f"      → {result.get('action', 'N/A').upper()} "
                  f"(信心 {result.get('confidence', 0):.0%})")
        except Exception as e:
            print(f"      ❌ 失败: {e}")


def test_report():
    """测试 8：表现报告"""
    print("\n" + "─" * 60)
    print("🧪 测试 8: Agent 表现报告")
    print("─" * 60)
    from main import StockMindAgent
    agent = StockMindAgent()
    agent.report()


def test_web():
    """测试 9：启动 Web"""
    print("\n" + "─" * 60)
    print("🧪 测试 9: Web 仪表盘")
    print("─" * 60)
    print("  启动后访问: http://localhost:8080")
    print("  按 Ctrl+C 停止")
    from main import StockMindAgent
    agent = StockMindAgent()
    try:
        agent.web()
    except KeyboardInterrupt:
        print("\n  已停止")


# ═══════════════════════════════════════════════════════════════
# 交互循环
# ═══════════════════════════════════════════════════════════════

tests = {
    '1': test_llm,
    '2': test_data_fetcher,
    '3': test_database,
    '4': test_analyze,
    '5': test_reflect,
    '6': test_skills,
    '7': test_batch,
    '8': test_report,
    '9': test_web,
}

while True:
    try:
        cmd = input("\nStockMind> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n退出")
        break

    if cmd in ('q', 'quit', 'exit'):
        print("👋 再见！")
        break
    elif cmd == 'all':
        for k in ['1', '2', '3', '4', '5', '6', '8']:  # 跳过 7 和 9
            tests[k]()
    elif cmd in tests:
        # 处理带参数的命令
        if cmd == '2':
            sym = input("  输入股票代码 (默认 AAPL): ").strip().upper() or "AAPL"
            test_data_fetcher(sym)
        elif cmd == '4':
            sym = input("  输入股票代码 (默认 AAPL): ").strip().upper() or "AAPL"
            test_analyze(sym)
        elif cmd == '7':
            syms = input("  输入股票代码，逗号分隔 (默认 AAPL,TSLA,NVDA): ").strip()
            sym_list = [s.strip().upper() for s in syms.split(",")] if syms else None
            test_batch(sym_list)
        else:
            tests[cmd]()
    elif cmd == 'help' or cmd == 'h':
        print("\n命令: 1-9, all, q")
        print("2/4: 会询问股票代码")
        print("7: 会询问多个股票代码")
    else:
        print(f"  未知命令: {cmd}，输入 help 查看")
