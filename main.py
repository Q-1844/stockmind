"""
StockMind · 股票心智 — 越用越强的股票分析 Agent
主入口：CLI + 交互式 REPL
"""
import sys
import argparse
import logging
from datetime import datetime

from core.database import DatabaseManager, get_connection
from core.data_fetcher import StockDataFetcher, get_fetcher
from core.llm_client import LLMClient, get_client
from core.skill_system import SkillSystem
from core.analyzer import StockAnalyzer
from core.reflector import Reflector
from core import database as db

# ═══════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 终端颜色
# ═══════════════════════════════════════════════════════════════
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{Color.RESET}"

# ═══════════════════════════════════════════════════════════════
# StockMindAgent
# ═══════════════════════════════════════════════════════════════
class StockMindAgent:
    """股票心智 Agent：编排所有核心模块"""

    def __init__(self) -> None:
        self.db_manager = DatabaseManager
        self.llm_client = get_client()
        self.data_fetcher = get_fetcher()
        self.skill_system = SkillSystem(self.db_manager, self.llm_client)
        self.analyzer = StockAnalyzer(
            self.db_manager, self.llm_client, self.data_fetcher, self.skill_system
        )
        self.reflector = Reflector(
            self.db_manager, self.llm_client, self.data_fetcher, self.skill_system
        )
        logger.info("StockMind Agent 初始化完成")

    # ─── 分析 ──────────────────────────────────────────────────
    def analyze(self, symbol: str) -> dict:
        """执行完整分析 → 保存决策 → 触发自动学习"""
        symbol = symbol.upper().strip()
        print(f"\n{_c(Color.CYAN, '🔍 正在分析')} {_c(Color.BOLD, symbol)} ...")

        result = self.analyzer.analyze(symbol)

        # 保存决策到数据库
        decision_id = db.add_decision(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            action=result["action"],
            reason=result["reason"],
            price_at_decision=result.get("dimensions", {})
                .get("fundamental", {})
                .get("price"),
            confidence=result["confidence"],
            dimensions=result.get("dimensions", {}),
        )
        result["decision_id"] = decision_id

        # 触发自动学习
        self.skill_system.auto_learn(symbol, result)

        # 打印结果
        self._print_analysis(result)
        return result

    # ─── 反思 ──────────────────────────────────────────────────
    def reflect(self, days_ago: int = 1) -> list[dict]:
        """反思过去 N 天的决策"""
        print(f"\n{_c(Color.CYAN, '🔄 正在反思')} 最近 {days_ago} 天的决策 ...")
        results = self.reflector.reflect_all(days_ago=days_ago)

        if not results:
            print(_c(Color.YELLOW, "  ⚠️  无可反思的决策记录"))
            return results

        for r in results:
            if r.get("error"):
                print(f"  {_c(Color.RED, '✗')} #{r.get('decision_id')} 错误: {r['error']}")
                continue
            if r.get("skipped"):
                print(f"  {_c(Color.DIM, '⏭️')} #{r.get('decision_id')} {r.get('symbol')} — 已跳过")
                continue

            icon = _c(Color.GREEN, "✓") if r.get("was_correct") else _c(Color.RED, "✗")
            profit = r.get("profit_pct", 0)
            profit_str = _c(Color.GREEN, f"{profit:+.2f}%") if profit >= 0 else _c(Color.RED, f"{profit:+.2f}%")
            print(f"  {icon} #{r.get('decision_id')} {r.get('symbol')} "
                  f"action={r.get('action')} 收益={profit_str}")

        correct = sum(1 for r in results if r.get("was_correct"))
        total = len(results)
        print(f"\n  📊 反思总结: {correct}/{total} 正确 "
              f"({_c(Color.GREEN, f'{correct/total:.0%}') if total else '-'})")
        return results

    # ─── 关注列表 ──────────────────────────────────────────────
    def watchlist_add(self, symbol: str) -> None:
        symbol = symbol.upper().strip()
        rid = db.add_to_watchlist(symbol)
        if rid == -1:
            print(_c(Color.YELLOW, f"  ⚠️  {symbol} 已在关注列表中"))
        else:
            print(_c(Color.GREEN, f"  ✅ 已添加 {symbol} 到关注列表"))

    def watchlist_remove(self, symbol: str) -> None:
        symbol = symbol.upper().strip()
        removed = db.remove_from_watchlist(symbol)
        if removed:
            print(_c(Color.GREEN, f"  ✅ 已从关注列表移除 {symbol}"))
        else:
            print(_c(Color.YELLOW, f"  ⚠️  {symbol} 不在关注列表中"))

    def watchlist_list(self) -> None:
        items = db.get_watchlist()
        if not items:
            print(_c(Color.DIM, "  关注列表为空"))
            return
        print(f"\n  {_c(Color.BOLD, '📋 关注列表')}")
        for item in items:
            added = item.get("added_at", "")[:10]
            print(f"    • {item['symbol']:6s}  添加于 {added}")

    def watchlist_analyze(self) -> list[dict]:
        items = db.get_watchlist()
        if not items:
            print(_c(Color.YELLOW, "  ⚠️  关注列表为空，请先添加股票"))
            return []
        print(f"\n{_c(Color.CYAN, '📊 正在分析关注列表中的所有股票 ...')}")
        results = []
        for item in items:
            result = self.analyze(item["symbol"])
            results.append(result)
        return results

    # ─── 报告 ──────────────────────────────────────────────────
    def report(self) -> dict:
        print(f"\n{_c(Color.CYAN, '📊 正在生成性能报告 ...')}")
        report_data = self.reflector.get_performance_report()
        self._print_report(report_data)
        return report_data

    # ─── 建议 ──────────────────────────────────────────────────
    def suggest(self) -> list[str]:
        print(f"\n{_c(Color.CYAN, '💡 正在生成改进建议 ...')}")
        suggestions = self.reflector.suggest_improvements()
        if not suggestions:
            print(_c(Color.DIM, "  暂无建议"))
            return suggestions
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
        return suggestions

    # ─── 技能 ──────────────────────────────────────────────────
    def skills_list(self) -> None:
        all_skills = db.get_all_skills()
        if not all_skills:
            print(_c(Color.DIM, "  🧠 技能库为空，随着分析积累将自动学习"))
            return
        print(f"\n  {_c(Color.BOLD, '🧠 技能库')} ({len(all_skills)} 条)")
        for s in all_skills:
            rate = s.get("success_rate", 0)
            rate_str = _c(Color.GREEN, f"{rate:.0%}") if rate >= 0.5 else _c(Color.RED, f"{rate:.0%}")
            print(f"    • [{s.get('category', '-')}] {s['name']} "
                  f"使用{s.get('usage_count', 0)}次 成功率={rate_str}")

    def skills_search(self, query: str) -> None:
        results = db.search_skills(query)
        if not results:
            print(_c(Color.DIM, f"  未找到与 '{query}' 相关的技能"))
            return
        print(f"\n  {_c(Color.BOLD, '🧠 搜索结果')} ({len(results)} 条)")
        for s in results:
            print(f"    • [{s.get('category', '-')}] {s['name']}")

    # ─── 交互式 REPL ──────────────────────────────────────────
    def interactive(self) -> None:
        print(f"\n{_c(Color.BOLD, '🧠 StockMind · 股票心智')}")
        print(_c(Color.DIM, "  输入 help 查看命令，quit 退出\n"))

        while True:
            try:
                line = input(_c(Color.CYAN, "stockmind> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_c(Color.DIM, '再见！')}")
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                print(_c(Color.DIM, "再见！"))
                break

            elif cmd == "help":
                self._print_help()

            elif cmd == "analyze":
                if len(parts) < 2:
                    print(_c(Color.YELLOW, "  ⚠️  用法: analyze <symbol>"))
                    continue
                self.analyze(parts[1])

            elif cmd == "reflect":
                days = int(parts[1]) if len(parts) > 1 else 1
                self.reflect(days_ago=days)

            elif cmd == "watchlist":
                sub = parts[1].lower() if len(parts) > 1 else "list"
                if sub == "add" and len(parts) > 2:
                    self.watchlist_add(parts[2])
                elif sub == "remove" and len(parts) > 2:
                    self.watchlist_remove(parts[2])
                elif sub == "analyze":
                    self.watchlist_analyze()
                else:
                    self.watchlist_list()

            elif cmd == "report":
                self.report()

            elif cmd == "suggest":
                self.suggest()

            elif cmd == "skills":
                sub = parts[1].lower() if len(parts) > 1 else "list"
                if sub == "search" and len(parts) > 2:
                    self.skills_search(" ".join(parts[2:]))
                else:
                    self.skills_list()

            else:
                print(_c(Color.YELLOW, f"  ⚠️  未知命令: {cmd}，输入 help 查看帮助"))

    # ─── Web 仪表盘 ───────────────────────────────────────────
    def web(self) -> None:
        print(_c(Color.YELLOW, "  ⚠️  Web 仪表盘功能开发中，敬请期待"))

    # ═══════════════════════════════════════════════════════════
    # 内部方法：格式化输出
    # ═══════════════════════════════════════════════════════════

    def _print_analysis(self, result: dict) -> None:
        action = result.get("action", "hold")
        icon_map = {"buy": "📈", "sell": "📉", "hold": "⏸️"}
        action_label = {"buy": "买入", "sell": "卖出", "hold": "观望"}
        icon = icon_map.get(action, "⏸️")
        label = action_label.get(action, action)
        confidence = result.get("confidence", 0)

        # 信心度颜色
        if confidence >= 0.7:
            conf_str = _c(Color.GREEN, f"{confidence:.0%}")
        elif confidence >= 0.4:
            conf_str = _c(Color.YELLOW, f"{confidence:.0%}")
        else:
            conf_str = _c(Color.RED, f"{confidence:.0%}")

        risk = result.get("risk_level", "medium")
        risk_map = {"low": _c(Color.GREEN, "低"), "medium": _c(Color.YELLOW, "中"), "high": _c(Color.RED, "高")}
        risk_str = risk_map.get(risk, risk)

        print(f"\n  {icon} {_c(Color.BOLD, label)} {result.get('symbol', '')}  "
              f"信心度={conf_str}  风险={risk_str}")

        if result.get("price_target"):
            print(f"  🎯 目标价: ${result['price_target']:.2f}")

        print(f"  📝 {result.get('reason', '')}")

        # 维度得分
        dims = result.get("dimensions", {})
        if dims:
            dim_labels = {
                "fundamental": "基本面", "technical": "技术面",
                "sentiment": "情绪面", "news": "新闻面", "macro": "宏观面",
            }
            print("  ── 维度得分 ──")
            for key in ("fundamental", "technical", "sentiment", "news", "macro"):
                d = dims.get(key, {})
                score = d.get("score", 0)
                if score > 0.1:
                    score_str = _c(Color.GREEN, f"{score:+.2f}")
                elif score < -0.1:
                    score_str = _c(Color.RED, f"{score:+.2f}")
                else:
                    score_str = f"{score:+.2f}"
                print(f"    {dim_labels.get(key, key):4s}: {score_str}  {d.get('reason', '')}")

        print()

    def _print_report(self, report_data: dict) -> None:
        accuracy = report_data.get("overall_accuracy", 0)
        if accuracy >= 0.6:
            acc_str = _c(Color.GREEN, f"{accuracy:.1%}")
        elif accuracy >= 0.4:
            acc_str = _c(Color.YELLOW, f"{accuracy:.1%}")
        else:
            acc_str = _c(Color.RED, f"{accuracy:.1%}")

        trend = report_data.get("recent_trend", "stable")
        trend_map = {
            "improving": _c(Color.GREEN, "📈 上升"),
            "declining": _c(Color.RED, "📉 下降"),
            "stable": _c(Color.YELLOW, "➡️ 稳定"),
        }
        trend_str = trend_map.get(trend, trend)

        print(f"\n  {_c(Color.BOLD, '📊 StockMind 性能报告')}")
        print(f"  ─────────────────────")
        print(f"  总决策数:   {report_data.get('total_decisions', 0)}")
        print(f"  总反思数:   {report_data.get('total_reflections', 0)}")
        print(f"  整体准确率: {acc_str}")
        print(f"  近期趋势:   {trend_str}")
        print(f"  连续错误:   {report_data.get('consecutive_errors', 0)} 次")
        print(f"  已学技能:   {report_data.get('skills_learned', 0)} 条")
        print(f"  平均信心度: {report_data.get('avg_confidence', 0):.2f}")

        # 按行动类型
        by_action = report_data.get("by_action", {})
        if by_action:
            print("\n  ── 按行动类型 ──")
            action_labels = {"buy": "买入", "sell": "卖出", "hold": "观望"}
            for action, stats in by_action.items():
                acc = stats.get("accuracy", 0)
                acc_s = _c(Color.GREEN, f"{acc:.1%}") if acc >= 0.5 else _c(Color.RED, f"{acc:.1%}")
                print(f"    {action_labels.get(action, action)}: {acc_s} ({stats.get('count', 0)} 次)")

        # 按股票
        by_symbol = report_data.get("by_symbol", {})
        if by_symbol:
            print("\n  ── 按股票 ──")
            for symbol, stats in by_symbol.items():
                acc = stats.get("accuracy", 0)
                acc_s = _c(Color.GREEN, f"{acc:.1%}") if acc >= 0.5 else _c(Color.RED, f"{acc:.1%}")
                print(f"    {symbol}: {acc_s} ({stats.get('count', 0)} 次)")

        print()

    @staticmethod
    def _print_help() -> None:
        print(f"""
  {_c(Color.BOLD, '🧠 StockMind 命令列表')}

  analyze <symbol>       分析股票（如: analyze AAPL）
  reflect [days]         反思过去 N 天的决策（默认 1 天）
  watchlist add <symbol>  添加到关注列表
  watchlist remove <sym>  从关注列表移除
  watchlist list          查看关注列表
  watchlist analyze       分析所有关注股票
  report                  查看性能报告
  suggest                 获取改进建议
  skills list             查看技能库
  skills search <query>   搜索技能
  help                    显示帮助
  quit                    退出
""")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockmind",
        description="StockMind · 股票心智 — 越用越强的股票分析 Agent",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="分析股票")
    p_analyze.add_argument("symbol", help="股票代码，如 AAPL")

    # reflect
    p_reflect = sub.add_parser("reflect", help="反思历史决策")
    p_reflect.add_argument("--days-ago", type=int, default=1, help="回溯天数（默认 1）")

    # watchlist
    p_watch = sub.add_parser("watchlist", help="关注列表管理")
    p_watch_sub = p_watch.add_subparsers(dest="watchlist_action")
    p_watch_add = p_watch_sub.add_parser("add", help="添加关注")
    p_watch_add.add_argument("symbol", help="股票代码")
    p_watch_rm = p_watch_sub.add_parser("remove", help="移除关注")
    p_watch_rm.add_argument("symbol", help="股票代码")
    p_watch_sub.add_parser("analyze", help="分析所有关注股票")

    # report
    sub.add_parser("report", help="查看性能报告")

    # suggest
    sub.add_parser("suggest", help="获取改进建议")

    # interactive
    sub.add_parser("interactive", help="启动交互式 REPL")

    # web
    sub.add_parser("web", help="启动 Web 仪表盘")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        agent = StockMindAgent()
    except Exception as e:
        print(_c(Color.RED, f"❌ 初始化失败: {e}"))
        sys.exit(1)

    command = args.command

    # 无参数时默认进入交互模式
    if not command:
        agent.interactive()
        return

    if command == "analyze":
        agent.analyze(args.symbol)

    elif command == "reflect":
        agent.reflect(days_ago=args.days_ago)

    elif command == "watchlist":
        wl_action = getattr(args, "watchlist_action", None)
        if wl_action == "add":
            agent.watchlist_add(args.symbol)
        elif wl_action == "remove":
            agent.watchlist_remove(args.symbol)
        elif wl_action == "analyze":
            agent.watchlist_analyze()
        else:
            agent.watchlist_list()

    elif command == "report":
        agent.report()

    elif command == "suggest":
        agent.suggest()

    elif command == "interactive":
        agent.interactive()

    elif command == "web":
        agent.web()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
