"""
StockMind · 股票心智 — 反思/学习模块
回顾历史决策，评估正确性，驱动技能反馈与自动学习
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from config import REFLECTION_INTERVAL_DAYS, MAX_CONSECUTIVE_ERRORS
from core import database as db
from core.llm_client import LLMClient
from core.data_fetcher import StockDataFetcher
from core.skill_system import SkillSystem

logger = logging.getLogger(__name__)


class Reflector:
    """反思引擎：回顾决策 → 评估盈亏 → 生成反思 → 更新学习"""

    def __init__(
        self,
        db_manager: Any,
        llm_client: LLMClient,
        data_fetcher: StockDataFetcher,
        skill_system: SkillSystem,
    ) -> None:
        """
        初始化反思引擎

        Args:
            db_manager: 数据库管理器（core.database 模块）
            llm_client: LLM 客户端实例
            data_fetcher: 数据抓取器实例
            skill_system: 技能系统实例
        """
        self.db = db_manager
        self.llm = llm_client
        self.data_fetcher = data_fetcher
        self.skill_system = skill_system

    # ═══════════════════════════════════════════════════════════════
    # 主入口：reflect_all
    # ═══════════════════════════════════════════════════════════════

    def reflect_all(self, days_ago: int = REFLECTION_INTERVAL_DAYS) -> list[dict]:
        """
        反思指定天数前的所有决策

        流程：获取决策 → 获取当前价 → 评估 → 保存反思 → 更新学习

        Args:
            days_ago: 回溯天数，默认取配置中的反思间隔

        Returns:
            反思结果列表，每项包含 decision_id / was_correct / profit_pct / notes 等
        """
        try:
            decisions = db.get_recent_decisions(days=days_ago)
            if not decisions:
                logger.info(f"[反思] 最近 {days_ago} 天内无决策记录")
                return []

            logger.info(f"[反思] 开始反思 {len(decisions)} 条决策（最近 {days_ago} 天）")
            results: list[dict] = []

            for decision in decisions:
                try:
                    result = self._reflect_single_decision(decision)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        f"[反思] 决策 #{decision.get('id')} 反思失败: {e}"
                    )
                    results.append({
                        "decision_id": decision.get("id"),
                        "symbol": decision.get("symbol", ""),
                        "error": str(e),
                    })

            logger.info(
                f"[反思] 完成 {len(results)} 条反思，"
                f"正确 {sum(1 for r in results if r.get('was_correct'))} 条"
            )
            return results

        except Exception as e:
            logger.error(f"[反思] reflect_all 执行失败: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════
    # 单只股票反思
    # ═══════════════════════════════════════════════════════════════

    def reflect_symbol(self, symbol: str) -> Optional[dict]:
        """
        反思某只股票最近一次决策

        Args:
            symbol: 股票代码

        Returns:
            反思结果字典，无决策时返回 None
        """
        symbol = symbol.upper().strip()
        try:
            decisions = db.get_decisions_by_symbol(symbol, limit=1)
            if not decisions:
                logger.info(f"[反思] {symbol} 无历史决策")
                return None

            decision = decisions[0]

            # 检查是否已有反思（避免重复反思同一条决策）
            existing = db.get_reflections_by_decision(decision["id"])
            if existing:
                logger.info(
                    f"[反思] {symbol} 决策 #{decision['id']} 已有反思，跳过"
                )
                return {
                    "decision_id": decision["id"],
                    "symbol": symbol,
                    "skipped": True,
                    "reason": "已有反思记录",
                }

            return self._reflect_single_decision(decision)

        except Exception as e:
            logger.error(f"[反思] {symbol} 反思失败: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # 内部方法：单条决策反思流程
    # ═══════════════════════════════════════════════════════════════

    def _reflect_single_decision(self, decision: dict) -> Optional[dict]:
        """
        对单条决策执行完整反思流程

        Args:
            decision: 决策记录字典

        Returns:
            反思结果字典
        """
        symbol = decision.get("symbol", "")
        decision_id = decision.get("id")

        # 1. 获取当前价格
        stock_info = self.data_fetcher.fetch_stock_info(symbol)
        current_price = stock_info.get("price", 0.0)
        if not current_price or current_price <= 0:
            logger.warning(f"[反思] {symbol} 无法获取当前价格，跳过")
            return None

        # 2. 评估决策
        evaluation = self._evaluate_decision(decision, current_price)

        # 3. 获取市场上下文（板块表现）
        market_context = self._get_market_context(symbol, stock_info)

        # 4. 生成反思笔记
        notes = self._generate_reflection_note(
            decision, evaluation, market_context
        )

        # 5. 保存反思到数据库
        check_date = datetime.now().strftime("%Y-%m-%d")
        reflection_id = db.add_reflection(
            decision_id=decision_id,
            check_date=check_date,
            price_after=current_price,
            was_correct=1 if evaluation["was_correct"] else 0,
            profit_pct=evaluation["profit_pct"],
            notes=notes,
        )

        # 6. 组装反思结果
        reflection = {
            "decision_id": decision_id,
            "reflection_id": reflection_id,
            "symbol": symbol,
            "action": decision.get("action", ""),
            "price_at_decision": decision.get("price_at_decision"),
            "current_price": current_price,
            "was_correct": evaluation["was_correct"],
            "profit_pct": evaluation["profit_pct"],
            "notes": notes,
        }

        # 7. 更新学习
        self._update_learning(symbol, decision, reflection)

        logger.info(
            f"[反思] {symbol} 决策 #{decision_id}: "
            f"action={decision.get('action')}, "
            f"{'✓ 正确' if evaluation['was_correct'] else '✗ 错误'}, "
            f"收益={evaluation['profit_pct']:+.2f}%"
        )

        return reflection

    # ═══════════════════════════════════════════════════════════════
    # 决策评估
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_decision(
        self, decision: dict, current_price: float
    ) -> dict:
        """
        评估历史决策是否正确

        - buy: 当前价 > 决策价 → 正确
        - sell: 当前价 < 决策价 → 正确
        - hold: 基于机会成本评估

        Args:
            decision: 决策记录
            current_price: 当前价格

        Returns:
            {"was_correct": bool, "profit_pct": float, "notes": str}
        """
        action = decision.get("action", "hold")
        price_at_decision = decision.get("price_at_decision")

        # 无决策价格时无法评估
        if price_at_decision is None or price_at_decision <= 0:
            return {
                "was_correct": False,
                "profit_pct": 0.0,
                "notes": "决策时无价格数据，无法评估",
            }

        # 计算盈亏百分比
        profit_pct = (current_price - price_at_decision) / price_at_decision * 100
        profit_pct = round(profit_pct, 4)

        # 根据行动类型判断正确性
        if action == "buy":
            was_correct = current_price > price_at_decision
            if was_correct:
                notes = f"买入正确：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，盈利 {profit_pct:+.2f}%"
            else:
                notes = f"买入错误：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，亏损 {profit_pct:+.2f}%"

        elif action == "sell":
            was_correct = current_price < price_at_decision
            if was_correct:
                notes = f"卖出正确：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，规避了 {abs(profit_pct):.2f}% 的下跌"
            else:
                notes = f"卖出错误：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，错失了 {profit_pct:+.2f}% 的涨幅"

        elif action == "hold":
            # hold 决策：评估机会成本，与市场平均对比
            was_correct = self._evaluate_hold_decision(
                price_at_decision, current_price
            )
            if was_correct:
                notes = f"观望正确：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，变动 {profit_pct:+.2f}%，避免了不利波动"
            else:
                notes = f"观望错误：决策价 ${price_at_decision:.2f} → 现价 ${current_price:.2f}，错失了 {profit_pct:+.2f}% 的机会"

        else:
            was_correct = False
            notes = f"未知行动类型: {action}"

        return {
            "was_correct": was_correct,
            "profit_pct": profit_pct,
            "notes": notes,
        }

    def _evaluate_hold_decision(
        self, price_at_decision: float, current_price: float
    ) -> bool:
        """
        评估 hold 决策的正确性（基于机会成本）

        hold 正确条件：价格变动幅度在 ±3% 以内（波动不大，观望合理）
        或价格下跌超过 3%（成功规避了下跌）

        Args:
            price_at_decision: 决策时价格
            current_price: 当前价格

        Returns:
            hold 是否正确
        """
        change_pct = (current_price - price_at_decision) / price_at_decision * 100

        # 价格下跌 → 观望正确（规避了亏损）
        if change_pct < -3.0:
            return True
        # 价格小幅波动 → 观望合理
        if abs(change_pct) <= 3.0:
            return True
        # 价格上涨超过 3% → 观望错误（错失了机会）
        return False

    # ═══════════════════════════════════════════════════════════════
    # 反思笔记生成
    # ═══════════════════════════════════════════════════════════════

    def _generate_reflection_note(
        self,
        decision: dict,
        evaluation: dict,
        market_context: dict,
    ) -> str:
        """
        使用 LLM 生成深度反思笔记

        Args:
            decision: 决策记录
            evaluation: 评估结果
            market_context: 市场上下文

        Returns:
            反思笔记字符串
        """
        try:
            symbol = decision.get("symbol", "")
            action = decision.get("action", "hold")
            price_at_decision = decision.get("price_at_decision", 0)
            current_price = evaluation.get("notes", "")
            was_correct = evaluation.get("was_correct", False)
            profit_pct = evaluation.get("profit_pct", 0.0)
            reason = decision.get("reason", "")
            confidence = decision.get("confidence", 0.5)

            # 计算决策距今天数
            decision_date = decision.get("date", "")
            days_passed = self._calc_days_passed(decision_date)

            # 构建市场上下文摘要
            market_summary = json.dumps(
                market_context, ensure_ascii=False, default=str
            )[:300]

            prompt = (
                f"You made a decision {days_passed} days ago to {action} {symbol} "
                f"at ${price_at_decision:.2f}. Now it's at ${evaluation.get('profit_pct', 0):.2f}% change. "
                f"The decision was {'correct' if was_correct else 'wrong'} "
                f"(profit: {profit_pct:+.2f}%).\n\n"
                f"Original reason: {reason}\n"
                f"Confidence at decision: {confidence:.2f}\n"
                f"Market context: {market_summary}\n\n"
                "Reflect on what went right/wrong and what you learned. "
                "Be specific about which factors you over/under-weighted. "
                "Write 2-4 sentences in Chinese."
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a self-reflective stock analysis agent learning from past decisions.",
                },
                {"role": "user", "content": prompt},
            ]

            note = self.llm.chat(messages, temperature=0.3)
            # 清理可能的 LLM 错误前缀
            if note.startswith("[LLM 错误]"):
                logger.warning("[反思] LLM 生成反思笔记失败，使用评估摘要")
                return evaluation.get("notes", "反思生成失败")

            return note.strip()

        except Exception as e:
            logger.error(f"[反思] 生成反思笔记失败: {e}")
            return evaluation.get("notes", f"反思生成异常: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 学习更新
    # ═══════════════════════════════════════════════════════════════

    def _update_learning(
        self,
        symbol: str,
        decision: dict,
        reflection: dict,
    ) -> None:
        """
        核心学习机制：更新技能反馈 → 自动学习 → 记录学习事件

        Args:
            symbol: 股票代码
            decision: 决策记录
            reflection: 反思结果
        """
        try:
            was_correct = reflection.get("was_correct", False)

            # 1. 更新技能反馈
            # 从决策的 dimensions_json 中提取使用的技能 ID
            dimensions_json = decision.get("dimensions_json", "{}")
            if isinstance(dimensions_json, str):
                try:
                    dimensions = json.loads(dimensions_json)
                except json.JSONDecodeError:
                    dimensions = {}
            else:
                dimensions = dimensions_json

            skill_ids_used = dimensions.get("skill_ids_used", [])
            for skill_id in skill_ids_used:
                self.skill_system.update_skill_feedback(
                    skill_id, was_successful=was_correct
                )

            # 2. 调用技能系统自动学习
            market_result = {
                "was_successful": was_correct,
                "skill_ids_used": skill_ids_used,
                "profit_pct": reflection.get("profit_pct", 0.0),
            }
            self.skill_system.auto_learn(symbol, decision, market_result)

            # 3. 记录学习事件
            logger.info(
                f"[学习] {symbol} 学习更新: "
                f"{'正确' if was_correct else '错误'}, "
                f"反馈技能 {len(skill_ids_used)} 条"
            )

        except Exception as e:
            logger.error(f"[学习] {symbol} 学习更新失败: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 性能报告
    # ═══════════════════════════════════════════════════════════════

    def get_performance_report(self) -> dict:
        """
        生成综合性能报告

        Returns:
            包含总览、按行动类型、按股票、趋势等信息的报告字典
        """
        try:
            # 总体统计
            overall_stats = db.get_accuracy_stats()
            total_decisions = overall_stats["total"]
            total_reflections = overall_stats["correct"] + (
                total_decisions - overall_stats["correct"]
            )
            overall_accuracy = overall_stats["accuracy"]

            # 按行动类型统计
            by_action = self._get_accuracy_by_action()

            # 按股票统计
            by_symbol = self._get_accuracy_by_symbol()

            # 近期趋势
            recent_trend = self._determine_recent_trend()

            # 连续错误次数
            consecutive_errors = db.get_consecutive_errors()

            # 技能学习数量
            all_skills = db.get_all_skills()
            skills_learned = len(all_skills)

            # 平均信心度
            decision_stats = db.get_decision_stats()
            avg_confidence = decision_stats.get("avg_confidence", 0.0)

            report = {
                "total_decisions": total_decisions,
                "total_reflections": total_reflections,
                "overall_accuracy": overall_accuracy,
                "by_action": by_action,
                "by_symbol": by_symbol,
                "recent_trend": recent_trend,
                "consecutive_errors": consecutive_errors,
                "skills_learned": skills_learned,
                "avg_confidence": avg_confidence,
            }

            logger.info(
                f"[报告] 总决策 {total_decisions} 条，"
                f"准确率 {overall_accuracy:.1%}，"
                f"趋势 {recent_trend}，"
                f"连续错误 {consecutive_errors} 次"
            )

            return report

        except Exception as e:
            logger.error(f"[报告] 生成性能报告失败: {e}")
            return {
                "total_decisions": 0,
                "total_reflections": 0,
                "overall_accuracy": 0.0,
                "by_action": {},
                "by_symbol": {},
                "recent_trend": "stable",
                "consecutive_errors": 0,
                "skills_learned": 0,
                "avg_confidence": 0.0,
            }

    def _get_accuracy_by_action(self) -> dict[str, dict]:
        """按行动类型统计准确率"""
        by_action: dict[str, dict] = {}
        try:
            for action in ("buy", "sell", "hold"):
                with db.get_connection() as conn:
                    row = conn.execute(
                        """SELECT COUNT(*) as total,
                                  SUM(CASE WHEN r.was_correct = 1 THEN 1 ELSE 0 END) as correct
                           FROM reflections r
                           JOIN decisions d ON r.decision_id = d.id
                           WHERE d.action = ? AND r.was_correct IS NOT NULL""",
                        (action,),
                    ).fetchone()
                    total = row["total"]
                    correct = row["correct"] or 0
                    by_action[action] = {
                        "count": total,
                        "accuracy": round(correct / total, 3) if total > 0 else 0.0,
                    }
        except Exception as e:
            logger.error(f"[报告] 按行动统计失败: {e}")
        return by_action

    def _get_accuracy_by_symbol(self) -> dict[str, dict]:
        """按股票统计准确率"""
        by_symbol: dict[str, dict] = {}
        try:
            with db.get_connection() as conn:
                rows = conn.execute(
                    """SELECT d.symbol,
                              COUNT(*) as total,
                              SUM(CASE WHEN r.was_correct = 1 THEN 1 ELSE 0 END) as correct
                       FROM reflections r
                       JOIN decisions d ON r.decision_id = d.id
                       WHERE r.was_correct IS NOT NULL
                       GROUP BY d.symbol
                       ORDER BY total DESC""",
                ).fetchall()
                for row in rows:
                    total = row["total"]
                    correct = row["correct"] or 0
                    by_symbol[row["symbol"]] = {
                        "count": total,
                        "accuracy": round(correct / total, 3) if total > 0 else 0.0,
                    }
        except Exception as e:
            logger.error(f"[报告] 按股票统计失败: {e}")
        return by_symbol

    def _determine_recent_trend(self) -> str:
        """
        判断近期趋势：improving / declining / stable

        比较最近 7 条反思与前 7 条反思的准确率
        """
        try:
            with db.get_connection() as conn:
                rows = conn.execute(
                    """SELECT was_correct FROM reflections
                       WHERE was_correct IS NOT NULL
                       ORDER BY check_date DESC LIMIT 14"""
                ).fetchall()

            if len(rows) < 4:
                return "stable"

            # 分为两半：近期 vs 早期
            mid = len(rows) // 2
            recent = rows[:mid]
            older = rows[mid:]

            recent_accuracy = sum(1 for r in recent if r["was_correct"] == 1) / len(recent)
            older_accuracy = sum(1 for r in older if r["was_correct"] == 1) / len(older)

            diff = recent_accuracy - older_accuracy
            if diff > 0.1:
                return "improving"
            elif diff < -0.1:
                return "declining"
            return "stable"

        except Exception:
            return "stable"

    # ═══════════════════════════════════════════════════════════════
    # 改进建议
    # ═══════════════════════════════════════════════════════════════

    def suggest_improvements(self) -> list[str]:
        """
        使用 LLM 分析性能报告并生成具体改进建议

        Returns:
            改进建议字符串列表
        """
        try:
            report = self.get_performance_report()

            # 报告数据不足时给出基础建议
            if report["total_reflections"] < 3:
                return [
                    "反思数据不足，继续积累决策和反思记录后再分析改进方向。",
                    f"当前已有 {report['total_decisions']} 条决策，"
                    f"{report['total_reflections']} 条反思，建议至少积累 10 条以上。",
                ]

            # 构建给 LLM 的报告摘要
            report_summary = {
                "overall_accuracy": f"{report['overall_accuracy']:.1%}",
                "total_decisions": report["total_decisions"],
                "total_reflections": report["total_reflections"],
                "by_action": {
                    k: f"{v['accuracy']:.1%} ({v['count']}次)"
                    for k, v in report["by_action"].items()
                },
                "by_symbol": {
                    k: f"{v['accuracy']:.1%} ({v['count']}次)"
                    for k, v in report["by_symbol"].items()
                },
                "recent_trend": report["recent_trend"],
                "consecutive_errors": report["consecutive_errors"],
                "skills_learned": report["skills_learned"],
                "avg_confidence": f"{report['avg_confidence']:.2f}",
            }

            prompt = (
                "Analyze this performance report for a stock analysis agent "
                "and suggest 3-5 specific, actionable improvements.\n\n"
                f"Report:\n{json.dumps(report_summary, ensure_ascii=False, indent=2)}\n\n"
                "Guidelines:\n"
                "- Be specific: mention exact stocks, actions, or metrics that need improvement\n"
                "- Example: 'Your accuracy on TSLA is low (30%). "
                "Consider paying more attention to technical indicators for volatile stocks.'\n"
                "- Focus on the weakest areas first\n"
                "- Suggest concrete analysis adjustments, not vague advice\n\n"
                "Return a JSON object:\n"
                '{ "suggestions": ["<suggestion 1 in Chinese>", "<suggestion 2>", ...] }'
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a performance coach for an AI stock analysis agent. "
                        "Give specific, data-driven improvement suggestions."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            result = self.llm.chat_json(messages, temperature=0.3)

            if "error" in result:
                logger.warning("[建议] LLM 生成改进建议失败，使用规则引擎")
                return self._rule_suggest_improvements(report)

            suggestions = result.get("suggestions", [])
            if not suggestions:
                return self._rule_suggest_improvements(report)

            logger.info(f"[建议] 生成 {len(suggestions)} 条改进建议")
            return suggestions

        except Exception as e:
            logger.error(f"[建议] 生成改进建议失败: {e}")
            return ["改进建议生成异常，请检查日志。"]

    def _rule_suggest_improvements(self, report: dict) -> list[str]:
        """规则引擎改进建议（LLM 失败时的兜底）"""
        suggestions: list[str] = []

        # 连续错误过多
        if report["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
            suggestions.append(
                f"连续错误已达 {report['consecutive_errors']} 次，"
                f"超过阈值 {MAX_CONSECUTIVE_ERRORS}，建议暂停主动决策，转为观望模式。"
            )

        # 整体准确率低
        if report["overall_accuracy"] < 0.4 and report["total_reflections"] >= 5:
            suggestions.append(
                f"整体准确率仅 {report['overall_accuracy']:.0%}，"
                "建议重新审视分析框架，加强多维度交叉验证。"
            )

        # 按行动类型分析
        for action, stats in report["by_action"].items():
            if stats["count"] >= 3 and stats["accuracy"] < 0.35:
                action_label = {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(
                    action, action
                )
                suggestions.append(
                    f"{action_label}决策准确率偏低（{stats['accuracy']:.0%}，"
                    f"共 {stats['count']} 次），建议提高该类决策的信心阈值。"
                )

        # 按股票分析
        for symbol, stats in report["by_symbol"].items():
            if stats["count"] >= 3 and stats["accuracy"] < 0.35:
                suggestions.append(
                    f"对 {symbol} 的分析准确率较低（{stats['accuracy']:.0%}，"
                    f"共 {stats['count']} 次），建议加强对该股票的技术面和基本面研究。"
                )

        # 趋势分析
        if report["recent_trend"] == "declining":
            suggestions.append(
                "近期准确率呈下降趋势，建议回顾最近的错误决策，"
                "识别是否有系统性偏差（如过度乐观或忽视风险信号）。"
            )

        # 信心度分析
        if report["avg_confidence"] > 0.8 and report["overall_accuracy"] < 0.5:
            suggestions.append(
                "平均信心度偏高但准确率不足，存在过度自信问题，"
                "建议在分析中加入更多反面论证。"
            )

        if not suggestions:
            suggestions.append("当前表现良好，继续保持多维度分析策略。")

        return suggestions

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════

    def _get_market_context(self, symbol: str, stock_info: dict) -> dict:
        """
        获取市场上下文信息，供反思参考

        Args:
            symbol: 股票代码
            stock_info: 股票基本信息

        Returns:
            市场上下文字典
        """
        try:
            sector_perf = self.data_fetcher.fetch_sector_performance()
            sector = stock_info.get("sector", "")
            sectors = sector_perf.get("sectors", {})

            # 找到所属板块表现
            sector_change = None
            for sec_name, sec_data in sectors.items():
                if sector and sec_name.lower() in sector.lower():
                    sector_change = sec_data.get("change_pct_1mo")
                    break

            return {
                "symbol": symbol,
                "sector": sector,
                "sector_change_pct": sector_change,
                "current_price": stock_info.get("price", 0),
            }
        except Exception as e:
            logger.warning(f"[反思] 获取 {symbol} 市场上下文失败: {e}")
            return {"symbol": symbol, "sector": "", "sector_change_pct": None}

    @staticmethod
    def _calc_days_passed(decision_date: str) -> int:
        """
        计算决策距今的天数

        Args:
            decision_date: 决策日期字符串 (YYYY-MM-DD)

        Returns:
            距今天数
        """
        try:
            dec_dt = datetime.strptime(decision_date, "%Y-%m-%d")
            return (datetime.now() - dec_dt).days
        except (ValueError, TypeError):
            return REFLECTION_INTERVAL_DAYS
