"""
StockMind · 股票心智 — 自我改进技能库
灵感来自 Hermes Agent，在每次决策后自动提取、检索、应用、反思技能
"""
import json
import logging
from typing import Optional

from config import SKILLS_DIR, SKILL_RETRIEVAL_LIMIT, ANALYSIS_DIMENSIONS
from core import database as db
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 技能内容 Markdown 模板
# ═══════════════════════════════════════════════════════════════
SKILL_CONTENT_TEMPLATE = """# Skill: {name}
## Condition
{condition}
## Action
{action}
## Evidence
{evidence}
## Confidence
{confidence}"""


class SkillSystem:
    """自我改进技能库：提取 → 保存 → 检索 → 应用 → 反馈 → 反思"""

    def __init__(self, db_manager, llm_client: LLMClient) -> None:
        """
        初始化技能系统

        Args:
            db_manager: 数据库管理器（当前使用 core.database 模块函数）
            llm_client: LLM 客户端实例，用于技能提取与反思
        """
        self.db = db_manager
        self.llm = llm_client

    # ───────────────────────────────────────────────────────────
    # 技能提取：从决策/反思中提炼可复用经验
    # ───────────────────────────────────────────────────────────

    def extract_skill(
        self,
        symbol: str,
        decision_result: dict,
        reflection_result: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        决策或反思后，用 LLM 提取可复用技能

        Args:
            symbol: 股票代码
            decision_result: 决策结果字典，包含 action / reason / confidence 等
            reflection_result: 可选的反思结果，包含 was_correct / profit_pct 等

        Returns:
            技能字典 {name, category, content, tags}，无值得提取的内容时返回 None
        """
        try:
            # 构建提示上下文
            context_parts = [
                f"股票代码: {symbol}",
                f"决策结果: {json.dumps(decision_result, ensure_ascii=False, default=str)}",
            ]
            if reflection_result:
                context_parts.append(
                    f"反思结果: {json.dumps(reflection_result, ensure_ascii=False, default=str)}"
                )
            context_str = "\n".join(context_parts)

            prompt = (
                "Based on this stock analysis experience, extract a reusable skill.\n\n"
                f"{context_str}\n\n"
                "If this experience contains a genuinely reusable insight (not trivial or obvious), "
                "extract it as a skill. Otherwise return null.\n\n"
                "Return a JSON object with these fields:\n"
                '- "name": short skill name in English snake_case\n'
                f'- "category": one of {ANALYSIS_DIMENSIONS}\n'
                '- "condition": when to apply this skill (1-2 sentences)\n'
                '- "action": what to do when the condition is met (1-2 sentences)\n'
                '- "evidence": brief evidence from this experience\n'
                '- "confidence": High / Medium / Low\n'
                '- "tags": list of 2-5 relevant tags (English)\n\n'
                "If nothing worth extracting, return: {\"skip\": true}"
            )

            messages = [
                {"role": "system", "content": "You are a skill extraction engine for a stock analysis agent."},
                {"role": "user", "content": prompt},
            ]

            result = self.llm.chat_json(messages, temperature=0.2)

            # 判断是否跳过
            if result.get("skip") or not result.get("name"):
                logger.info(f"[技能提取] {symbol} 无值得提取的技能")
                return None

            # 校验 category
            category = result.get("category", "fundamental")
            if category not in ANALYSIS_DIMENSIONS:
                logger.warning(f"[技能提取] 无效分类 '{category}'，回退到 'fundamental'")
                category = "fundamental"

            # 组装 Markdown 内容
            content = SKILL_CONTENT_TEMPLATE.format(
                name=result.get("name", "unnamed_skill"),
                condition=result.get("condition", ""),
                action=result.get("action", ""),
                evidence=result.get("evidence", ""),
                confidence=result.get("confidence", "Medium"),
            )

            skill = {
                "name": result["name"],
                "category": category,
                "content": content,
                "tags": result.get("tags", []),
            }

            logger.info(f"[技能提取] 从 {symbol} 提取技能: {skill['name']} ({category})")
            return skill

        except Exception as e:
            logger.error(f"[技能提取] 提取失败: {e}")
            return None

    # ───────────────────────────────────────────────────────────
    # 技能保存
    # ───────────────────────────────────────────────────────────

    def save_skill(self, skill: dict) -> int:
        """
        保存技能到数据库

        Args:
            skill: 技能字典，需包含 name / category / content / tags

        Returns:
            新技能的 ID；保存失败返回 -1
        """
        try:
            skill_id = db.add_skill(
                name=skill["name"],
                category=skill.get("category", ""),
                content=skill.get("content", ""),
                tags=skill.get("tags", []),
            )
            logger.info(f"[技能保存] 已保存技能 #{skill_id}: {skill['name']}")
            return skill_id
        except Exception as e:
            # 唯一约束冲突（同名技能已存在）
            if "UNIQUE constraint" in str(e):
                logger.warning(f"[技能保存] 技能已存在: {skill['name']}，跳过")
            else:
                logger.error(f"[技能保存] 保存失败: {e}")
            return -1

    # ───────────────────────────────────────────────────────────
    # 技能检索
    # ───────────────────────────────────────────────────────────

    def search_skills(self, query: str, limit: int = SKILL_RETRIEVAL_LIMIT) -> list[dict]:
        """
        使用 FTS5 全文检索技能

        Args:
            query: 检索关键词
            limit: 返回数量上限

        Returns:
            匹配的技能列表
        """
        try:
            results = db.search_skills(query, limit=limit)
            logger.debug(f"[技能检索] 查询 '{query}' 命中 {len(results)} 条技能")
            return results
        except Exception as e:
            logger.error(f"[技能检索] 检索失败: {e}")
            return []

    # ───────────────────────────────────────────────────────────
    # 技能应用：为 LLM 提供相关经验上下文
    # ───────────────────────────────────────────────────────────

    def apply_skills(self, symbol: str, context: str) -> str:
        """
        检索与当前分析相关的技能，格式化为 LLM 上下文

        Args:
            symbol: 股票代码
            context: 当前分析上下文描述（用于检索匹配）

        Returns:
            格式化的技能上下文字符串，无匹配时返回空字符串
        """
        try:
            # 用股票代码 + 上下文关键词检索
            query_parts = [symbol]
            # 从上下文中提取关键词（取前 50 字符避免查询过长）
            if context:
                query_parts.append(context[:50])
            query = " ".join(query_parts)

            skills = self.search_skills(query, limit=SKILL_RETRIEVAL_LIMIT)
            if not skills:
                return ""

            # 格式化为上下文字符串
            lines = ["Relevant past experience:"]
            for i, skill in enumerate(skills, 1):
                # 截取内容摘要（取前 200 字符）
                content_excerpt = skill.get("content", "")
                if len(content_excerpt) > 200:
                    content_excerpt = content_excerpt[:200] + "..."
                category = skill.get("category", "unknown")
                success_rate = skill.get("success_rate", 0)
                lines.append(
                    f"{i}. [{skill.get('name', 'unnamed')}] "
                    f"(category: {category}, success_rate: {success_rate:.0%}):\n"
                    f"   {content_excerpt}"
                )

            result = "\n".join(lines)
            logger.debug(f"[技能应用] 为 {symbol} 匹配到 {len(skills)} 条技能")
            return result

        except Exception as e:
            logger.error(f"[技能应用] 应用技能失败: {e}")
            return ""

    # ───────────────────────────────────────────────────────────
    # 技能反馈更新
    # ───────────────────────────────────────────────────────────

    def update_skill_feedback(self, skill_id: int, was_successful: bool) -> None:
        """
        更新技能使用计数和成功率

        Args:
            skill_id: 技能 ID
            was_successful: 本次应用是否成功
        """
        try:
            db.update_skill_usage(skill_id)
            db.update_skill_success_rate(skill_id, success=was_successful)
            logger.debug(
                f"[技能反馈] 技能 #{skill_id} 反馈: "
                f"{'成功' if was_successful else '失败'}"
            )
        except Exception as e:
            logger.error(f"[技能反馈] 更新失败 (skill_id={skill_id}): {e}")

    # ───────────────────────────────────────────────────────────
    # 技能反思：审查已有技能的时效性与一致性
    # ───────────────────────────────────────────────────────────

    def reflect_on_skills(self) -> list[dict]:
        """
        用 LLM 审查所有技能，识别过时或矛盾的技能

        Returns:
            需要更新或删除的技能列表，每项包含:
            {skill_id, skill_name, action: "update"|"remove", reason}
        """
        try:
            all_skills = db.get_all_skills()
            if not all_skills:
                logger.info("[技能反思] 技能库为空，无需反思")
                return []

            # 构建技能摘要供 LLM 审查
            skill_summaries = []
            for s in all_skills:
                skill_summaries.append({
                    "id": s["id"],
                    "name": s["name"],
                    "category": s["category"],
                    "content_preview": s.get("content", "")[:150],
                    "usage_count": s.get("usage_count", 0),
                    "success_rate": s.get("success_rate", 0),
                    "tags": s.get("tags", []),
                })

            prompt = (
                "Review the following learned skills for a stock analysis agent.\n"
                "Identify skills that are:\n"
                "1. Outdated (based on old market conditions that no longer apply)\n"
                "2. Contradictory (conflict with other skills)\n"
                "3. Low quality (very low success rate after sufficient usage, e.g. usage_count >= 5 and success_rate < 0.3)\n\n"
                f"Skills:\n{json.dumps(skill_summaries, ensure_ascii=False, indent=2)}\n\n"
                "Return a JSON object with key 'actions', which is a list of:\n"
                '- {"skill_id": int, "skill_name": str, "action": "update"|"remove", "reason": str}\n\n'
                "If all skills are fine, return: {\"actions\": []}"
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a skill quality reviewer for an AI stock analysis agent.",
                },
                {"role": "user", "content": prompt},
            ]

            result = self.llm.chat_json(messages, temperature=0.2)

            actions = result.get("actions", [])
            if actions:
                logger.info(f"[技能反思] 识别到 {len(actions)} 条需要处理的技能")
            else:
                logger.info("[技能反思] 所有技能状态良好")

            return actions

        except Exception as e:
            logger.error(f"[技能反思] 反思失败: {e}")
            return []

    # ───────────────────────────────────────────────────────────
    # 技能摘要生成
    # ───────────────────────────────────────────────────────────

    def generate_skill_summary(self) -> str:
        """
        生成人类可读的技能库摘要

        Returns:
            摘要字符串，如 "I have learned 15 skills: 5 fundamental, 4 technical, ..."
        """
        try:
            all_skills = db.get_all_skills()
            total = len(all_skills)
            if total == 0:
                return "I have not learned any skills yet."

            # 按分类统计
            category_counts: dict[str, int] = {dim: 0 for dim in ANALYSIS_DIMENSIONS}
            for skill in all_skills:
                cat = skill.get("category", "")
                if cat in category_counts:
                    category_counts[cat] += 1

            # 分类中文名映射
            category_labels = {
                "fundamental": "fundamental",
                "technical": "technical",
                "sentiment": "sentiment",
                "news": "news",
                "macro": "macro",
            }

            parts = [f"{category_counts[dim]} {category_labels[dim]}" for dim in ANALYSIS_DIMENSIONS if category_counts[dim] > 0]
            detail = ", ".join(parts)

            return f"I have learned {total} skills: {detail}"

        except Exception as e:
            logger.error(f"[技能摘要] 生成失败: {e}")
            return "Unable to generate skill summary."

    # ───────────────────────────────────────────────────────────
    # 自动学习：决策后的主入口
    # ───────────────────────────────────────────────────────────

    def auto_learn(
        self,
        symbol: str,
        decision: dict,
        market_result: Optional[dict] = None,
    ) -> None:
        """
        每次决策后自动调用，驱动技能的反馈更新与新技能提取

        流程:
        1. 如果有市场结果（反思），更新相关技能的反馈
        2. 尝试从本次决策中提取新技能
        3. 如果值得保存，则持久化

        Args:
            symbol: 股票代码
            decision: 决策结果字典
            market_result: 可选的市场验证结果，包含:
                - was_successful: 决策是否正确
                - skill_ids_used: 本次决策使用了哪些技能 ID
                - profit_pct: 收益率
        """
        try:
            # ── 步骤 1: 反馈更新 ──
            if market_result is not None:
                was_successful = market_result.get("was_successful", False)
                skill_ids_used = market_result.get("skill_ids_used", [])

                # 更新已使用技能的反馈
                for sid in skill_ids_used:
                    self.update_skill_feedback(sid, was_successful=was_successful)

                logger.info(
                    f"[自动学习] {symbol} 反馈更新: "
                    f"{'成功' if was_successful else '失败'}，"
                    f"涉及 {len(skill_ids_used)} 条技能"
                )

            # ── 步骤 2: 提取新技能 ──
            reflection = None
            if market_result is not None:
                reflection = market_result

            new_skill = self.extract_skill(symbol, decision, reflection_result=reflection)

            # ── 步骤 3: 保存值得保留的技能 ──
            if new_skill is not None:
                skill_id = self.save_skill(new_skill)
                if skill_id > 0:
                    logger.info(
                        f"[自动学习] {symbol} 新技能已保存: "
                        f"{new_skill['name']} (#{skill_id})"
                    )
                else:
                    logger.info(f"[自动学习] {symbol} 技能未保存（可能已存在）")
            else:
                logger.debug(f"[自动学习] {symbol} 本次决策无值得提取的技能")

        except Exception as e:
            logger.error(f"[自动学习] {symbol} 自动学习失败: {e}")
