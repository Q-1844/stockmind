"""
StockMind · 股票心智 — 核心分析引擎
整合所有数据源、LLM 推理与技能检索，产出股票分析决策
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from config import (
    HISTORY_LIMIT,
    CONFIDENCE_THRESHOLD,
    MAX_CONSECUTIVE_ERRORS,
    ANALYSIS_DIMENSIONS,
)
from core import database as db
from core.llm_client import LLMClient
from core.data_fetcher import StockDataFetcher
from core.skill_system import SkillSystem

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 维度权重配置
# ═══════════════════════════════════════════════════════════════
DIMENSION_WEIGHTS = {
    "fundamental": 0.30,
    "technical": 0.20,
    "sentiment": 0.15,
    "news": 0.20,
    "macro": 0.15,
}


class StockAnalyzer:
    """核心分析引擎：多维度分析 → LLM 综合 → 风控校验 → 决策输出"""

    def __init__(
        self,
        db_manager: Any,
        llm_client: LLMClient,
        data_fetcher: StockDataFetcher,
        skill_system: SkillSystem,
    ) -> None:
        """
        初始化分析引擎

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

    # ═══════════════════════════════════════════════════════════
    # 主入口：analyze
    # ═══════════════════════════════════════════════════════════

    def analyze(self, symbol: str) -> dict:
        """
        对指定股票执行完整分析，返回决策结果

        Args:
            symbol: 股票代码，如 "AAPL"

        Returns:
            完整决策字典，包含 action / confidence / reason / dimensions 等
        """
        symbol = symbol.upper().strip()
        logger.info(f"[分析引擎] 开始分析 {symbol}")

        try:
            # ── 1. 获取全量数据 ──
            stock_info = self.data_fetcher.fetch_stock_info(symbol)
            financials = self.data_fetcher.fetch_financial_statements(symbol)
            prices = self.data_fetcher.fetch_historical_prices(symbol, period="3mo")
            news = self.data_fetcher.fetch_news(symbol)
            indicators = self.data_fetcher.fetch_technical_indicators(symbol)
            sector_perf = self.data_fetcher.fetch_sector_performance()
            earnings = self.data_fetcher.fetch_earnings_calendar(symbol)

            # ── 2. 五维度分析 ──
            dimensions: dict[str, dict] = {}

            dimensions["fundamental"] = self._analyze_fundamental(symbol, stock_info, financials)
            dimensions["technical"] = self._analyze_technical(symbol, indicators, prices)
            dimensions["sentiment"] = self._analyze_sentiment(symbol, news)
            dimensions["news"] = self._analyze_news(symbol, news, earnings)
            dimensions["macro"] = self._analyze_macro(symbol, sector_perf, stock_info)

            # ── 3. 获取历史决策 ──
            history = db.get_decisions_by_symbol(symbol, limit=HISTORY_LIMIT)

            # ── 4. 检索相关技能 ──
            context_desc = f"{stock_info.get('sector', '')} {stock_info.get('industry', '')} analysis"
            skills_context = self.skill_system.apply_skills(symbol, context_desc)

            # ── 5. 综合决策 ──
            decision = self._synthesize_decision(symbol, dimensions, history, skills_context)

            # ── 6. 风控校验 ──
            consecutive_errors = db.get_consecutive_errors(symbol)
            decision = self._check_risk(symbol, decision, consecutive_errors)

            # ── 7. 计算目标价 ──
            price_target = self._estimate_price_target(stock_info, decision)

            # ── 8. 组装结果 ──
            result = {
                "symbol": symbol,
                "action": decision["action"],
                "confidence": round(decision["confidence"], 4),
                "reason": decision["reason"],
                "dimensions": dimensions,
                "price_target": price_target,
                "risk_level": decision["risk_level"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.info(
                f"[分析引擎] {symbol} 分析完成: "
                f"action={result['action']}, confidence={result['confidence']:.2f}, "
                f"risk={result['risk_level']}"
            )
            return result

        except Exception as e:
            logger.error(f"[分析引擎] {symbol} 分析失败: {e}")
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.0,
                "reason": f"分析过程异常: {str(e)}",
                "dimensions": {dim: {"score": 0.0, "reason": "分析失败"} for dim in ANALYSIS_DIMENSIONS},
                "price_target": None,
                "risk_level": "high",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    # ═══════════════════════════════════════════════════════════
    # 维度一：基本面分析
    # ═══════════════════════════════════════════════════════════

    def _analyze_fundamental(self, symbol: str, stock_info: dict, financials: dict) -> dict:
        """
        基本面分析：PE 估值、营收增长、利润率、负债比率

        Args:
            symbol: 股票代码
            stock_info: 股票基本信息
            financials: 财务报表数据

        Returns:
            {"score": float(-1~1), "reason": str}
        """
        try:
            # 尝试 LLM 分析
            result = self._llm_fundamental(symbol, stock_info, financials)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"[基本面] {symbol} LLM 分析失败，回退规则引擎: {e}")

        # 规则引擎兜底
        return self._rule_fundamental(stock_info, financials)

    def _llm_fundamental(self, symbol: str, stock_info: dict, financials: dict) -> Optional[dict]:
        """使用 LLM 进行基本面分析"""
        # 构建精简的数据摘要，避免 token 浪费
        data_summary = {
            "pe_ratio": stock_info.get("pe_ratio"),
            "eps": stock_info.get("eps"),
            "revenue_growth": stock_info.get("revenue_growth"),
            "profit_margin": stock_info.get("profit_margin"),
            "dividend_yield": stock_info.get("dividend_yield"),
            "sector": stock_info.get("sector"),
            "industry": stock_info.get("industry"),
            "income_quarterly": financials.get("income_statement", {}).get("quarterly", {}),
            "balance_quarterly": financials.get("balance_sheet", {}).get("quarterly", {}),
        }

        prompt = (
            f"Analyze the fundamental data for stock {symbol}.\n\n"
            f"Data:\n{json.dumps(data_summary, ensure_ascii=False, default=str)}\n\n"
            "Evaluate these aspects:\n"
            "1. PE ratio valuation (compared to sector average, overvalued if PE > 30)\n"
            "2. Revenue growth (positive growth is bullish)\n"
            "3. Profit margin (higher is better, >20% is strong)\n"
            "4. Debt ratio (total debt / total assets, lower is better)\n\n"
            "Return a JSON object:\n"
            '{ "score": <float between -1 and 1>, "reason": "<brief explanation in Chinese>" }\n'
            "Score guide: -1=very bearish, -0.5=bearish, 0=neutral, 0.5=bullish, 1=very bullish"
        )

        messages = [
            {"role": "system", "content": "You are a fundamental stock analyst. Be concise and data-driven."},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        reason = str(result.get("reason", "LLM 基本面分析"))
        return {"score": round(score, 4), "reason": reason}

    def _rule_fundamental(self, stock_info: dict, financials: dict) -> dict:
        """规则引擎基本面分析（LLM 失败时的兜底方案）"""
        score = 0.0
        reasons = []

        # PE 估值评分
        pe = stock_info.get("pe_ratio")
        if pe is not None:
            if pe < 0:
                score -= 0.3
                reasons.append("亏损状态(PE为负)")
            elif pe < 15:
                score += 0.3
                reasons.append(f"低估值(PE={pe:.1f})")
            elif pe < 25:
                score += 0.1
                reasons.append(f"合理估值(PE={pe:.1f})")
            elif pe < 40:
                score -= 0.1
                reasons.append(f"偏高估值(PE={pe:.1f})")
            else:
                score -= 0.3
                reasons.append(f"高估值风险(PE={pe:.1f})")

        # 营收增长评分
        rev_growth = stock_info.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > 0.3:
                score += 0.3
                reasons.append(f"高增长(营收+{rev_growth:.0%})")
            elif rev_growth > 0.1:
                score += 0.15
                reasons.append(f"稳健增长(营收+{rev_growth:.0%})")
            elif rev_growth > 0:
                score += 0.05
                reasons.append(f"微增(营收+{rev_growth:.0%})")
            elif rev_growth > -0.1:
                score -= 0.1
                reasons.append(f"营收微降({rev_growth:.0%})")
            else:
                score -= 0.3
                reasons.append(f"营收大幅下滑({rev_growth:.0%})")

        # 利润率评分
        margin = stock_info.get("profit_margin")
        if margin is not None:
            if margin > 0.2:
                score += 0.25
                reasons.append(f"高利润率({margin:.0%})")
            elif margin > 0.1:
                score += 0.1
                reasons.append(f"利润率良好({margin:.0%})")
            elif margin > 0:
                score += 0.0
                reasons.append(f"利润率偏低({margin:.0%})")
            else:
                score -= 0.2
                reasons.append("亏损")

        # 负债比率（从资产负债表提取）
        balance = financials.get("balance_sheet", {}).get("quarterly", {})
        debt_ratio = self._extract_debt_ratio(balance)
        if debt_ratio is not None:
            if debt_ratio < 0.3:
                score += 0.15
                reasons.append(f"低负债率({debt_ratio:.0%})")
            elif debt_ratio < 0.5:
                score += 0.0
                reasons.append(f"负债率适中({debt_ratio:.0%})")
            elif debt_ratio < 0.7:
                score -= 0.1
                reasons.append(f"负债率偏高({debt_ratio:.0%})")
            else:
                score -= 0.25
                reasons.append(f"高负债风险({debt_ratio:.0%})")

        score = max(-1.0, min(1.0, score))
        reason = "；".join(reasons) if reasons else "数据不足，无法评估基本面"
        return {"score": round(score, 4), "reason": reason}

    @staticmethod
    def _extract_debt_ratio(balance_data: dict) -> Optional[float]:
        """从资产负债表数据中提取负债比率"""
        if not balance_data:
            return None
        try:
            # 取最近一期数据
            periods = list(balance_data.values())
            if not periods:
                return None
            latest = periods[0] if isinstance(periods[0], dict) else {}
            total_debt = latest.get("Total Debt") or latest.get("Long Term Debt") or 0
            total_assets = latest.get("Total Assets") or 0
            if total_assets and total_assets > 0:
                return float(total_debt) / float(total_assets)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        return None

    # ═══════════════════════════════════════════════════════════
    # 维度二：技术面分析
    # ═══════════════════════════════════════════════════════════

    def _analyze_technical(self, symbol: str, indicators: dict, prices: list) -> dict:
        """
        技术面分析：SMA 趋势、RSI 超买超卖、MACD 交叉、布林带位置

        Args:
            symbol: 股票代码
            indicators: 技术指标数据
            prices: 历史价格列表

        Returns:
            {"score": float(-1~1), "reason": str}
        """
        try:
            result = self._llm_technical(symbol, indicators, prices)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"[技术面] {symbol} LLM 分析失败，回退规则引擎: {e}")

        return self._rule_technical(indicators)

    def _llm_technical(self, symbol: str, indicators: dict, prices: list) -> Optional[dict]:
        """使用 LLM 进行技术面分析"""
        # 只取最近 10 条价格避免 token 过长
        recent_prices = prices[-10:] if len(prices) > 10 else prices
        price_summary = [
            {"date": p.get("date", ""), "close": p.get("close", 0)}
            for p in recent_prices
        ]

        data_summary = {
            "current_price": indicators.get("current_price"),
            "sma_20": indicators.get("sma_20"),
            "sma_50": indicators.get("sma_50"),
            "sma_200": indicators.get("sma_200"),
            "price_vs_sma20": indicators.get("price_vs_sma20"),
            "price_vs_sma50": indicators.get("price_vs_sma50"),
            "price_vs_sma200": indicators.get("price_vs_sma200"),
            "rsi_14": indicators.get("rsi_14"),
            "macd": indicators.get("macd"),
            "bollinger_bands": indicators.get("bollinger_bands"),
            "volume_ratio": indicators.get("volume_ratio"),
            "recent_closes": price_summary,
        }

        prompt = (
            f"Analyze the technical indicators for stock {symbol}.\n\n"
            f"Data:\n{json.dumps(data_summary, ensure_ascii=False, default=str)}\n\n"
            "Evaluate:\n"
            "1. SMA trend (price above SMA20/50/200 = bullish; golden cross vs death cross)\n"
            "2. RSI (above 70 = overbought; below 30 = oversold)\n"
            "3. MACD crossover (MACD above signal = bullish; histogram direction)\n"
            "4. Bollinger Band position (near upper = overbought; near lower = oversold)\n\n"
            "Return a JSON object:\n"
            '{ "score": <float between -1 and 1>, "reason": "<brief explanation in Chinese>" }'
        )

        messages = [
            {"role": "system", "content": "You are a technical stock analyst. Be concise and data-driven."},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        reason = str(result.get("reason", "LLM 技术面分析"))
        return {"score": round(score, 4), "reason": reason}

    def _rule_technical(self, indicators: dict) -> dict:
        """规则引擎技术面分析"""
        score = 0.0
        reasons = []

        current_price = indicators.get("current_price", 0)

        # SMA 趋势评分
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        sma_200 = indicators.get("sma_200")

        if sma_20 and current_price:
            if current_price > sma_20:
                score += 0.15
                reasons.append("价格在SMA20上方")
            else:
                score -= 0.15
                reasons.append("价格在SMA20下方")

        if sma_50 and current_price:
            if current_price > sma_50:
                score += 0.1
                reasons.append("价格在SMA50上方")
            else:
                score -= 0.1
                reasons.append("价格在SMA50下方")

        if sma_200 and current_price:
            if current_price > sma_200:
                score += 0.1
                reasons.append("价格在SMA200上方(长期趋势向上)")
            else:
                score -= 0.1
                reasons.append("价格在SMA200下方(长期趋势向下)")

        # 金叉/死叉判断
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                score += 0.1
                reasons.append("SMA20>SMA50(短期趋势强)")
            else:
                score -= 0.1
                reasons.append("SMA20<SMA50(短期趋势弱)")

        # RSI 评分
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            if rsi > 80:
                score -= 0.3
                reasons.append(f"严重超买(RSI={rsi:.0f})")
            elif rsi > 70:
                score -= 0.15
                reasons.append(f"超买(RSI={rsi:.0f})")
            elif rsi < 20:
                score += 0.3
                reasons.append(f"严重超卖(RSI={rsi:.0f})")
            elif rsi < 30:
                score += 0.15
                reasons.append(f"超卖(RSI={rsi:.0f})")
            else:
                reasons.append(f"RSI中性({rsi:.0f})")

        # MACD 评分
        macd = indicators.get("macd", {})
        macd_line = macd.get("macd_line")
        signal_line = macd.get("signal_line")
        histogram = macd.get("histogram")

        if macd_line is not None and signal_line is not None:
            if macd_line > signal_line:
                score += 0.15
                reasons.append("MACD金叉(多头信号)")
            else:
                score -= 0.15
                reasons.append("MACD死叉(空头信号)")

        if histogram is not None:
            if histogram > 0:
                score += 0.05
                reasons.append("MACD柱状图为正")
            else:
                score -= 0.05
                reasons.append("MACD柱状图为负")

        # 布林带评分
        bollinger = indicators.get("bollinger_bands", {})
        upper = bollinger.get("upper")
        lower = bollinger.get("lower")
        middle = bollinger.get("middle")

        if upper and lower and middle and current_price:
            band_width = upper - lower
            if band_width > 0:
                position = (current_price - lower) / band_width
                if position > 0.9:
                    score -= 0.15
                    reasons.append("接近布林上轨(超买)")
                elif position > 0.7:
                    score -= 0.05
                    reasons.append("布林带偏上")
                elif position < 0.1:
                    score += 0.15
                    reasons.append("接近布林下轨(超卖)")
                elif position < 0.3:
                    score += 0.05
                    reasons.append("布林带偏下")

        # 成交量比评分
        vol_ratio = indicators.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio > 2.0:
                reasons.append(f"放量明显(量比={vol_ratio:.1f})")
            elif vol_ratio > 1.5:
                reasons.append(f"温和放量(量比={vol_ratio:.1f})")
            elif vol_ratio < 0.5:
                reasons.append(f"缩量明显(量比={vol_ratio:.1f})")

        score = max(-1.0, min(1.0, score))
        reason = "；".join(reasons) if reasons else "技术指标数据不足"
        return {"score": round(score, 4), "reason": reason}

    # ═══════════════════════════════════════════════════════════
    # 维度三：情绪面分析
    # ═══════════════════════════════════════════════════════════

    def _analyze_sentiment(self, symbol: str, news: list) -> dict:
        """
        情绪面分析：新闻正负面计数、LLM 细粒度情绪解读

        Args:
            symbol: 股票代码
            news: 新闻列表

        Returns:
            {"score": float(-1~1), "reason": str}
        """
        if not news:
            return {"score": 0.0, "reason": "无新闻数据，情绪中性"}

        try:
            result = self._llm_sentiment(symbol, news)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"[情绪面] {symbol} LLM 分析失败，回退规则引擎: {e}")

        return self._rule_sentiment(news)

    def _llm_sentiment(self, symbol: str, news: list) -> Optional[dict]:
        """使用 LLM 进行情绪面分析"""
        # 只取前 8 条新闻，控制 token
        news_sample = news[:8]
        news_summaries = [
            {"title": n.get("title", ""), "hint": n.get("sentiment_hint", "")}
            for n in news_sample
        ]

        prompt = (
            f"Analyze the market sentiment for stock {symbol} based on recent news.\n\n"
            f"News:\n{json.dumps(news_summaries, ensure_ascii=False, default=str)}\n\n"
            "Consider:\n"
            "1. Overall tone of news coverage\n"
            "2. Proportion of positive vs negative headlines\n"
            "3. Severity of negative news (regulatory, fraud vs minor miss)\n"
            "4. Market narrative trend (improving or deteriorating)\n\n"
            "Return a JSON object:\n"
            '{ "score": <float between -1 and 1>, "reason": "<brief explanation in Chinese>" }'
        )

        messages = [
            {"role": "system", "content": "You are a sentiment analyst. Be concise and objective."},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        reason = str(result.get("reason", "LLM 情绪面分析"))
        return {"score": round(score, 4), "reason": reason}

    def _rule_sentiment(self, news: list) -> dict:
        """规则引擎情绪面分析"""
        positive = 0
        negative = 0
        neutral = 0

        for item in news:
            hint = item.get("sentiment_hint", "neutral")
            if hint == "positive":
                positive += 1
            elif hint == "negative":
                negative += 1
            else:
                neutral += 1

        total = len(news)
        if total == 0:
            return {"score": 0.0, "reason": "无新闻数据"}

        # 计算情绪得分
        score = (positive - negative) / total
        score = max(-1.0, min(1.0, score))

        reason = f"正面{positive}条，负面{negative}条，中性{neutral}条"
        return {"score": round(score, 4), "reason": reason}

    # ═══════════════════════════════════════════════════════════
    # 维度四：新闻面分析
    # ═══════════════════════════════════════════════════════════

    def _analyze_news(self, symbol: str, news: list, earnings: list) -> dict:
        """
        新闻面分析：新闻事件影响评估 + 财报催化剂

        Args:
            symbol: 股票代码
            news: 新闻列表
            earnings: 财报日历列表

        Returns:
            {"score": float(-1~1), "reason": str}
        """
        try:
            result = self._llm_news(symbol, news, earnings)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"[新闻面] {symbol} LLM 分析失败，回退规则引擎: {e}")

        return self._rule_news(news, earnings)

    def _llm_news(self, symbol: str, news: list, earnings: list) -> Optional[dict]:
        """使用 LLM 进行新闻面分析"""
        news_sample = news[:8]
        news_titles = [n.get("title", "") for n in news_sample]
        earnings_summary = [
            {"date": e.get("date", ""), "eps_estimate": e.get("eps_estimate")}
            for e in earnings[:4]
        ]

        prompt = (
            f"Analyze the news impact and upcoming catalysts for stock {symbol}.\n\n"
            f"Recent headlines:\n" + "\n".join(f"- {t}" for t in news_titles) + "\n\n"
            f"Upcoming earnings:\n{json.dumps(earnings_summary, ensure_ascii=False, default=str)}\n\n"
            "Evaluate:\n"
            "1. Materiality of news events (earnings, M&A, regulatory, product launches)\n"
            "2. Whether upcoming earnings could be a catalyst\n"
            "3. Overall news trajectory (improving, stable, deteriorating)\n\n"
            "Return a JSON object:\n"
            '{ "score": <float between -1 and 1>, "reason": "<brief explanation in Chinese>" }'
        )

        messages = [
            {"role": "system", "content": "You are a news impact analyst. Focus on materiality and catalysts."},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        reason = str(result.get("reason", "LLM 新闻面分析"))
        return {"score": round(score, 4), "reason": reason}

    def _rule_news(self, news: list, earnings: list) -> dict:
        """规则引擎新闻面分析"""
        score = 0.0
        reasons = []

        # 基于新闻标题关键词评估影响
        high_impact_positive = ["beat", "upgrade", "fda approval", "buyback", "dividend increase"]
        high_impact_negative = ["miss", "downgrade", "lawsuit", "investigation", "fraud", "recall", "bankruptcy"]

        for item in news:
            title = item.get("title", "").lower()
            for kw in high_impact_positive:
                if kw in title:
                    score += 0.1
                    reasons.append(f"利好消息: {item.get('title', '')[:30]}")
                    break
            for kw in high_impact_negative:
                if kw in title:
                    score -= 0.15
                    reasons.append(f"利空消息: {item.get('title', '')[:30]}")
                    break

        # 财报催化剂
        if earnings:
            score += 0.05
            reasons.append(f"即将发布财报({len(earnings)}个日期)")

        score = max(-1.0, min(1.0, score))
        reason = "；".join(reasons) if reasons else "无重大新闻事件"
        return {"score": round(score, 4), "reason": reason}

    # ═══════════════════════════════════════════════════════════
    # 维度五：宏观面分析
    # ═══════════════════════════════════════════════════════════

    def _analyze_macro(self, symbol: str, sector_perf: dict, stock_info: dict) -> dict:
        """
        宏观面分析：板块表现、宏观环境关联度

        Args:
            symbol: 股票代码
            sector_perf: 板块表现数据
            stock_info: 股票基本信息

        Returns:
            {"score": float(-1~1), "reason": str}
        """
        try:
            result = self._llm_macro(symbol, sector_perf, stock_info)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"[宏观面] {symbol} LLM 分析失败，回退规则引擎: {e}")

        return self._rule_macro(sector_perf, stock_info)

    def _llm_macro(self, symbol: str, sector_perf: dict, stock_info: dict) -> Optional[dict]:
        """使用 LLM 进行宏观面分析"""
        sector = stock_info.get("sector", "")
        industry = stock_info.get("industry", "")

        # 提取板块表现摘要
        sectors = sector_perf.get("sectors", {})
        sector_summary = {}
        for sec_name, sec_data in sectors.items():
            change = sec_data.get("change_pct_1mo")
            if change is not None:
                sector_summary[sec_name] = f"{change:+.1f}%"

        prompt = (
            f"Analyze the macro environment for stock {symbol} "
            f"(sector: {sector}, industry: {industry}).\n\n"
            f"Sector performance (1-month change):\n"
            f"{json.dumps(sector_summary, ensure_ascii=False, indent=2)}\n\n"
            "Evaluate:\n"
            "1. How the stock's sector is performing relative to others\n"
            "2. Whether the macro environment favors this sector\n"
            "3. Sector rotation trends (money flowing in or out)\n\n"
            "Return a JSON object:\n"
            '{ "score": <float between -1 and 1>, "reason": "<brief explanation in Chinese>" }'
        )

        messages = [
            {"role": "system", "content": "You are a macro economic analyst. Be concise and forward-looking."},
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        reason = str(result.get("reason", "LLM 宏观面分析"))
        return {"score": round(score, 4), "reason": reason}

    def _rule_macro(self, sector_perf: dict, stock_info: dict) -> dict:
        """规则引擎宏观面分析"""
        score = 0.0
        reasons = []

        stock_sector = stock_info.get("sector", "")
        sectors = sector_perf.get("sectors", {})

        # 查找所属板块表现
        stock_sector_change = None
        sector_changes = []

        for sec_name, sec_data in sectors.items():
            change = sec_data.get("change_pct_1mo")
            if change is not None:
                sector_changes.append(change)
                # 匹配股票所属板块
                if stock_sector and sec_name.lower() in stock_sector.lower():
                    stock_sector_change = change

        if not sector_changes:
            return {"score": 0.0, "reason": "板块数据不足"}

        avg_sector_change = sum(sector_changes) / len(sector_changes)

        # 所属板块表现评分
        if stock_sector_change is not None:
            if stock_sector_change > 5:
                score += 0.3
                reasons.append(f"所属板块强势({stock_sector_change:+.1f}%)")
            elif stock_sector_change > 0:
                score += 0.15
                reasons.append(f"所属板块正增长({stock_sector_change:+.1f}%)")
            elif stock_sector_change > -5:
                score -= 0.1
                reasons.append(f"所属板块微跌({stock_sector_change:+.1f}%)")
            else:
                score -= 0.3
                reasons.append(f"所属板块弱势({stock_sector_change:+.1f}%)")

            # 相对大盘板块的表现
            relative = stock_sector_change - avg_sector_change
            if relative > 3:
                score += 0.1
                reasons.append("板块跑赢大盘")
            elif relative < -3:
                score -= 0.1
                reasons.append("板块跑输大盘")

        # 整体市场环境
        positive_sectors = sum(1 for c in sector_changes if c > 0)
        total_sectors = len(sector_changes)
        if positive_sectors > total_sectors * 0.7:
            score += 0.1
            reasons.append("多数板块上涨(市场偏多)")
        elif positive_sectors < total_sectors * 0.3:
            score -= 0.1
            reasons.append("多数板块下跌(市场偏空)")

        score = max(-1.0, min(1.0, score))
        reason = "；".join(reasons) if reasons else "宏观环境中性"
        return {"score": round(score, 4), "reason": reason}

    # ═══════════════════════════════════════════════════════════
    # 综合决策
    # ═══════════════════════════════════════════════════════════

    def _synthesize_decision(
        self,
        symbol: str,
        dimensions: dict,
        history: list,
        skills_context: str,
    ) -> dict:
        """
        综合所有维度得分，结合历史决策和技能经验，产出最终决策

        Args:
            symbol: 股票代码
            dimensions: 五维度分析结果
            history: 历史决策列表
            skills_context: 技能上下文字符串

        Returns:
            {"action": str, "confidence": float, "reason": str, "risk_level": str}
        """
        # ── 加权得分 ──
        weighted_score = 0.0
        total_weight = 0.0
        for dim in ANALYSIS_DIMENSIONS:
            weight = DIMENSION_WEIGHTS.get(dim, 0.0)
            dim_score = dimensions.get(dim, {}).get("score", 0.0)
            weighted_score += weight * dim_score
            total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight

        # ── 尝试 LLM 综合决策 ──
        try:
            llm_decision = self._llm_synthesize(symbol, dimensions, history, skills_context, weighted_score)
            if llm_decision is not None:
                return llm_decision
        except Exception as e:
            logger.warning(f"[综合决策] {symbol} LLM 综合失败，回退规则引擎: {e}")

        # ── 规则引擎兜底 ──
        return self._rule_synthesize(dimensions, weighted_score)

    def _llm_synthesize(
        self,
        symbol: str,
        dimensions: dict,
        history: list,
        skills_context: str,
        weighted_score: float,
    ) -> Optional[dict]:
        """使用 LLM 进行综合决策"""
        # 构建维度摘要
        dim_summary = {}
        for dim in ANALYSIS_DIMENSIONS:
            d = dimensions.get(dim, {})
            dim_summary[dim] = {"score": d.get("score", 0.0), "reason": d.get("reason", "")}

        # 构建历史摘要（最近 5 条）
        history_summary = []
        for h in history[:5]:
            history_summary.append({
                "date": h.get("date", ""),
                "action": h.get("action", ""),
                "confidence": h.get("confidence", 0),
            })

        prompt = (
            f"Make a final investment decision for stock {symbol}.\n\n"
            f"Dimension scores (weighted average: {weighted_score:.3f}):\n"
            f"{json.dumps(dim_summary, ensure_ascii=False, indent=2)}\n\n"
        )

        if history_summary:
            prompt += f"Recent decision history:\n{json.dumps(history_summary, ensure_ascii=False, indent=2)}\n\n"

        if skills_context:
            prompt += f"{skills_context}\n\n"

        prompt += (
            "Based on all evidence, make a decision.\n\n"
            "Return a JSON object:\n"
            '{\n'
            '  "action": "buy" | "sell" | "hold",\n'
            '  "confidence": <float between 0 and 1>,\n'
            '  "reason": "<comprehensive explanation in Chinese, 2-4 sentences>",\n'
            '  "risk_level": "low" | "medium" | "high"\n'
            '}'
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior investment analyst making a final decision. "
                    "Be balanced and consider counter-arguments. "
                    "Confidence should reflect the agreement across dimensions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        result = self.llm.chat_json(messages, temperature=0.2)
        if "error" in result:
            return None

        action = result.get("action", "hold")
        if action not in ("buy", "sell", "hold"):
            action = "hold"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        risk_level = result.get("risk_level", "medium")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        reason = str(result.get("reason", "LLM 综合决策"))

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "reason": reason,
            "risk_level": risk_level,
        }

    def _rule_synthesize(self, dimensions: dict, weighted_score: float) -> dict:
        """规则引擎综合决策"""
        # 根据加权得分决定行动
        if weighted_score > 0.2:
            action = "buy"
        elif weighted_score < -0.2:
            action = "sell"
        else:
            action = "hold"

        # 信心度 = 加权得分的绝对值映射到 0~1，加上维度一致性调整
        base_confidence = abs(weighted_score)

        # 检查维度一致性：所有维度得分方向是否一致
        scores = [dimensions.get(dim, {}).get("score", 0.0) for dim in ANALYSIS_DIMENSIONS]
        positive_count = sum(1 for s in scores if s > 0)
        negative_count = sum(1 for s in scores if s < 0)

        # 一致性加成：方向越一致，信心越高
        if action == "buy":
            consistency_bonus = positive_count / len(scores) * 0.2
        elif action == "sell":
            consistency_bonus = negative_count / len(scores) * 0.2
        else:
            consistency_bonus = 0.0

        confidence = min(1.0, base_confidence + consistency_bonus + 0.3)

        # 风险等级
        if confidence > 0.7 and abs(weighted_score) > 0.4:
            risk_level = "low"
        elif confidence < 0.4 or abs(weighted_score) < 0.1:
            risk_level = "high"
        else:
            risk_level = "medium"

        # 生成原因
        dim_reasons = []
        for dim in ANALYSIS_DIMENSIONS:
            d = dimensions.get(dim, {})
            dim_score = d.get("score", 0.0)
            dim_reason = d.get("reason", "")
            if dim_score != 0.0 and dim_reason:
                dim_reasons.append(f"{dim}({dim_score:+.2f}): {dim_reason}")

        reason = f"加权得分{weighted_score:+.3f}，" + "；".join(dim_reasons[:3])

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "reason": reason,
            "risk_level": risk_level,
        }

    # ═══════════════════════════════════════════════════════════
    # 风控校验
    # ═══════════════════════════════════════════════════════════

    def _check_risk(self, symbol: str, decision: dict, consecutive_errors: int) -> dict:
        """
        风控校验：连续错误过多时强制 hold，调整信心度

        Args:
            symbol: 股票代码
            decision: 综合决策结果
            consecutive_errors: 连续错误次数

        Returns:
            校验后的决策字典
        """
        action = decision.get("action", "hold")
        confidence = decision.get("confidence", 0.5)
        reason = decision.get("reason", "")
        risk_level = decision.get("risk_level", "medium")

        # ── 规则一：连续错误过多，强制 hold ──
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.warning(
                f"[风控] {symbol} 连续错误 {consecutive_errors} 次，"
                f"强制降级为 hold"
            )
            action = "hold"
            confidence = min(confidence, 0.3)
            reason = f"[风控] 连续{consecutive_errors}次判断失误，自动降级为观望。原判断: {reason}"
            risk_level = "high"

        # ── 规则二：信心度过低，强制 hold ──
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                f"[风控] {symbol} 信心度 {confidence:.2f} 低于阈值 "
                f"{CONFIDENCE_THRESHOLD}，降级为 hold"
            )
            action = "hold"
            reason = f"[风控] 信心度不足({confidence:.2f}<{CONFIDENCE_THRESHOLD})，建议观望。{reason}"
            risk_level = "high"

        # ── 规则三：高风险时降低信心度 ──
        if risk_level == "high" and action != "hold":
            confidence *= 0.7
            reason = f"[风控] 高风险环境，信心度下调。{reason}"

        # ── 规则四：连续错误递减信心 ──
        if consecutive_errors > 0 and consecutive_errors < MAX_CONSECUTIVE_ERRORS:
            penalty = 0.1 * consecutive_errors
            confidence = max(0.0, confidence - penalty)
            if penalty > 0:
                reason = f"[风控] 近期{consecutive_errors}次失误，信心度下调{penalty:.1f}。{reason}"

        confidence = max(0.0, min(1.0, confidence))

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "reason": reason,
            "risk_level": risk_level,
        }

    # ═══════════════════════════════════════════════════════════
    # 目标价估算
    # ═══════════════════════════════════════════════════════════

    def _estimate_price_target(self, stock_info: dict, decision: dict) -> Optional[float]:
        """
        基于当前价格和决策方向粗略估算目标价

        Args:
            stock_info: 股票基本信息
            decision: 决策结果

        Returns:
            目标价格，数据不足时返回 None
        """
        current_price = stock_info.get("price", 0)
        if not current_price or current_price <= 0:
            return None

        action = decision.get("action", "hold")
        confidence = decision.get("confidence", 0.5)

        # 根据决策方向和信心度估算目标价变动幅度
        if action == "buy":
            # 买入信号：预期上涨 5%~20%，信心越高预期越大
            target_pct = 0.05 + confidence * 0.15
        elif action == "sell":
            # 卖出信号：预期下跌 5%~20%
            target_pct = -(0.05 + confidence * 0.15)
        else:
            # 观望：预期变动不大
            target_pct = 0.0

        target_price = current_price * (1 + target_pct)
        return round(target_price, 2)
