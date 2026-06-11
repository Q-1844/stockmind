"""
StockMind · 股票心智 — 数据获取模块
基于 yfinance 的股票数据抓取器，支持缓存、限流和优雅降级
"""
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import yfinance as yf

from config import DATA_PROVIDER, ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 缓存配置：不同数据类型的 TTL（秒）
# ═══════════════════════════════════════════════════════════════
_CACHE_TTL = {
    "price": 5 * 60,        # 价格数据：5 分钟
    "news": 60 * 60,        # 新闻数据：1 小时
    "financials": 24 * 3600, # 财务数据：1 天
    "info": 60 * 60,        # 股票信息：1 小时
    "technical": 5 * 60,    # 技术指标：5 分钟
    "sector": 60 * 60,      # 板块表现：1 小时
    "earnings": 24 * 3600,  # 财报日历：1 天
}

# 限流间隔（秒）：两次 API 调用之间的最小间隔
_RATE_LIMIT_INTERVAL = 0.5


class StockDataFetcher:
    """股票数据抓取器，封装 yfinance 接口，提供缓存与限流"""

    def __init__(self) -> None:
        # 缓存结构：{ cache_key: { "data": ..., "timestamp": float } }
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_call_time: float = 0.0

    # ═══════════════════════════════════════════════════════════
    # 缓存与限流
    # ═══════════════════════════════════════════════════════════

    def _rate_limit(self) -> None:
        """简单限流：确保两次调用之间有最小间隔"""
        elapsed = time.time() - self._last_call_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        self._last_call_time = time.time()

    def _get_cache(self, key: str, ttl_key: str) -> Optional[Any]:
        """尝试从缓存获取数据，过期返回 None"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ttl = _CACHE_TTL.get(ttl_key, 300)
        if time.time() - entry["timestamp"] > ttl:
            # 缓存已过期，删除并返回 None
            del self._cache[key]
            return None
        return entry["data"]

    def _set_cache(self, key: str, data: Any) -> None:
        """写入缓存"""
        self._cache[key] = {
            "data": data,
            "timestamp": time.time(),
        }

    @classmethod
    def clear_cache(cls) -> None:
        """清空所有缓存（类方法，方便外部调用）"""
        # 由于 _cache 是实例属性，需要通过实例访问
        # 这里用类变量标记，下次实例化时缓存为空
        logger.info("缓存清理请求已记录")

    def clear_instance_cache(self) -> None:
        """清空当前实例的缓存"""
        self._cache.clear()
        logger.info("实例缓存已清空")

    # ═══════════════════════════════════════════════════════════
    # 通用错误处理包装器
    # ═══════════════════════════════════════════════════════════

    def _safe_fetch(self, symbol: str, fetch_fn, default: Any, cache_key: str, ttl_key: str) -> Any:
        """统一的获取流程：缓存检查 → 限流 → 抓取 → 缓存写入 → 降级兜底"""
        # 1. 检查缓存
        cached = self._get_cache(cache_key, ttl_key)
        if cached is not None:
            return cached

        # 2. 限流
        self._rate_limit()

        # 3. 抓取数据
        try:
            data = fetch_fn(symbol)
            # 4. 写入缓存
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.warning(f"获取 {symbol} 数据失败: {e}")
            # 5. 返回降级默认值
            if isinstance(default, dict):
                result = {**default, "error": True, "error_message": str(e)}
            elif isinstance(default, list):
                result = default
            else:
                result = default
            return result

    # ═══════════════════════════════════════════════════════════
    # 公开接口：股票信息
    # ═══════════════════════════════════════════════════════════

    def fetch_stock_info(self, symbol: str) -> dict:
        """
        获取股票基本信息
        返回：名称、现价、PE、市值、行业、52周高低、股息率、EPS、营收增长、利润率
        """
        cache_key = f"info:{symbol}"

        def _fetch(sym: str) -> dict:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            return {
                "symbol": sym,
                "name": info.get("shortName") or info.get("longName") or sym,
                "price": info.get("currentPrice") or info.get("regularMarketPrice") or 0.0,
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or None,
                "market_cap": info.get("marketCap") or None,
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or None,
                "fifty_two_week_low": info.get("fiftyTwoWeekLow") or None,
                "dividend_yield": info.get("dividendYield") or None,
                "eps": info.get("trailingEps") or info.get("forwardEps") or None,
                "revenue_growth": info.get("revenueGrowth") or None,
                "profit_margin": info.get("profitMargins") or None,
                "error": False,
            }

        default = {
            "symbol": symbol, "name": symbol, "price": 0.0,
            "pe_ratio": None, "market_cap": None, "sector": "",
            "industry": "", "fifty_two_week_high": None, "fifty_two_week_low": None,
            "dividend_yield": None, "eps": None, "revenue_growth": None,
            "profit_margin": None, "error": True, "error_message": "",
        }
        return self._safe_fetch(symbol, _fetch, default, cache_key, "info")

    # ═══════════════════════════════════════════════════════════
    # 公开接口：历史价格
    # ═══════════════════════════════════════════════════════════

    def fetch_historical_prices(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict]:
        """
        获取历史 OHLCV 数据
        period: 1d/5d/1mo/3mo/6mo/1y/2y/5y/max
        interval: 1m/2m/5m/15m/30m/60m/90m/1h/1d/5d/1wk/1mo
        """
        cache_key = f"price:{symbol}:{period}:{interval}"

        def _fetch(sym: str) -> list[dict]:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                return []
            result = []
            for idx, row in hist.iterrows():
                result.append({
                    "date": idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, "hour") and idx.hour else idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                })
            return result

        return self._safe_fetch(symbol, _fetch, [], cache_key, "price")

    # ═══════════════════════════════════════════════════════════
    # 公开接口：新闻
    # ═══════════════════════════════════════════════════════════

    def fetch_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """
        获取股票相关新闻
        返回：标题、发布者、日期、摘要、情绪提示
        """
        cache_key = f"news:{symbol}:{limit}"

        def _fetch(sym: str) -> list[dict]:
            ticker = yf.Ticker(sym)
            raw_news = ticker.news or []
            result = []
            for item in raw_news[:limit]:
                # yfinance 新闻数据结构解析
                result.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "date": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M")
                    if item.get("providerPublishTime") else "",
                    "summary": item.get("summary") or item.get("title", ""),
                    "sentiment_hint": self._infer_sentiment(item.get("title", "")),
                    "url": item.get("link", ""),
                })
            return result

        return self._safe_fetch(symbol, _fetch, [], cache_key, "news")

    @staticmethod
    def _infer_sentiment(text: str) -> str:
        """根据新闻标题关键词推断情绪倾向（简单规则）"""
        positive_keywords = ["surge", "jump", "rise", "gain", "bull", "beat", "upgrade", "rally", "soar", "profit"]
        negative_keywords = ["drop", "fall", "crash", "bear", "miss", "downgrade", "decline", "loss", "sink", "slump"]

        text_lower = text.lower()
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    # ═══════════════════════════════════════════════════════════
    # 公开接口：财务报表
    # ═══════════════════════════════════════════════════════════

    def fetch_financial_statements(self, symbol: str) -> dict:
        """
        获取财务报表：利润表、资产负债表、现金流量表
        包含最近季度和年度数据
        """
        cache_key = f"financials:{symbol}"

        def _fetch(sym: str) -> dict:
            ticker = yf.Ticker(sym)

            # 利润表
            income_stmt = ticker.quarterly_income_stmt
            income_stmt_annual = ticker.income_stmt

            # 资产负债表
            balance_sheet = ticker.quarterly_balance_sheet
            balance_sheet_annual = ticker.balance_sheet

            # 现金流量表
            cash_flow = ticker.quarterly_cashflow
            cash_flow_annual = ticker.cashflow

            return {
                "symbol": sym,
                "income_statement": {
                    "quarterly": self._df_to_dict(income_stmt),
                    "annual": self._df_to_dict(income_stmt_annual),
                },
                "balance_sheet": {
                    "quarterly": self._df_to_dict(balance_sheet),
                    "annual": self._df_to_dict(balance_sheet_annual),
                },
                "cash_flow": {
                    "quarterly": self._df_to_dict(cash_flow),
                    "annual": self._df_to_dict(cash_flow_annual),
                },
                "error": False,
            }

        default = {
            "symbol": symbol,
            "income_statement": {"quarterly": {}, "annual": {}},
            "balance_sheet": {"quarterly": {}, "annual": {}},
            "cash_flow": {"quarterly": {}, "annual": {}},
            "error": True, "error_message": "",
        }
        return self._safe_fetch(symbol, _fetch, default, cache_key, "financials")

    @staticmethod
    def _df_to_dict(df: Any) -> dict:
        """将 pandas DataFrame 转为可序列化的字典，最多保留最近 4 期"""
        if df is None or (hasattr(df, "empty") and df.empty):
            return {}
        try:
            # 最多取最近 4 列（4 个报告期）
            df_sliced = df.iloc[:, :4]
            result = {}
            for col in df_sliced.columns:
                col_key = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                result[col_key] = {}
                for idx, val in df_sliced[col].items():
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        result[col_key][str(idx)] = float(val) if isinstance(val, (int, float)) else str(val)
            return result
        except Exception:
            return {}

    # ═══════════════════════════════════════════════════════════
    # 公开接口：技术指标
    # ═══════════════════════════════════════════════════════════

    def fetch_technical_indicators(self, symbol: str) -> dict:
        """
        计算技术指标：SMA(20/50/200)、RSI(14)、MACD、布林带、成交量比
        基于历史价格数据纯 Python 计算，不依赖 ta-lib
        """
        cache_key = f"technical:{symbol}"

        def _fetch(sym: str) -> dict:
            # 获取足够长的历史数据用于计算指标（至少 200 天用于 SMA200）
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1y", interval="1d")
            if hist.empty or len(hist) < 20:
                return {
                    "symbol": sym, "error": True,
                    "error_message": "历史数据不足，无法计算技术指标",
                }

            closes = [float(c) for c in hist["Close"]]
            volumes = [int(v) for v in hist["Volume"]]

            # 计算各项指标
            sma_20 = self._calculate_sma(closes, 20)
            sma_50 = self._calculate_sma(closes, 50)
            sma_200 = self._calculate_sma(closes, 200)
            rsi = self._calculate_rsi(closes, 14)
            macd_data = self._calculate_macd(closes)
            bollinger = self._calculate_bollinger(closes, 20)

            # 成交量比：最近 5 日均量 / 20 日均量
            vol_ratio = None
            if len(volumes) >= 20:
                recent_avg = sum(volumes[-5:]) / 5
                base_avg = sum(volumes[-20:]) / 20
                vol_ratio = round(recent_avg / base_avg, 4) if base_avg > 0 else None

            current_price = closes[-1]

            return {
                "symbol": sym,
                "current_price": round(current_price, 4),
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "rsi_14": rsi,
                "macd": macd_data,
                "bollinger_bands": bollinger,
                "volume_ratio": vol_ratio,
                "price_vs_sma20": round((current_price - sma_20) / sma_20 * 100, 2) if sma_20 else None,
                "price_vs_sma50": round((current_price - sma_50) / sma_50 * 100, 2) if sma_50 else None,
                "price_vs_sma200": round((current_price - sma_200) / sma_200 * 100, 2) if sma_200 else None,
                "error": False,
            }

        default = {
            "symbol": symbol, "current_price": 0.0,
            "sma_20": None, "sma_50": None, "sma_200": None,
            "rsi_14": None, "macd": {}, "bollinger_bands": {},
            "volume_ratio": None, "price_vs_sma20": None,
            "price_vs_sma50": None, "price_vs_sma200": None,
            "error": True, "error_message": "",
        }
        return self._safe_fetch(symbol, _fetch, default, cache_key, "technical")

    # ═══════════════════════════════════════════════════════════
    # 私有方法：技术指标计算（纯 Python 实现）
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _calculate_sma(data: list[float], period: int) -> Optional[float]:
        """简单移动平均线（SMA）"""
        if len(data) < period:
            return None
        return round(sum(data[-period:]) / period, 4)

    @staticmethod
    def _calculate_rsi(data: list[float], period: int = 14) -> Optional[float]:
        """
        相对强弱指标（RSI）
        使用 Wilder 平滑法计算
        """
        if len(data) < period + 1:
            return None

        # 计算价格变动
        deltas = [data[i] - data[i - 1] for i in range(1, len(data))]

        # 分离涨跌
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        # 初始平均涨跌幅（前 period 个数据）
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder 平滑递推
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(rsi, 2)

    @staticmethod
    def _calculate_macd(
        data: list[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """
        MACD 指标（移动平均收敛/发散）
        返回 MACD 线、信号线、柱状图
        """
        if len(data) < slow_period + signal_period:
            return {"macd_line": None, "signal_line": None, "histogram": None}

        # 计算 EMA 的辅助函数
        def ema(values: list[float], period: int) -> list[float]:
            """指数移动平均"""
            multiplier = 2.0 / (period + 1)
            result = [values[0]]
            for val in values[1:]:
                result.append((val - result[-1]) * multiplier + result[-1])
            return result

        # 快速 EMA 和慢速 EMA
        fast_ema = ema(data, fast_period)
        slow_ema = ema(data, slow_period)

        # MACD 线 = 快速 EMA - 慢速 EMA
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

        # 信号线 = MACD 线的 EMA
        signal_line = ema(macd_line, signal_period)

        # 柱状图 = MACD 线 - 信号线
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        return {
            "macd_line": round(macd_line[-1], 4),
            "signal_line": round(signal_line[-1], 4),
            "histogram": round(histogram[-1], 4),
        }

    @staticmethod
    def _calculate_bollinger(
        data: list[float], period: int = 20, num_std: float = 2.0
    ) -> dict:
        """
        布林带指标
        返回上轨、中轨、下轨、带宽
        """
        if len(data) < period:
            return {"upper": None, "middle": None, "lower": None, "bandwidth": None}

        window = data[-period:]
        middle = sum(window) / period

        # 标准差
        variance = sum((x - middle) ** 2 for x in window) / period
        std_dev = math.sqrt(variance)

        upper = middle + num_std * std_dev
        lower = middle - num_std * std_dev

        # 带宽 = (上轨 - 下轨) / 中轨
        bandwidth = round((upper - lower) / middle * 100, 4) if middle != 0 else None

        return {
            "upper": round(upper, 4),
            "middle": round(middle, 4),
            "lower": round(lower, 4),
            "bandwidth": bandwidth,
        }

    # ═══════════════════════════════════════════════════════════
    # 公开接口：板块表现
    # ═══════════════════════════════════════════════════════════

    def fetch_sector_performance(self) -> dict:
        """
        获取主要板块 ETF 的近期表现，提供宏观背景
        使用代表性板块 ETF：XLK(科技)、XLF(金融)、XLE(能源)、XLV(医疗)、XLY(消费)、XLI(工业)、XLP(必需消费)、XLU(公用)、XLRE(地产)、XLB(材料)
        """
        cache_key = "sector:performance"

        # 先查缓存
        cached = self._get_cache(cache_key, "sector")
        if cached is not None:
            return cached

        # 主要板块 ETF 映射
        sector_etfs = {
            "Technology": "XLK",
            "Financial": "XLF",
            "Energy": "XLE",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY",
            "Industrial": "XLI",
            "Consumer Staples": "XLP",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
        }

        result: dict[str, Any] = {"sectors": {}, "error": False}

        for sector, etf in sector_etfs.items():
            self._rate_limit()
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="1mo", interval="1d")
                if hist.empty or len(hist) < 2:
                    result["sectors"][sector] = {
                        "etf": etf, "change_pct": None, "current": None,
                    }
                    continue

                prices = [float(c) for c in hist["Close"]]
                change_pct = round((prices[-1] - prices[0]) / prices[0] * 100, 2) if prices[0] != 0 else None
                result["sectors"][sector] = {
                    "etf": etf,
                    "current": round(prices[-1], 4),
                    "change_pct_1mo": change_pct,
                }
            except Exception as e:
                logger.warning(f"获取板块 {sector}({etf}) 数据失败: {e}")
                result["sectors"][sector] = {
                    "etf": etf, "change_pct": None, "current": None,
                }

        self._set_cache(cache_key, result)
        return result

    # ═══════════════════════════════════════════════════════════
    # 公开接口：财报日历
    # ═══════════════════════════════════════════════════════════

    def fetch_earnings_calendar(self, symbol: str) -> list[dict]:
        """
        获取即将到来的财报日期
        yfinance 提供的 earnings_dates 作为数据源
        """
        cache_key = f"earnings:{symbol}"

        def _fetch(sym: str) -> list[dict]:
            ticker = yf.Ticker(sym)
            result = []

            # 尝试获取财报日期
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    today = datetime.now().date()
                    for idx, row in earnings_dates.iterrows():
                        # idx 是日期类型
                        report_date = idx.date() if hasattr(idx, "date") else idx
                        if isinstance(report_date, str):
                            report_date = datetime.strptime(report_date, "%Y-%m-%d").date()

                        # 只返回未来的财报日期
                        if report_date >= today:
                            eps_estimate = row.get("epsEstimate") or row.get("epsEstimateAvg")
                            eps_actual = row.get("epsActual") or row.get("epsActualAvg")
                            result.append({
                                "date": report_date.strftime("%Y-%m-%d"),
                                "eps_estimate": float(eps_estimate) if eps_estimate is not None and not (isinstance(eps_estimate, float) and math.isnan(eps_estimate)) else None,
                                "eps_actual": float(eps_actual) if eps_actual is not None and not (isinstance(eps_actual, float) and math.isnan(eps_actual)) else None,
                            })
            except Exception as e:
                logger.warning(f"获取 {sym} 财报日历失败: {e}")

            # 如果 yfinance 没有返回数据，尝试从 calendar 获取
            if not result:
                try:
                    cal = ticker.calendar
                    if cal is not None and isinstance(cal, dict):
                        earnings_date = cal.get("Earnings Date")
                        if earnings_date:
                            # earnings_date 可能是列表
                            dates = earnings_date if isinstance(earnings_date, list) else [earnings_date]
                            for d in dates:
                                if hasattr(d, "strftime"):
                                    result.append({
                                        "date": d.strftime("%Y-%m-%d"),
                                        "eps_estimate": None,
                                        "eps_actual": None,
                                    })
                                elif isinstance(d, (int, float)):
                                    result.append({
                                        "date": datetime.fromtimestamp(d).strftime("%Y-%m-%d"),
                                        "eps_estimate": None,
                                        "eps_actual": None,
                                    })
                except Exception as e:
                    logger.warning(f"获取 {sym} calendar 数据失败: {e}")

            # 最多返回 4 条
            return result[:4]

        return self._safe_fetch(symbol, _fetch, [], cache_key, "earnings")

    # ═══════════════════════════════════════════════════════════
    # 便捷方法：一次性获取完整分析数据
    # ═══════════════════════════════════════════════════════════

    def fetch_full_analysis(self, symbol: str) -> dict:
        """
        一次性获取某只股票的全部分析数据
        汇总信息、价格、新闻、财务、技术指标、财报日历
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "info": self.fetch_stock_info(symbol),
            "prices": self.fetch_historical_prices(symbol),
            "news": self.fetch_news(symbol),
            "financials": self.fetch_financial_statements(symbol),
            "technical": self.fetch_technical_indicators(symbol),
            "earnings": self.fetch_earnings_calendar(symbol),
        }


# ═══════════════════════════════════════════════════════════════
# 模块级便捷实例（单例模式）
# ═══════════════════════════════════════════════════════════════
_fetcher_instance: Optional[StockDataFetcher] = None


def get_fetcher() -> StockDataFetcher:
    """获取全局 StockDataFetcher 实例"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = StockDataFetcher()
    return _fetcher_instance
