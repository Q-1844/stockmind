"""
StockMind · 股票心智 — 统一 LLM 客户端
支持 MiniMax / OpenAI / DeepSeek 三家 API，基于 OpenAI 兼容 SDK
"""
import re
import json
import time
import logging
from typing import Optional

from openai import OpenAI

from config import LLM_PROVIDER, LLM_CONFIG

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 各供应商的默认单价（美元 / 千 token），用于估算费用
# ═══════════════════════════════════════════════════════════════
_PRICING = {
    "minimax":  {"input": 0.001, "output": 0.002},
    "openai":   {"input": 0.00015, "output": 0.0006},
    "deepseek": {"input": 0.00014, "output": 0.00028},
}


class LLMClient:
    """统一大模型客户端，封装 chat / chat_json / 重试 / 用量统计"""

    def __init__(self, provider: Optional[str] = None) -> None:
        # 确定供应商
        self.provider: str = (provider or LLM_PROVIDER).lower()
        if self.provider not in LLM_CONFIG:
            raise ValueError(
                f"不支持的 LLM 供应商: {self.provider}，"
                f"可选: {', '.join(LLM_CONFIG.keys())}"
            )

        # 读取对应配置
        cfg = LLM_CONFIG[self.provider]
        api_key: str = cfg["api_key"]
        if not api_key:
            raise ValueError(f"供应商 {self.provider} 的 API Key 未配置")

        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
        )
        self.model: str = cfg["model"]
        self.default_temperature: float = cfg["temperature"]
        self.default_max_tokens: int = cfg["max_tokens"]

        # 用量统计
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0

    # ═══════════════════════════════════════════════════════════
    # 核心方法：chat
    # ═══════════════════════════════════════════════════════════

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送 chat completion 请求，返回内容字符串

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 采样温度，默认使用配置值
            max_tokens: 最大输出 token 数，默认使用配置值

        Returns:
            模型输出的文本内容；出错时返回错误信息字符串
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
            )
        except Exception as e:
            error_msg = self._format_error(e)
            logger.error(f"LLM 请求失败 [{self.provider}]: {error_msg}")
            return f"[LLM 错误] {error_msg}"

        # 统计用量
        self._record_usage(response)

        content = response.choices[0].message.content
        return content if content else ""

    # ═══════════════════════════════════════════════════════════
    # 核心方法：chat_json
    # ═══════════════════════════════════════════════════════════

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> dict:
        """
        发送请求并期望 JSON 格式响应，解析后返回 dict

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 采样温度

        Returns:
            解析后的字典；解析失败返回 {"error": ...}
        """
        # 在系统提示中强调 JSON 输出
        json_messages = list(messages)
        has_system = any(m.get("role") == "system" for m in json_messages)
        json_hint = "请务必以纯 JSON 格式返回结果，不要包含 markdown 代码块标记或其他文字。"
        if has_system:
            for m in json_messages:
                if m.get("role") == "system":
                    m["content"] = m["content"] + "\n" + json_hint
                    break
        else:
            json_messages.insert(0, {"role": "system", "content": json_hint})

        # 尝试使用 response_format（部分供应商支持）
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=json_messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=self.default_max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 不支持 response_format 时回退到普通请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=json_messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=self.default_max_tokens,
            )

        self._record_usage(response)

        content = response.choices[0].message.content or ""
        return self._parse_json(content)

    # ═══════════════════════════════════════════════════════════
    # 核心方法：chat_with_retry
    # ═══════════════════════════════════════════════════════════

    def chat_with_retry(
        self,
        messages: list[dict],
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """
        带指数退避的重试 chat

        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            **kwargs: 传递给 chat() 的额外参数（temperature, max_tokens 等）

        Returns:
            模型输出文本；全部重试失败后返回错误信息
        """
        last_error = ""
        for attempt in range(max_retries):
            result = self.chat(messages, **kwargs)
            # chat 内部已捕获异常并返回 [LLM 错误] 前缀
            if not result.startswith("[LLM 错误]"):
                return result

            last_error = result
            # 指数退避：1s, 2s, 4s ...
            wait = 2 ** attempt
            logger.warning(f"第 {attempt + 1}/{max_retries} 次重试，等待 {wait}s ...")
            time.sleep(wait)

        logger.error(f"重试 {max_retries} 次后仍失败: {last_error}")
        return last_error

    # ═══════════════════════════════════════════════════════════
    # 工具方法：estimate_tokens
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        粗略估算 token 数量
        中文约 1 token / 1.5 字符 → len/3
        英文约 1 token / 4 字符 → len/4
        混合文本取折中
        """
        if not text:
            return 0
        # 统计中文字符比例
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        ratio = chinese_chars / len(text) if text else 0
        # 按中英比例加权：中文 /3，英文 /4
        avg_divisor = 3 * ratio + 4 * (1 - ratio)
        return max(1, int(len(text) / avg_divisor))

    # ═══════════════════════════════════════════════════════════
    # 工具方法：get_usage_stats
    # ═══════════════════════════════════════════════════════════

    def get_usage_stats(self) -> dict:
        """
        返回用量统计：总 token、总调用次数、估算费用
        """
        total_tokens = self.total_input_tokens + self.total_output_tokens
        pricing = _PRICING.get(self.provider, {"input": 0, "output": 0})
        estimated_cost = (
            self.total_input_tokens * pricing["input"]
            + self.total_output_tokens * pricing["output"]
        ) / 1000.0

        return {
            "provider": self.provider,
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_calls": self.total_calls,
            "estimated_cost_usd": round(estimated_cost, 6),
        }

    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════

    def _record_usage(self, response) -> None:
        """从响应中提取并累加 token 用量"""
        self.total_calls += 1
        if hasattr(response, "usage") and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0

    @staticmethod
    def _parse_json(content: str) -> dict:
        """从模型输出中提取 JSON，兼容 markdown 代码块包裹的情况"""
        # 去掉 markdown 代码块标记
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        cleaned = re.sub(r"\s*```", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}\n原始内容: {content[:200]}")
            return {"error": f"JSON 解析失败: {e}", "raw_content": content}

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """将异常格式化为可读的错误信息"""
        # 区分常见错误类型
        exc_name = type(exc).__name__
        msg = str(exc)

        # 速率限制
        if "rate_limit" in msg.lower() or "429" in msg:
            return f"速率限制（429）: {msg}"
        # 超时
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return f"请求超时: {msg}"
        # 认证
        if "auth" in msg.lower() or "401" in msg or "403" in msg:
            return f"认证失败（{exc_name}）: {msg}"
        # 服务器错误
        if "500" in msg or "502" in msg or "503" in msg:
            return f"服务端错误: {msg}"

        return f"{exc_name}: {msg}"


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════
_client_instance: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """获取全局 LLMClient 单例实例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
