"""
llm_client.py
统一 LLM 调用客户端，支持 DeepSeek / 混元 / 任意 OpenAI 兼容接口

环境变量优先级：
  API Key  : DEEPSEEK_API_KEY > HUNYUAN_API_KEY
  URL      : LLM_API_URL（默认 DeepSeek）
  Model    : LLM_MODEL（默认 deepseek-chat）

切换示例：
  # 用 DeepSeek
  export DEEPSEEK_API_KEY="..."
  export LLM_API_URL="https://api.deepseek.com/v1/chat/completions"
  export LLM_MODEL="deepseek-chat"

  # 切换混元
  export HUNYUAN_API_KEY="..."
  export LLM_API_URL="https://api.hunyuan.cloud.tencent.com/v1/chat/completions"
  export LLM_MODEL="hunyuan-standard"
"""

import os
import time
import requests

# ── 配置（全部从环境变量读，代码零改动）──────────────────────────

API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("HUNYUAN_API_KEY")
    or ""
)

CHAT_URL = os.environ.get(
    "LLM_API_URL",
    "https://api.deepseek.com/v1/chat/completions",
)

CHAT_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")


def _detect_provider() -> str:
    if "hunyuan" in CHAT_URL:
        return "hunyuan"
    if "deepseek" in CHAT_URL:
        return "deepseek"
    return "openai-compat"


# ── 核心调用 ─────────────────────────────────────────────────────

def call_llm(
    system: str,
    user: str,
    max_tokens: int = 1000,
    retry: int = 3,
    timeout: int = 60,
) -> str:
    """
    调用 LLM，返回文本内容。失败时重试，最终失败返回空字符串。
    """
    if not API_KEY:
        raise ValueError(
            "未设置 API Key。请设置 HUNYUAN_API_KEY 或 DEEPSEEK_API_KEY"
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    provider = _detect_provider()

    for attempt in range(retry):
        try:
            resp = requests.post(
                CHAT_URL, headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"[llm/{provider}] ⚠️  HTTP {status}，第{attempt+1}次失败")
            # 401/403 不重试，直接报错
            if status in (401, 403):
                raise ValueError(f"API 鉴权失败 ({status})，请检查 API Key") from e

        except Exception as e:
            print(f"[llm/{provider}] ⚠️  第{attempt+1}次失败: {e}")

        if attempt < retry - 1:
            wait = 2 ** attempt
            print(f"[llm/{provider}] 等待 {wait}s 后重试…")
            time.sleep(wait)

    print(f"[llm/{provider}] ❌ 全部 {retry} 次均失败，返回空")
    return ""


def current_config() -> dict:
    """返回当前配置摘要，方便调试"""
    return {
        "provider": _detect_provider(),
        "url":      CHAT_URL,
        "model":    CHAT_MODEL,
        "key_set":  bool(API_KEY),
        "key_source": (
            "HUNYUAN_API_KEY" if os.environ.get("HUNYUAN_API_KEY")
            else "DEEPSEEK_API_KEY" if os.environ.get("DEEPSEEK_API_KEY")
            else "未设置"
        ),
    }


if __name__ == "__main__":
    cfg = current_config()
    print("当前 LLM 配置：")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
