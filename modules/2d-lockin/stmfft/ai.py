from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import json
import os
from pathlib import Path
import re
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models?input_modalities=text,image&output_modalities=text&sort=most-popular"
DEFAULT_MODEL = "~openai/gpt-sol-latest"


@dataclass
class AIConfig:
    provider: str = "openrouter"
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout_seconds: int = 120

    def public_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """你是 STM/STS Fourier 数据分析助手。你收到的是当前软件视图、采集元数据、预处理、FFT、Bragg 标定、mask 和 iFFT/局域复包络结果。

分析时必须遵守：
1. 先描述图中和数字中可以直接观察的事实，再给候选解释。
2. 明确区分 Bragg 峰、结构重构、缺陷 form factor/structure factor、QPI、电子有序和处理伪影。
3. q 的单位是 cycles/nm，周期是 1/|q|；除非输入另有说明，不要擅自加入 2π。
4. 检查裁剪、调平、逐行校正、窗函数、mask 半径、边界泄漏、Bragg 泄漏、漂移和 setpoint 串扰。
5. 普通形貌或 dI/dV Fourier 峰不能单独证明自旋、CDW、PDW 或具体原子结构。
6. 对局域振幅和相位，说明低振幅区相位不可靠；判断畴界前要求 mask 宽度和峰位敏感性检验。
7. 不虚构没有提供的温度、偏压、电流、材料或实验条件。缺失信息直接指出。
8. 用中文回答，结构为：直接观察、数值与处理检查、候选解释、当前不能证明什么、最优先的下一步。
"""


def load_key_from_env_file(path: str | Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^\s*OPENROUTER_API_KEY\s*=\s*([^\s#]+)\s*$", text)
    return match.group(1).strip().strip("\"'") if match else None


def resolve_openrouter_key(app_dir: str | Path, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    for name in (".env.local", ".env"):
        key = load_key_from_env_file(Path(app_dir) / name)
        if key:
            return key
    return None


def save_openrouter_key(path: str | Path, key: str) -> None:
    path = Path(path)
    key = key.strip()
    if not key:
        raise ValueError("OpenRouter API key 不能为空。")
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    line = f"OPENROUTER_API_KEY={key}"
    pattern = re.compile(r"(?m)^\s*OPENROUTER_API_KEY\s*=.*$")
    if pattern.search(existing):
        updated = pattern.sub(line, existing)
    else:
        separator = "" if not existing or existing.endswith("\n") else "\n"
        updated = existing + separator + line + "\n"
    path.write_text(updated, encoding="utf-8")


def load_config(path: str | Path) -> AIConfig:
    path = Path(path)
    if not path.exists():
        return AIConfig()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return AIConfig(
            provider="openrouter",
            endpoint=value.get("endpoint", DEFAULT_ENDPOINT),
            model=value.get("model", DEFAULT_MODEL),
            timeout_seconds=int(value.get("timeout_seconds", 120)),
        )
    except Exception:
        return AIConfig()


def save_config(path: str | Path, config: AIConfig) -> None:
    Path(path).write_text(json.dumps(config.public_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def build_openrouter_payload(
    config: AIConfig,
    analysis_summary: dict,
    image_png: bytes,
    user_question: str = "请分析当前 STM Fourier 结果，判断选中峰对应实空间分布是否可信，并提出下一步。",
) -> dict:
    encoded = base64.b64encode(image_png).decode("ascii")
    summary_text = json.dumps(analysis_summary, ensure_ascii=False, indent=2)
    user_text = f"用户问题：{user_question.strip()}\n\n当前分析参数与数值摘要：\n{summary_text}"
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2200,
    }


def extract_response_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        error = response.get("error")
        if isinstance(error, dict):
            raise RuntimeError(str(error.get("message") or error))
        raise RuntimeError("AI 返回中没有 choices。")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_vision_models(response: dict) -> list[dict]:
    output: list[dict] = []
    for item in response.get("data") or []:
        architecture = item.get("architecture") or {}
        inputs = set(architecture.get("input_modalities") or [])
        outputs = set(architecture.get("output_modalities") or [])
        model_id = str(item.get("id") or "").strip()
        if not model_id or "image" not in inputs or (outputs and "text" not in outputs):
            continue
        pricing = item.get("pricing") or {}
        output.append(
            {
                "id": model_id,
                "name": str(item.get("name") or model_id),
                "context_length": item.get("context_length"),
                "prompt_price": pricing.get("prompt"),
                "completion_price": pricing.get("completion"),
            }
        )
    return output


def list_openrouter_vision_models(api_key: str, timeout_seconds: int = 60) -> list[dict]:
    request = Request(
        MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "X-OpenRouter-Title": "STM Fourier Analyzer"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"获取模型列表失败（HTTP {exc.code}）：{body[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 OpenRouter：{exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouter 模型列表无法解析。") from exc
    return parse_vision_models(parsed)


def request_openrouter_analysis(config: AIConfig, api_key: str, payload: dict) -> tuple[str, dict]:
    request = Request(
        config.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "STM Fourier Analyzer",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(body)
            message = (details.get("error") or {}).get("message") or body
        except Exception:
            message = body or str(exc)
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 OpenRouter：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("OpenRouter 请求超时。") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouter 返回了无法解析的内容。") from exc
    return extract_response_text(parsed), parsed.get("usage") or {}
