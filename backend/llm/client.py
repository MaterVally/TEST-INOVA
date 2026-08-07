"""
LLM and multimodal LLM client — async/sync wrappers with KV caching.
"""
import ast
import json
import re
from typing import Any

import numpy as np
from openai import AsyncOpenAI, OpenAI

from ..config.settings import (
    API_BASE,
    API_KEY,
    EMBED_MODEL,
    MM_API_BASE,
    MM_API_KEY,
    MM_MODEL_NAME,
    MODEL_NAME,
)
from ..storage.kv_storage import BaseKVStorage
from ..utils.base import compute_args_hash, logger, wrap_embedding_func_with_attrs

# ============================================================================
# Singleton client pool
# ============================================================================

_CLIENTS = {
    "text_sync": None, "text_async": None,
    "mm_sync": None,   "mm_async": None,
}


def _get_client(is_async: bool = False, is_multimodal: bool = False):
    key = f"{'mm' if is_multimodal else 'text'}_{'async' if is_async else 'sync'}"
    if _CLIENTS[key] is None:
        api_key  = MM_API_KEY  if is_multimodal else API_KEY
        base_url = MM_API_BASE if is_multimodal else API_BASE
        client_cls = AsyncOpenAI if is_async else OpenAI
        _CLIENTS[key] = client_cls(api_key=api_key, base_url=base_url)
    return _CLIENTS[key]


# ============================================================================
# Embedding
# ============================================================================

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts)


# ============================================================================
# Text LLM
# ============================================================================

async def model_if_cache(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    client = _get_client(is_async=True, is_multimodal=False)
    hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    args_hash = None
    if hashing_kv:
        args_hash = compute_args_hash(MODEL_NAME, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached:
            return cached["return"]

    response = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    if hashing_kv and args_hash:
        await hashing_kv.upsert({args_hash: {"return": content, "model": MODEL_NAME}})
        await hashing_kv.index_done_callback()

    return content


def get_llm_response(cur_prompt: str, system_content: str) -> str:
    client = _get_client(is_async=False, is_multimodal=False)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": cur_prompt},
        ],
        max_tokens=4096,
        frequency_penalty=0.3,
    )
    return completion.choices[0].message.content


# ============================================================================
# Multimodal LLM
# ============================================================================

async def multimodel_if_cache(
    user_prompt: str,
    img_base: str,
    system_prompt: str,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    client = _get_client(is_async=True, is_multimodal=True)
    hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)

    messages = []
    messages.extend(history_messages)
    messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
            {"type": "text", "text": user_prompt},
        ],
    })

    args_hash = None
    if hashing_kv:
        args_hash = compute_args_hash(MM_MODEL_NAME, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached:
            return cached["return"]

    response = await client.chat.completions.create(
        model=MM_MODEL_NAME, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    if hashing_kv and args_hash:
        await hashing_kv.upsert({args_hash: {"return": content, "model": MM_MODEL_NAME}})
        await hashing_kv.index_done_callback()

    return content


def get_mmllm_response(cur_prompt: str, system_content: str, img_base: str) -> str:
    client = _get_client(is_async=False, is_multimodal=True)
    completion = client.chat.completions.create(
        model=MM_MODEL_NAME,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
                {"type": "text", "text": cur_prompt},
            ]},
        ],
        max_tokens=4096,
    )
    return completion.choices[0].message.content


# ============================================================================
# JSON helpers
# ============================================================================

def normalize_to_json(output: str) -> dict | None:
    output = output.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", output, re.DOTALL)
    if match:
        output = match.group(1)
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        logger.debug(f"未找到 JSON 对象: {output[:100]}...")
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    try:
        repaired = re.sub(r"\bTrue\b", "true", json_str)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError) as e:
        logger.debug(f"JSON解码失败: {e}")
        return None


def normalize_to_json_list(output: str) -> list[Any]:
    cleaned = output.replace('\\"', '"').strip()
    match = re.search(r"\[\s*(\{.*?\})*?\s*]", cleaned, re.DOTALL)
    if not match:
        logger.warning("未找到有效的JSON列表片段")
        return []
    json_str = match.group(0)
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}$", "}", json_str)
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        logger.warning("完整列表解析失败，尝试逐项解析...")
    items = []
    for item_match in re.finditer(r"\{.*?\}", json_str, re.DOTALL):
        try:
            items.append(json.loads(item_match.group(0)))
        except json.JSONDecodeError:
            continue
    return items
