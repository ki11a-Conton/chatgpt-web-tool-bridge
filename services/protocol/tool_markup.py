"""上游 ChatGPT 网页后端只有纯文本通道，工具调用只能用 XML 文本协议承载。

本模块是这套 XML 协议的唯一定义处：提示词构造 + 解析/剥离。
Anthropic 与 OpenAI 兼容层都从这里取，避免两份正则各写一份后漂移。
"""
from __future__ import annotations

import html
import json
import re

XML_TOOL_RULE = """Tool output adapter: when calling tools, output ONLY this XML and no prose/markdown:
<tool_calls><tool_call><tool_name>TOOL_NAME</tool_name><parameters><PARAM><![CDATA[value]]></PARAM></parameters></tool_call></tool_calls>"""

_TOOL_BLOCK_PATTERN = (
    r"(?is)<tool_call\b[^>]*>(.*?)</tool_call>"
    r"|<function_call\b[^>]*>(.*?)</function_call>"
    r"|<invoke\b[^>]*>(.*?)</invoke>"
)

_MARKUP_PATTERN = (
    r"(?is)<tool_calls\b[^>]*>.*?</tool_calls>"
    r"|<tool_call\b[^>]*>.*?</tool_call>"
    r"|<function_call\b[^>]*>.*?</function_call>"
    r"|<invoke\b[^>]*>.*?</invoke>"
)

# 未闭合的标记尾巴：模型写一半被截断（流被掐、输出超限）时，整段标记会原样
# 漏进正文发给客户端，客户端拿到的是半截 XML 而不是工具调用。从第一个标记
# 开始一律截掉——标记之后的内容本来也不该是正文。
_MARKUP_TAIL_PATTERN = (
    r"(?is)<tool_calls\b.*$"
    r"|<tool_call\b.*$"
    r"|<function_call\b.*$"
    r"|<invoke\b.*$"
)

_MARKER_RE = re.compile(r"(?is)<tool_calls\b|<tool_call\b|<function_call\b|<invoke\b")


def tool_meta(tool: dict[str, object]) -> tuple[str, str, object]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(tool.get("name") or fn.get("name") or "").strip()
    desc = str(tool.get("description") or fn.get("description") or "").strip()
    schema = tool.get("input_schema") or tool.get("parameters") or fn.get("input_schema") or fn.get("parameters") or {}
    return name, desc, schema


_TOOL_USE_RULES = """Tool use rules:
- If the user asks to list/read/search files, inspect project state, run a command, or answer from local code, you MUST call a suitable tool first. Do not say you cannot access files.
- To call tools, output ONLY XML and no prose/markdown:
<tool_calls><tool_call><tool_name>TOOL_NAME</tool_name><parameters><PARAM><![CDATA[value]]></PARAM></parameters></tool_call></tool_calls>
- Put parameters under <parameters> using the exact schema names."""


def build_tool_prompt(tools: object) -> str:
    if not isinstance(tools, list):
        return ""
    blocks = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name, desc, schema = tool_meta(tool)
        if name:
            blocks.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(schema, ensure_ascii=False)}")
    if not blocks:
        return ""
    return "Available tools:\n" + "\n".join(blocks) + "\n\n" + _TOOL_USE_RULES


def strip_tool_markup(text: str) -> str:
    """剥掉完整标记块，再截掉未闭合的标记尾巴。"""
    cleaned = re.sub(_MARKUP_PATTERN, "", text or "")
    return re.sub(_MARKUP_TAIL_PATTERN, "", cleaned).strip()


def streamable_text(text: str) -> str:
    text = text or ""
    match = _MARKER_RE.search(text)
    return text[:match.start()].rstrip() if match else text


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, object]]]:
    text = re.sub(r"(?is)```.*?```", "", text or "").strip()
    blocks = re.findall(_TOOL_BLOCK_PATTERN, text)
    result = []
    for block in (next((part for part in match if part), "") for match in blocks):
        name = xml_value(block, "tool_name") or xml_value(block, "name") or xml_value(block, "function")
        params = xml_value(block, "parameters") or xml_value(block, "input") or xml_value(block, "arguments") or "{}"
        if name:
            result.append((name, parse_tool_params(params)))
    return result


def xml_value(text: str, tag: str) -> str:
    match = re.search(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", text)
    if not match:
        return ""
    value = match.group(1).strip()
    cdata = re.fullmatch(r"(?is)<!\[CDATA\[(.*?)]]>", value)
    return html.unescape(cdata.group(1) if cdata else value).strip()


def parse_tool_params(raw: str) -> dict[str, object]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {m.group(1): parse_tool_value(m.group(2)) for m in re.finditer(r"(?is)<([\w.-]+)\b[^>]*>(.*?)</\1>", raw)}


def parse_tool_value(raw: str) -> object:
    value = xml_value(f"<x>{raw}</x>", "x")
    try:
        return json.loads(value)
    except Exception:
        return value
