"""对照 Chat2API PR #147 的 bug 类：工具标记在「没有可用调用」时泄漏成正文。

PR #147 的根因是清理逻辑只在 toolCalls.length > 0 时运行，导致未知工具名的
XML 原样留在 content 里发给客户端。这里检验我们是否也有同样的问题。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.protocol import openai_tool_bridge as b  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "read",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
]

CASES = {
    "unknown XML tool": '<tool_calls><tool_call><tool_name>run_shell</tool_name><parameters><command>ls</command></parameters></tool_call></tool_calls>',
    "unknown XML + prose": '<tool_calls><tool_call><tool_name>run_shell</tool_name><parameters><command>ls</command></parameters></tool_call></tool_calls>\nDone already.',
    "unknown JSON env": '{"tool_calls":[{"name":"run_shell","arguments":{"command":"ls"}}]}',
    "valid XML": '<tool_calls><tool_call><tool_name>read</tool_name><parameters><file_path>x</file_path></parameters></tool_call></tool_calls>',
    "partial XML leftover": 'Checking.\n<tool_call><tool_name>read</tool_name><parameters><file_path>a',
}

for label, raw in CASES.items():
    text, calls = b.parse_reply(raw, TOOLS)
    names = [c["function"]["name"] for c in calls]
    print(f"{label:24s} calls={names} text={text[:80]!r}")

print()
print("unknown_tool_names(unknown XML) =", b.unknown_tool_names(CASES["unknown XML tool"], TOOLS))
print("looks_like_tool_envelope(unknown XML) =", b.looks_like_tool_envelope(CASES["unknown XML tool"]))
print("clean_text(unknown XML) =", repr(b.clean_text(CASES["unknown XML tool"])[:80]))
