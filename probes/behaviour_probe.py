"""Behavioural acceptance at DSH scale: how often does the model still refuse to call a tool?

The transport layer is clean (41/41 recent real turns succeeded), so what remains is
model behaviour. This runs a wider task matrix and classifies every outcome, including
the failure modes we have seen: own-sandbox echo, refusal, plan-only, web-artifact claim,
and unknown/extra tool arguments.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.protocol.openai_tool_bridge import (  # noqa: E402
    announces_action,
    claims_web_artifact,
    explains_instead_of_acting,
    is_refusal,
    tool_system_prompt,
)

REPO_ROOT = Path(os.environ.get("CHATGPT2API_ROOT", Path(__file__).resolve().parents[2]))
KEY = json.load(open(REPO_ROOT / "config.json", encoding="utf-8"))["auth-key"]
URL = "http://127.0.0.1:3000/v1/chat/completions"
SYSTEM = Path(__file__).with_name("dsh_system_prompt.txt").read_text(encoding="utf-8")
RUN = int(time.time())


def T(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


def arr(props, req=()):
    return {"type": "array", "items": {"type": "object", "properties": props, "required": list(req)}}


TOOLS = [
    T("pwsh", "Run a PowerShell command on the user's Windows machine and return stdout/stderr.", {"command": {"type": "string"}, "description": {"type": "string"}, "workdir": {"type": "string"}, "timeoutMs": {"type": "number"}, "run_in_background": {"type": "boolean"}}, ["command", "description"]),
    T("read", "Read a UTF-8 text file.", {"file_path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}}, ["file_path"]),
    T("write", "Create or fully replace a UTF-8 text file.", {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"]),
    T("edit", "Replace an exact string in a file.", {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, ["file_path", "old_string", "new_string"]),
    T("grep", "Search file contents with a regular expression.", {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, ["pattern"]),
    T("glob", "Find files by path pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    T("skill", "Load the full instructions for a skill.", {"name": {"type": "string"}}, ["name"]),
    T("todo_write", "Record and update a task list to plan multi-step work.", {"todos": arr({"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, ["content", "status"])}, ["todos"]),
    T("present", "Declare existing files as final deliverables for the user.", {"files": arr({"path": {"type": "string"}, "description": {"type": "string"}}, ["path"]), "description": {"type": "string"}}, ["files"]),
    T("ask_user_question", "Ask the user a concise question when confirmation is needed.", {"questions": arr({"id": {"type": "string"}, "question": {"type": "string"}, "header": {"type": "string"}, "multi_select": {"type": "boolean"}, "options": arr({"label": {"type": "string"}, "description": {"type": "string"}}, ["label"])}, ["id", "question"])}, ["questions"]),
    T("web_search", "Search the web for current information.", {"queries": {"type": "array", "items": {"type": "string"}}}, ["queries"]),
    T("web_fetch", "Fetch the content of a specific HTTP(S) URL.", {"url": {"type": "string"}}, ["url"]),
]

# label, request, tools that would be a correct first move
TASKS = [
    ("read file", "读取 D:\\chatgpt2api\\start.ps1 并说明它的作用", {"read"}),
    ("run git", "运行 git status --short", {"pwsh"}),
    ("list dir", "列出 D:\\chatgpt2api\\services\\protocol 下的 Python 文件", {"glob", "pwsh"}),
    ("write file", "把 hello 写入 D:\\chatgpt2api\\runtime\\behave_out.txt", {"write"}),
    ("make pptx", "帮我生成一个 pptx 文件，主题是长沙高端旅游，保存到 D:\\chatgpt2api\\runtime\\t.pptx，现在就动手", {"skill", "write", "pwsh", "todo_write"}),
    ("plan task", "这个任务分三步：先看仓库状态、再改代码、最后跑测试。先把计划记下来。", {"todo_write", "skill", "pwsh", "read"}),
    ("search code", "在工作区里找哪些文件定义了 CACHEABLE_TEXT_KEYS", {"grep", "glob", "pwsh"}),
    ("edit file", "把 D:\\chatgpt2api\\runtime\\behave_out.txt 里的 hello 改成 world", {"edit", "read", "write"}),
    ("run tests", "跑一下这个仓库的单元测试", {"pwsh"}),
    ("fetch url", "帮我看看 https://example.com 这个页面写了什么", {"web_fetch", "web_search"}),
]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"__err__": exc.code, "body": exc.read().decode("utf-8", "replace")[:150]}


def classify(out, allowed):
    """Return (verdict, detail). verdict is OK / MISS / BAD."""
    if "__err__" in out:
        return "MISS", f"HTTP {out['__err__']} {out['body'][:70]}"
    choice = out["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    text = str(message.get("content") or "")
    if not calls:
        if not text.strip():
            return "MISS", "empty reply"
        if is_refusal(text):
            return "MISS", f"refusal: {text[:80]}"
        if explains_instead_of_acting(text):
            return "MISS", f"plan-only: {text[:80]}"
        if claims_web_artifact(text):
            return "MISS", f"web-artifact: {text[:80]}"
        return "MISS", f"prose: {text[:80]}"
    schema = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    for call in calls:
        name = call["function"]["name"]
        if name not in schema:
            return "BAD", f"unknown tool {name!r}"
        args = json.loads(call["function"]["arguments"] or "{}")
        missing = [key for key in schema[name].get("required", []) if key not in args]
        extra = [key for key in args if key not in schema[name].get("properties", {})]
        if missing:
            return "BAD", f"{name} missing {missing}"
        if extra:
            return "BAD", f"{name} extra args {extra}"
    name = calls[0]["function"]["name"]
    if name not in allowed:
        return "BAD", f"unexpected tool {name!r}"
    return "OK", f"{name}({calls[0]['function']['arguments'][:70]})"


print(f"prompt chars: {len(tool_system_prompt(TOOLS))} (tools={len(TOOLS)})")
verdicts = Counter()
for index, (label, user, allowed) in enumerate(TASKS, start=1):
    out = post({"model": "gpt-6", "user": f"behave-{index}-{RUN}",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "tools": TOOLS})
    verdict, detail = classify(out, allowed)
    verdicts[verdict] += 1
    print(f"[{index:>2}] {label:<12} {verdict:<5} {detail}")

total = sum(verdicts.values())
print(f"\nOK={verdicts['OK']}/{total}  MISS={verdicts['MISS']}  BAD={verdicts['BAD']}")
