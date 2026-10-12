"""Acceptance at DSH scale: 19 tools + the real captured DSH system prompt.

CatGPT-Gateway notes prompt-based tool calling is reliable for 1-7 tools and that
complex schemas need retries. DSH declares 30+, so this checks that the compact
listing still produces dispatchable calls at that scale.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.protocol.openai_tool_bridge import tool_system_prompt  # noqa: E402

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
    T("spawn_teammate", "Create one named, durable teammate.", {"name": {"type": "string"}, "description": {"type": "string"}, "prompt": {"type": "string"}, "context": {"type": "string", "enum": ["fresh", "fork"]}}, ["name", "description", "prompt"]),
    T("send_message", "Send one durable message to another Team member.", {"target": {"type": "string"}, "message": {"type": "string"}}, ["target", "message"]),
    T("team_task_create", "Create one unowned pending task on the shared Team task board.", {"subject": {"type": "string"}, "description": {"type": "string"}, "write_scopes": {"type": "array", "items": {"type": "string"}}, "blocked_by": {"type": "array", "items": {"type": "string"}}}, ["subject", "description"]),
    T("workflow", "Run a JavaScript workflow script that orchestrates subagents at scale.", {"script": {"type": "string"}, "meta": {"type": "object"}, "args": {"type": "object"}}, ["script", "meta"]),
    T("subagent", "Delegate a self-contained task to a subagent.", {"description": {"type": "string"}, "prompt": {"type": "string"}, "run_in_background": {"type": "boolean"}}, ["description", "prompt"]),
    T("web_search", "Search the web for current information.", {"queries": {"type": "array", "items": {"type": "string"}}}, ["queries"]),
    T("web_fetch", "Fetch the content of a specific HTTP(S) URL.", {"url": {"type": "string"}}, ["url"]),
    T("job_output", "Read a background job.", {"job_id": {"type": "string"}, "wait": {"type": "boolean"}, "timeout_ms": {"type": "number"}}, ["job_id"]),
    T("load_workspace_dependencies", "Get absolute paths to bundled Python and library directories.", {}, []),
]

TASKS = [
    ("read", "读取 D:\\chatgpt2api\\start.ps1 并说明它做什么", {"read"}),
    ("shell", "运行 git status --short", {"pwsh"}),
    ("glob", "列出 D:\\chatgpt2api\\services\\protocol 下的 Python 文件", {"glob", "pwsh"}),
    ("write", "把 hello 写入 D:\\chatgpt2api\\runtime\\scale_out.txt", {"write"}),
    ("skill", "帮我生成一个 pptx 文件，主题是长沙高端旅游介绍，现在就动手", {"skill", "write", "pwsh", "todo_write", "ask_user_question"}),
    ("todo", "这个任务分三步：先看仓库状态、再改代码、最后跑测试。先把计划记下来。", {"todo_write", "pwsh", "read"}),
    ("search", "在工作区里找哪些文件定义了 CACHEABLE_TEXT_KEYS", {"grep", "glob", "pwsh"}),
]

print(f"prompt chars: {len(tool_system_prompt(TOOLS))} (tools={len(TOOLS)})")


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"__err__": exc.code, "body": exc.read().decode("utf-8", "replace")[:160]}


passed = 0
for index, (label, user, allowed) in enumerate(TASKS, start=1):
    out = post({"model": "gpt-6", "user": f"scale-{index}-{RUN}",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "tools": TOOLS})
    if "__err__" in out:
        print(f"[{index}] {label:<8} HTTP {out['__err__']} {out['body'][:90]}")
        continue
    choice = out["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        print(f"[{index}] {label:<8} MISS finish={choice['finish_reason']} {str(choice['message'].get('content'))[:80]!r}")
        continue
    name = calls[0]["function"]["name"]
    args = json.loads(calls[0]["function"]["arguments"] or "{}")
    schema = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    missing = [key for key in schema.get(name, {}).get("required", []) if key not in args]
    ok = name in allowed and not missing
    passed += ok
    print(f"[{index}] {label:<8} {'OK ' if ok else 'BAD'} {name}({json.dumps(args, ensure_ascii=False)[:110]})"
          + (f" missing={missing}" if missing else ""))

print(f"\ndispatchable at DSH scale: {passed}/{len(TASKS)}")
