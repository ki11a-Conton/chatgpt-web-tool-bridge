"""Closed-loop PPT test: ask for a deck, execute the returned tool calls for real,
feed the results back, and confirm a real .pptx lands on disk.

This mirrors exactly what DSH does with the calls the bridge returns.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 探针假设仓库根目录就是当前工作目录的父级；路径都从环境变量或相对位置推导。
REPO_ROOT = Path(os.environ.get("CHATGPT2API_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT))

KEY = json.load(open(REPO_ROOT / "config.json", encoding="utf-8"))["auth-key"]
URL = "http://127.0.0.1:3000/v1/chat/completions"
SYSTEM = Path(__file__).with_name("dsh_system_prompt.txt").read_text(encoding="utf-8")
RUNTIME_PY = os.environ.get("DSH_RUNTIME_PY", sys.executable)
OUT = Path(os.environ.get("PPT_OUT", Path.cwd() / "changsha.pptx"))
OUT.unlink(missing_ok=True)


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    tool("pwsh", "Run a PowerShell command on the user's Windows machine and return stdout/stderr.", {"command": {"type": "string"}, "description": {"type": "string"}, "workdir": {"type": "string"}}, ["command", "description"]),
    tool("write", "Create or fully replace a text file.", {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"]),
    tool("read", "Read a UTF-8 text file.", {"file_path": {"type": "string"}}, ["file_path"]),
    tool("skill", "Load the full instructions for a skill.", {"name": {"type": "string"}}, ["name"]),
]

ASK = "帮我生成一个 10 页的长沙高端旅游介绍 PPT，保存成 pptx 文件到 D:\\chatgpt2api\\runtime\\changsha.pptx"


def post(messages, nonce):
    req = urllib.request.Request(URL, json.dumps({"model": "gpt-6", "user": nonce, "messages": messages, "tools": TOOLS}).encode(),
                                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"__err__": exc.code, "body": exc.read().decode("utf-8", "replace")[:200]}


def execute(name, args):
    if name == "skill":
        # The real client answers this from its own skill catalog; emulate the load.
        return (
            "office-pptx skill loaded. Create PowerPoint files with the bundled python-pptx runtime. "
            f'Use this interpreter, which already has python-pptx: "{RUNTIME_PY}". '
            "Run it with the script path quoted."
        )
    if name == "write":
        target = Path(str(args.get("file_path")))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args.get("content") or ""), encoding="utf-8")
        return f"wrote {target} ({len(str(args.get('content') or ''))} chars)"
    if name == "read":
        try:
            return Path(str(args.get("file_path"))).read_text(encoding="utf-8", errors="replace")[:2000]
        except Exception as exc:
            return f"read failed: {exc}"
    if name == "pwsh":
        command = str(args.get("command") or "")
        done = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        output = (done.stdout or "") + (done.stderr or "")
        return output[:2000] or f"(exit {done.returncode}, no output)"
    return f"unknown tool {name}"


messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]
for step in range(1, 6):
    out = post(messages, f"ppt-loop-{step}-{int(time.time())}")
    if "__err__" in out:
        print(f"step {step}: HTTP {out['__err__']} {out['body'][:150]}")
        break
    choice = out["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    text = str(message.get("content") or "")
    if not calls:
        print(f"step {step}: FINAL finish={choice['finish_reason']} text={text[:200]!r}")
        break
    messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
    for call in calls:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")
        print(f"step {step}: {name} {json.dumps(args, ensure_ascii=False)[:150]}")
        observation = execute(name, args)
        print(f"         -> {observation[:200].replace(chr(10), ' | ')}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})

print("\n" + "=" * 72)
if OUT.exists():
    size = OUT.stat().st_size
    valid = False
    try:
        import zipfile
        with zipfile.ZipFile(OUT) as zf:
            slides = [n for n in zf.namelist() if n.startswith("ppt/slides/slide")]
        valid = size > 20000 and len(slides) >= 10
        print(f"changsha.pptx exists: {size} bytes, {len(slides)} slide parts -> {'PASS' if valid else 'SUSPECT'}")
    except Exception as exc:
        print(f"changsha.pptx exists but unreadable: {exc} -> FAIL")
else:
    print("changsha.pptx was NOT created -> FAIL")
