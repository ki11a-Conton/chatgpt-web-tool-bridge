"""验证「跳过 bootstrap」在完整对话链路上是否稳定可靠。

bootstrap 每轮 ~355ms，只为拿 pow_script_sources / pow_data_build（PoW 的
环境指纹之一）。若服务端不校验这两个字段，就能安全缓存或省略。
多轮 + 真实工具调用都要覆盖，避免只看单次成功就下结论。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.account_service import account_service  # noqa: E402
from services.openai_backend_api import OpenAIBackendAPI  # noqa: E402
from services.protocol import conversation as conv  # noqa: E402
from services.protocol import openai_tool_bridge as bridge  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a UTF-8 text file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
]


def run_turn(backend, messages, model="auto"):
    raw = ""
    for event in conv.conversation_events(backend, messages=messages, model=model):
        if event.get("type") == "conversation.delta":
            raw += str(event.get("delta") or "")
    return raw


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    token = account_service.get_text_access_token()

    print("=== A. 跳过 bootstrap，连续多轮对话 ===")
    backend = OpenAIBackendAPI(access_token=token)
    ok = 0
    skipped_ms = []
    for index in range(rounds):
        start = time.perf_counter()
        try:
            requirements = backend._get_chat_requirements()
            raw = run_turn(backend, [{"role": "user", "content": f"第{index+1}轮，回答两个字：收到"}])
            elapsed = (time.perf_counter() - start) * 1000
            status = "OK" if raw.strip() else "EMPTY"
            if raw.strip():
                ok += 1
            skipped_ms.append(elapsed)
            print(f"  第{index+1}轮: {status:6s} {elapsed:7.0f} ms  回复={raw[:20]!r}")
        except Exception as exc:
            print(f"  第{index+1}轮: FAIL {type(exc).__name__}: {str(exc)[:120]}")

    print(f"  => {ok}/{rounds} 成功，平均 {sum(skipped_ms)/max(1,len(skipped_ms)):.0f} ms")

    print("\n=== B. 跳过 bootstrap 时的工具调用 ===")
    backend2 = OpenAIBackendAPI(access_token=token)
    messages = bridge.to_upstream_messages(
        [{"role": "user", "content": "读取 D:\\chatgpt2api\\start.ps1"}], TOOLS)
    try:
        raw = run_turn(backend2, messages)
        _text, calls = bridge.parse_reply(raw, TOOLS)
        names = [c["function"]["name"] for c in calls]
        print(f"  工具调用: {names}")
        print("  => 工具链路正常" if names else "  => 工具链路异常")
    except Exception as exc:
        print(f"  FAIL {type(exc).__name__}: {str(exc)[:150]}")

    print("\n=== C. 对照组：完整 bootstrap 的一轮耗时 ===")
    backend3 = OpenAIBackendAPI(access_token=token)
    start = time.perf_counter()
    backend3._bootstrap()
    boot_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    backend3._get_chat_requirements()
    sentinel_ms = (time.perf_counter() - start) * 1000
    print(f"  bootstrap {boot_ms:.0f} ms + sentinel {sentinel_ms:.0f} ms = 可省 {boot_ms:.0f} ms/轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
