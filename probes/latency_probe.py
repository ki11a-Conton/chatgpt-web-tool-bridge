"""拆解一轮请求的耗时构成，定位"慢"到底慢在哪。

阶段：bootstrap(GET /) -> sentinel prepare -> PoW 求解 -> sentinel finalize
      -> POST /backend-api/conversation -> 首字节(TTFB) -> 整轮结束
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.account_service import account_service  # noqa: E402
from services.openai_backend_api import OpenAIBackendAPI  # noqa: E402
from utils.helper import iter_sse_payloads  # noqa: E402
from utils.pow import build_proof_token  # noqa: E402


def timed(label, fn, marks):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    marks.append((label, elapsed))
    return result


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    all_marks = []
    for index in range(rounds):
        marks = []
        token = account_service.get_text_access_token()
        backend = OpenAIBackendAPI(access_token=token)

        timed("1 bootstrap(GET /)", backend._bootstrap, marks)
        requirements = timed("2 sentinel prepare+finalize", backend._get_chat_requirements, marks)

        # 单独量一次 PoW 求解成本
        pow_marks = []
        prep_path = "/backend-api/sentinel/chat-requirements/prepare"
        prep = backend.session.post(
            backend.base_url + prep_path,
            headers=backend._headers(prep_path, {"Content-Type": "application/json"}),
            json={"p": ""},
            timeout=30,
        ).json()
        proof = prep.get("proofofwork") or {}
        if proof.get("required"):
            timed(
                "3 PoW solve",
                lambda: build_proof_token(proof.get("seed", ""), proof.get("difficulty", ""),
                                          backend.user_agent, script_sources=backend.pow_script_sources,
                                          data_build=backend.pow_data_build),
                pow_marks,
            )
        marks.extend(pow_marks)

        path, timezone = backend._chat_target()
        payload = backend._conversation_payload([{"role": "user", "content": "回答两个字：收到"}], "auto", timezone)

        start = time.perf_counter()
        response = backend.session.post(
            backend.base_url + path,
            headers=backend._conversation_headers(path, requirements),
            json=payload, timeout=300, stream=True,
        )
        marks.append(("4 POST accepted", time.perf_counter() - start))

        first = None
        done = None
        for raw in iter_sse_payloads(response):
            if first is None:
                first = time.perf_counter()
            if raw == "[DONE]":
                done = time.perf_counter()
        response.close()
        if first:
            marks.append(("5 TTFB(first event)", first - start))
        if done:
            marks.append(("6 full turn", done - start))

        total = sum(v for k, v in marks if k.startswith(("1 ", "2 ", "4 "))) + (
            marks[-1][1] if marks[-1][0] == "6 full turn" else 0)
        print(f"--- round {index + 1} ---")
        for label, elapsed in marks:
            print(f"  {label:28s} {elapsed * 1000:8.0f} ms")
        all_marks.append(marks)
        backend.close()

    print("\n=== 各阶段均值 ===")
    labels = [label for label, _ in all_marks[0]]
    for label in labels:
        values = [dict(m)[label] for m in all_marks if label in dict(m)]
        if values:
            print(f"  {label:28s} {sum(values) / len(values) * 1000:8.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
