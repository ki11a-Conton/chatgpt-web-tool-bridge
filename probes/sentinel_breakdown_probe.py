"""拆解 sentinel 那 ~1159ms 到底花在哪：prepare / turnstile / PoW / finalize。

只有知道哪一段最贵，才知道有没有可优化空间。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.account_service import account_service  # noqa: E402
from services.openai_backend_api import OpenAIBackendAPI  # noqa: E402
from utils.pow import build_legacy_requirements_token, build_proof_token  # noqa: E402

try:
    from utils.pow import solve_turnstile_token
except ImportError:
    solve_turnstile_token = None


def main() -> int:
    token = account_service.get_text_access_token()
    backend = OpenAIBackendAPI(access_token=token)

    print("=== 各段耗时（5 次采样） ===")
    rows = []
    for index in range(5):
        row = {}
        base = "/backend-api/sentinel/chat-requirements"

        start = time.perf_counter()
        p_token = build_legacy_requirements_token(backend.user_agent, backend.pow_script_sources,
                                                  backend.pow_data_build)
        row["p_token build"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        response = backend.session.post(
            backend.base_url + base + "/prepare",
            headers=backend._headers(base + "/prepare", {"Content-Type": "application/json"}),
            json={"p": p_token}, timeout=30)
        prepare = response.json()
        row["prepare (HTTP)"] = (time.perf_counter() - start) * 1000

        proof = prepare.get("proofofwork") or {}
        start = time.perf_counter()
        proof_token = ""
        if proof.get("required"):
            proof_token = build_proof_token(proof.get("seed", ""), proof.get("difficulty", ""),
                                            backend.user_agent, script_sources=backend.pow_script_sources,
                                            data_build=backend.pow_data_build)
        row["PoW solve"] = (time.perf_counter() - start) * 1000

        turnstile = prepare.get("turnstile") or {}
        start = time.perf_counter()
        turnstile_token = ""
        if turnstile.get("required") and turnstile.get("dx") and solve_turnstile_token:
            turnstile_token = solve_turnstile_token(turnstile["dx"], p_token) or ""
        row["turnstile solve"] = (time.perf_counter() - start) * 1000
        row["_turnstile_required"] = bool(turnstile.get("required"))
        row["_dx_len"] = len(turnstile.get("dx") or "")
        row["_so_required"] = bool((prepare.get("so") or {}).get("required"))

        start = time.perf_counter()
        backend.session.post(
            backend.base_url + base + "/finalize",
            headers=backend._headers(base + "/finalize", {"Content-Type": "application/json"}),
            json={"prepare_token": prepare.get("prepare_token", ""), "proof_token": proof_token,
                  "turnstile_token": turnstile_token}, timeout=30)
        row["finalize (HTTP)"] = (time.perf_counter() - start) * 1000
        rows.append(row)

    labels = [k for k in rows[0] if not k.startswith("_")]
    for label in labels:
        values = [r[label] for r in rows]
        print(f"  {label:20s} 均值 {sum(values)/len(values):7.0f} ms   最小 {min(values):6.0f} ms")
    print(f"  turnstile.required = {rows[0]['_turnstile_required']}  (dx {rows[0]['_dx_len']} 字符)")
    print(f"  so.required        = {rows[0]['_so_required']}")
    total = sum(sum(r[label] for label in labels) for r in rows) / len(rows)
    print(f"  => sentinel 合计均值 {total:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
