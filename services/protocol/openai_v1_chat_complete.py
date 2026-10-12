from __future__ import annotations

import time
import uuid
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.chat_completion_cache import cache_key, chat_completion_cache, normalize_text_messages
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    collect_image_outputs,
    collect_text,
    count_message_image_tokens,
    count_message_text_tokens,
    count_text_tokens,
    encode_images,
    is_transport_failure,
    normalize_messages,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    text_backend,
)
from services.protocol.openai_tool_bridge import (
    announces_action,
    calls_missing_arguments,
    calls_destructive,
    calls_usable,
    claims_web_artifact,
    claims_unverified_lookup,
    corrective_messages,
    deflects_to_user,
    explains_instead_of_acting,
    function_tools,
    is_refusal,
    looks_like_tool_envelope,
    parse_reply,
    to_upstream_messages,
    tool_choice_required,
    unknown_tool_names,
)
from services.protocol.reasoning import thinking_effort_from_body
from services.protocol.web_search_tool import (
    WEB_SEARCH_TOOL_TYPES,
    has_unsupported_tools,
    is_web_search_chat_request,
    run_web_search,
    search_query_from_messages,
    text_with_url_citations,
)
from utils.helper import build_chat_image_markdown_content, extract_chat_image, extract_chat_prompt, is_image_chat_request, parse_image_count
from utils.image_tokens import (
    chat_usage_from_image_usage,
    count_image_inputs_tokens,
    count_image_output_items_tokens,
    image_usage,
    image_output_metadata,
)

TOOL_UNAVAILABLE_SYSTEM_MESSAGE = (
    "This compatibility backend cannot execute local tools, shell commands, non-search tools, "
    "or file operations. Do not claim to have run tools or inspected external resources. "
    "If a user asks you to use a tool, say that tool execution is unavailable through this backend."
)


def completion_chunk(model: str, delta: dict[str, Any], finish_reason: str | None = None, completion_id: str = "", created: int | None = None) -> dict[str, Any]:
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion_response(
    model: str,
    content: str,
    created: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt_text_tokens = count_message_text_tokens(messages, model) if messages else 0
    prompt_image_tokens = count_message_image_tokens(messages, model) if messages else 0
    prompt_tokens = prompt_text_tokens + prompt_image_tokens
    completion_tokens = count_text_tokens(content, model) if messages else 0
    message = {"role": "assistant", "content": content}
    if annotations:
        message["annotations"] = annotations
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "text_tokens": prompt_text_tokens,
                "image_tokens": prompt_image_tokens,
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "text_tokens": completion_tokens,
                "image_tokens": 0,
                "reasoning_tokens": 0,
            },
        },
    }


def _with_log_metadata(
    payload: dict[str, Any],
    account_email: str = "",
    conversation_id: str = "",
    image_urls: Iterable[str] | None = None,
    image_attempts: Iterable[dict[str, Any]] | None = None,
    image_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if account_email:
        payload["_account_email"] = account_email
    if conversation_id:
        payload["_conversation_id"] = conversation_id
    urls = [str(url).strip() for url in image_urls or [] if str(url).strip()]
    if urls:
        payload["_image_urls"] = list(dict.fromkeys(urls))
    attempts = [dict(item) for item in image_attempts or [] if isinstance(item, dict)]
    if attempts:
        payload["_image_attempts"] = attempts
    if image_data:
        payload["_image_metadata"] = image_output_metadata(image_data)
    return payload


def _backend_account_email(backend: object) -> str:
    return str(getattr(backend, "account_email", "") or "").strip()


def stream_text_chat_completion(
    backend,
    messages: list[dict[str, Any]],
    model: str,
    thinking_effort: str = "",
) -> Iterator[dict[str, Any]]:
    """无工具轮次：真流式转发上游增量。带工具的轮次走 tool_turn（缓冲式）。

    空回复在客户端会被当成坏响应，所以先拿到第一个非空增量再开始下发；整轮都空则
    带原因重试，最后仍为空就报 provider 错误。
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    request = ConversationRequest(model=model, messages=messages, thinking_effort=thinking_effort)
    sent_role = False
    produced = False
    transport_failures = 0
    for attempt in range(3):
        produced = False
        try:
            for delta_text in stream_text_deltas(backend, request):
                if not delta_text:
                    continue
                produced = True
                if not sent_role:
                    sent_role = True
                    yield _with_log_metadata(
                        completion_chunk(model, {"role": "assistant", "content": delta_text}, None, completion_id, created),
                        _backend_account_email(backend),
                    )
                else:
                    yield _with_log_metadata(
                        completion_chunk(model, {"content": delta_text}, None, completion_id, created),
                        _backend_account_email(backend),
                    )
        except Exception as exc:
            if produced or not _is_transport_failure(exc):
                raise
            transport_failures += 1
            if attempt == 2:
                raise _transport_upstream_error() from exc
            continue
        if produced:
            break
        if attempt < 2:
            request = ConversationRequest(
                model=model,
                messages=[
                    *messages,
                    {"role": "user", "content": "Your previous reply was empty. Answer the request above in plain text now."},
                ],
                thinking_effort=thinking_effort,
            )
    if not produced:
        raise _transport_upstream_error() if transport_failures else _empty_upstream_error()
    if not sent_role:
        yield _with_log_metadata(
            completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created),
            _backend_account_email(backend),
        )
    yield _with_log_metadata(completion_chunk(model, {}, "stop", completion_id, created), _backend_account_email(backend))


def _upstream_error(code: str, message: str) -> HTTPException:
    """上游不可用时按 provider 错误对外，让客户端退避重试。"""
    return HTTPException(
        status_code=503,
        detail={"error": {"message": message, "type": "server_error", "code": code}},
        headers={"Retry-After": "20"},
    )


def _empty_upstream_error() -> HTTPException:
    """空回复多半是账号/模型限流：按 provider 错误对外，让客户端退避重试。"""
    return _upstream_error(
        "upstream_empty_response",
        "ChatGPT Web returned an empty response for this turn. Retry shortly.",
    )


def _malformed_upstream_error() -> HTTPException:
    return _upstream_error(
        "upstream_malformed_tool_envelope",
        "ChatGPT Web returned a truncated tool call instead of a valid one. Retry shortly.",
    )


def _needs_repair(raw: str, text: str, calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
    if not (raw or "").strip():
        return True
    if not (text or "").strip():
        # 上游只回了路由噪声之类的东西：raw 非空但没有任何可见正文，也不是工具调用。
        return True
    if looks_like_tool_envelope(raw):
        return True
    if unknown_tool_names(raw, tools):
        return True
    if calls:
        return True  # 到这里说明调用不可用（缺必填参数）
    return (
        is_refusal(text)
        or announces_action(text)
        or deflects_to_user(text)
        or claims_web_artifact(text)
        or explains_instead_of_acting(text)
        or claims_unverified_lookup(text)
    )


def _empty_problem(raw: str) -> str:
    """空回复几乎总是模型改用了它自己的内部工具，什么都没走文本通道。"""
    del raw
    return (
        "your reply was empty. Do not use any built-in or internal tool of your own and do not "
        "perform the work yourself: emit the tool call as the JSON object described above, "
        "because only the client can run it"
    )


def _repair_problem(raw: str, tools: list[dict[str, Any]]) -> str:
    if not (raw or "").strip():
        return _empty_problem(raw)
    unknown = unknown_tool_names(raw, tools)
    if unknown:
        return f"unknown tool name(s): {', '.join(unknown)}"
    if looks_like_tool_envelope(raw):
        return "the JSON was truncated or malformed"
    _text, calls = parse_reply(raw, tools)
    if calls and calls_missing_arguments(calls, tools):
        return "required parameters were missing"
    if claims_web_artifact(_text):
        return (
            "you claimed the deliverable is done or offered a download instead of calling a tool. "
            "The work only happens on the client machine: call a tool now, and if a previous tool call "
            "failed, fix that cause and retry the tool"
        )
    if explains_instead_of_acting(_text):
        return (
            "you described a plan or what you are about to do instead of doing it. "
            "Emit the tool call for the first concrete step right now, as the JSON object and nothing else"
        )
    if claims_unverified_lookup(_text):
        return (
            "you described the content of a page or resource without calling a tool. "
            "Call the tool that actually retrieves it now, as the JSON object and nothing else"
        )
    return "you refused, announced the action, or handed the command back to the user"


def _is_transport_failure(exc: BaseException) -> bool:
    """连接层失败（curl 超时/TLS/重置）是可重试的瞬时错误，不该一次就对外报错。"""
    return is_transport_failure(exc)


def _transport_upstream_error() -> HTTPException:
    return _upstream_error(
        "upstream_unreachable",
        "Could not reach ChatGPT Web for this turn. Retry shortly.",
    )


def tool_turn(
    backend,
    messages: list[dict[str, Any]],
    model: str,
    thinking_effort: str,
    tools: list[dict[str, Any]],
    attempts: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """带工具的一轮：缓冲整轮解析，不可用时带原因纠正重试，全部失败才报 503。

    每次尝试都是一次独立的上游调用，会由号池重新选账号，所以偶发的空回复/坏信封
    通常在下一次就恢复正常；只有连续 attempts 次都不可用才算真的上游失败。
    """
    current_messages = list(messages)
    last_raw = ""
    last_text = ""
    last_calls: list[dict[str, Any]] = []
    transport_failures = 0
    for attempt in range(max(1, attempts)):
        request = ConversationRequest(model=model, messages=current_messages, thinking_effort=thinking_effort)
        try:
            raw = "".join(stream_text_deltas(backend, request))
        except Exception as exc:
            if not _is_transport_failure(exc):
                raise
            transport_failures += 1
            if attempt == max(1, attempts) - 1:
                raise _transport_upstream_error() from exc
            continue
        text, calls = parse_reply(raw, tools)
        destructive = calls_destructive(calls, tools) if calls_usable(calls, tools) else ""
        if calls_usable(calls, tools) and not destructive:
            # 调用可用，但同一条回复里可能还夹着误导性文字（「已完成/给你下载链接」）。
            # 客户端会照发这段文字，所以有调用时一律不带正文。
            return "", calls
        if not _needs_repair(raw, text, calls, tools):
            return text, calls
        last_raw, last_text, last_calls = raw, text, calls
        if attempt == max(1, attempts) - 1:
            break
        current_messages = corrective_messages(
            messages,
            raw or "(empty reply)",
            tools,
            destructive or _repair_problem(raw, tools),
        )
    if transport_failures and not last_raw.strip():
        raise _transport_upstream_error()
    if not last_raw.strip():
        raise _empty_upstream_error()
    if not last_calls and looks_like_tool_envelope(last_raw):
        raise _malformed_upstream_error()
    if last_calls and calls_destructive(last_calls, tools):
        raise _malformed_upstream_error()
    return last_text, last_calls


def _text_pieces(text: str, size: int = 200) -> list[str]:
    return [text[index:index + size] for index in range(0, len(text), size)]

def tool_stream_chunks(
    model: str,
    text: str,
    calls: list[dict[str, Any]],
    account_email: str = "",
) -> Iterator[dict[str, Any]]:
    """上游没有真流式；工具轮次在拿到完整结果后再发 SSE，保证 repair/重试发生在发出任何字节之前。"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    for piece in _text_pieces(text):
        delta = {"role": "assistant", "content": piece} if not sent_role else {"content": piece}
        sent_role = True
        yield _with_log_metadata(completion_chunk(model, delta, None, completion_id, created), account_email)
    for index, call in enumerate(calls):
        head: dict[str, Any] = {
            "tool_calls": [{
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }],
        }
        if index == 0 and not sent_role:
            head["role"] = "assistant"
            sent_role = True
        yield _with_log_metadata(completion_chunk(model, head, None, completion_id, created), account_email)
        yield _with_log_metadata(
            completion_chunk(
                model,
                {"tool_calls": [{"index": index, "function": {"arguments": call["function"]["arguments"]}}]},
                None,
                completion_id,
                created,
            ),
            account_email,
        )
    if not sent_role:
        yield _with_log_metadata(
            completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created),
            account_email,
        )
    finish_reason = "tool_calls" if calls else "stop"
    yield _with_log_metadata(completion_chunk(model, {}, finish_reason, completion_id, created), account_email)


def collect_chat_content(chunks: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        content = str(delta.get("content") or "")
        if content:
            parts.append(content)
    return "".join(parts)


def chat_messages_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        return [message for message in messages if isinstance(message, dict)]
    prompt = str(body.get("prompt") or "").strip()
    if prompt:
        return [{"role": "user", "content": prompt}]
    raise HTTPException(status_code=400, detail={"error": "messages or prompt is required"})


def chat_image_args(body: dict[str, Any]) -> tuple[str, str, int, list[tuple[bytes, str, str]], str | None]:
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    prompt = extract_chat_prompt(body)
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    images = [
        (data, f"image_{idx}.png", mime)
        for idx, (data, mime) in enumerate(extract_chat_image(body), start=1)
    ]
    base_url = str(body.get("base_url") or "").strip() or None
    return model, prompt, parse_image_count(body.get("n")), images, base_url


def text_chat_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    tools = function_tools(body)
    raw_messages = chat_messages_from_body(body)
    if tools:
        raw_messages = to_upstream_messages(raw_messages, tools, required=tool_choice_required(body))
    messages = normalize_text_messages(normalize_messages(raw_messages))
    if not tools and has_unsupported_tools(body, WEB_SEARCH_TOOL_TYPES):
        messages.insert(0, {"role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE})
    return model, messages, tools


def chat_completion_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in annotations:
        if item.get("type") != "url_citation":
            continue
        output.append({
            "type": "url_citation",
            "url_citation": {
                "start_index": item.get("start_index", 0),
                "end_index": item.get("end_index", 0),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
            },
        })
    return output


def web_search_chat_response(messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
    query = search_query_from_messages(messages)
    if not query:
        raise HTTPException(status_code=400, detail={"error": "messages or prompt is required for web search"})
    text, annotations = text_with_url_citations(run_web_search(query))
    return completion_response(
        model,
        text,
        messages=messages,
        annotations=chat_completion_annotations(annotations),
    )


def stream_web_search_chat_completion(messages: list[dict[str, Any]], model: str) -> Iterator[dict[str, Any]]:
    query = search_query_from_messages(messages)
    if not query:
        raise HTTPException(status_code=400, detail={"error": "messages or prompt is required for web search"})
    text, _annotations = text_with_url_citations(run_web_search(query))
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield completion_chunk(model, {"role": "assistant", "content": text}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


def image_result_content(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, list) and data:
        return build_chat_image_markdown_content(result)
    return str(result.get("message") or "Image generation completed.")


def image_chat_response(body: dict[str, Any]) -> dict[str, Any]:
    model, prompt, n, images, base_url = chat_image_args(body)
    result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
        base_url=base_url,
        message_as_error=True,
        call_id=str(body.get("_call_id") or ""),
        trace_image_perf=bool(body.get("_trace_image_perf")),
    )))
    response = completion_response(model, image_result_content(result), int(result.get("created") or 0) or None)
    usage = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        input_image_tokens=count_image_inputs_tokens(images, model),
        output_tokens=count_image_output_items_tokens(result.get("data")),
    )
    response["usage"] = chat_usage_from_image_usage(usage)
    _with_log_metadata(
        response,
        str(result.get("_account_email") or ""),
        str(result.get("_conversation_id") or ""),
        result.get("_image_urls") if isinstance(result.get("_image_urls"), list) else None,
        result.get("_image_attempts") if isinstance(result.get("_image_attempts"), list) else None,
        result.get("data"),
    )
    return response


def image_chat_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    model, prompt, n, images, base_url = chat_image_args(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
        base_url=base_url,
        message_as_error=True,
        call_id=str(body.get("_call_id") or ""),
        trace_image_perf=bool(body.get("_trace_image_perf")),
    ))
    yield from stream_image_chat_completion(image_outputs, model)


def stream_image_chat_completion(image_outputs: Iterable[ImageOutput], model: str) -> Iterator[dict[str, Any]]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    sent_text = ""
    for output in image_outputs:
        content = ""
        if output.kind == "progress":
            content = output.text
            sent_text += content
        elif output.kind == "result":
            content = build_chat_image_markdown_content({"data": output.data})
        elif output.kind == "message":
            content = output.text[len(sent_text):] if output.text.startswith(sent_text) else output.text
        if not content:
            continue
        if not sent_role:
            sent_role = True
            yield _with_log_metadata(
                completion_chunk(model, {"role": "assistant", "content": content}, None, completion_id, created),
                output.account_email,
                output.conversation_id,
                output.image_urls,
                output.image_attempts,
                output.data if output.kind == "result" else None,
            )
        else:
            yield _with_log_metadata(
                completion_chunk(model, {"content": content}, None, completion_id, created),
                output.account_email,
                output.conversation_id,
                output.image_urls,
                output.image_attempts,
                output.data if output.kind == "result" else None,
            )
    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


def plain_text_turn(
    backend,
    messages: list[dict[str, Any]],
    model: str,
    thinking_effort: str,
    attempts: int = 3,
) -> str:
    """无工具的一轮：空回复同样要重试，最后仍为空就按 provider 错误报出去。

    客户端会把「finish_reason=stop 但 content 为空」当成坏响应（DSH 直接报
    `returned a completed response with no content`），所以这里不能把空串当成功。
    """
    last = ""
    transport_failures = 0
    for attempt in range(max(1, attempts)):
        request = ConversationRequest(model=model, messages=messages, thinking_effort=thinking_effort)
        try:
            last = "".join(stream_text_deltas(backend, request))
        except Exception as exc:
            if not _is_transport_failure(exc):
                raise
            transport_failures += 1
            if attempt == max(1, attempts) - 1:
                raise _transport_upstream_error() from exc
            continue
        if last.strip():
            return last
        if attempt < max(1, attempts) - 1:
            retry_messages = [
                *messages,
                {"role": "user", "content": "Your previous reply was empty. Answer the request above in plain text now."},
            ]
            messages = retry_messages
    if transport_failures:
        raise _transport_upstream_error()
    raise _empty_upstream_error()


def text_completion_response(model: str, messages: list[dict[str, Any]], thinking_effort: str) -> dict[str, Any]:
    """无工具轮次：纯文本完成。带工具的轮次走 tool_completion_response。"""
    backend = text_backend()
    response = completion_response(
        model,
        plain_text_turn(backend, messages, model, thinking_effort),
        messages=messages,
    )
    return _with_log_metadata(response, _backend_account_email(backend))


def tool_completion_response(
    model: str,
    messages: list[dict[str, Any]],
    thinking_effort: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    backend = text_backend()
    text, calls = tool_turn(backend, messages, model, thinking_effort, tools)
    response = completion_response(model, text, messages=messages)
    if calls:
        choice = response["choices"][0]
        choice["finish_reason"] = "tool_calls"
        choice["message"]["content"] = None
        choice["message"]["tool_calls"] = calls
    return _with_log_metadata(response, _backend_account_email(backend))


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    if body.get("stream"):
        if is_image_chat_request(body):
            return image_chat_events(body)
        model, messages, tools = text_chat_parts(body)
        if is_web_search_chat_request(body) and not has_unsupported_tools(body, WEB_SEARCH_TOOL_TYPES):
            return stream_web_search_chat_completion(messages, model)
        thinking_effort = thinking_effort_from_body(body)
        if tools:
            backend = text_backend()
            text, calls = tool_turn(backend, messages, model, thinking_effort, tools)
            return tool_stream_chunks(model, text, calls, _backend_account_email(backend))
        key = cache_key(body, messages, stream=True)
        return chat_completion_cache.get_or_compute_stream(
            key,
            lambda: stream_text_chat_completion(text_backend(), messages, model, thinking_effort),
        )
    if is_image_chat_request(body):
        return image_chat_response(body)
    model, messages, tools = text_chat_parts(body)
    if is_web_search_chat_request(body) and not has_unsupported_tools(body, WEB_SEARCH_TOOL_TYPES):
        return web_search_chat_response(messages, model)
    thinking_effort = thinking_effort_from_body(body)
    if tools:
        return tool_completion_response(model, messages, thinking_effort, tools)
    key = cache_key(body, messages, stream=False)
    return chat_completion_cache.get_or_compute_response(
        key,
        lambda: text_completion_response(model, messages, thinking_effort),
    )
