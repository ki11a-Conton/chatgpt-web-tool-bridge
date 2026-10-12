"""OpenAI /v1/chat/completions 的工具调用桥接。

上游是 ChatGPT 网页后端：没有原生 tool_calls 通道，收到 role:"tool" 会 500。
桥接层只翻译、不执行命令；协议为严格 JSON 信封，XML 与上游沙箱回声作为兜底。
"""
import json

import pytest
from fastapi import HTTPException

from services.protocol import anthropic_v1_messages as anthropic
from services.protocol import openai_tool_bridge as bridge
from services.protocol import openai_v1_chat_complete as chat
from services.protocol.tool_markup import parse_tool_calls

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

DECLARED_SHELL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command on the user's machine and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}

SHELL_REPLY = (
    "<tool_calls><tool_call><tool_name>run_shell</tool_name>"
    "<parameters><command><![CDATA[git status]]></command></parameters>"
    "</tool_call></tool_calls>"
)

JSON_REPLY = json.dumps({"tool_calls": [{"name": "run_shell", "arguments": {"command": "git status"}}]})

JUNK_REPLY = '{"paths":["read_file","run_shell"]}I cannot read files.'

REFUSAL_TEXT = "I don't currently have a shell tool available in this chat to run git status."

EMPTY_ARGS_REPLY = "<tool_calls><tool_call><tool_name>run_shell</tool_name><parameters></parameters></tool_call></tool_calls>"

EMPTY_ARGS_JSON = json.dumps({"tool_calls": [{"name": "run_shell", "arguments": {}}]})

REAL_REFUSALS = [
    "I can't execute run_shell or run local commands in this chat backend.",
    "I don't currently have a shell tool available in this chat to run git status.",
    "I don't have access to the `run_shell` tool in this chat session.",
    "I can't directly inspect your local working tree from this chat environment.",
    "I'm unable to read that local file in this chat session.",
]


def _tool_delta_calls(chunks):
    calls = []
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]
        for item in delta.get("tool_calls") or []:
            calls.append(item)
    return calls


def _joined_arguments(chunks) -> dict:
    raw = "".join(str(item["function"]["arguments"]) for item in _tool_delta_calls(chunks))
    return json.loads(raw)


def _fake_deltas(monkeypatch, replies, seen=None):
    state = {"n": 0}

    def fake(backend, request):
        index = state["n"]
        state["n"] += 1
        if seen is not None:
            seen.append(request.messages)
        return iter([replies[min(index, len(replies) - 1)]])

    monkeypatch.setattr(chat, "stream_text_deltas", fake)
    return state


# --- 工具筛选 --------------------------------------------------------------

def test_function_tools_keeps_only_function_type():
    body = {"tools": [SHELL_TOOL, {"type": "web_search"}]}
    assert bridge.function_tools(body) == [SHELL_TOOL]


def test_function_tools_accepts_bare_name_shape():
    bare = {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}
    assert bridge.function_tools({"tools": [bare]}) == [bare]


def test_function_tools_disabled_by_tool_choice_none():
    assert bridge.function_tools({"tools": [SHELL_TOOL], "tool_choice": "none"}) == []


def test_function_tools_restricted_by_named_tool_choice():
    body = {"tools": [SHELL_TOOL, READ_TOOL], "tool_choice": {"type": "function", "function": {"name": "read_file"}}}
    assert bridge.function_tools(body) == [READ_TOOL]


def test_tool_choice_required_detected():
    assert bridge.tool_choice_required({"tool_choice": "required"}) is True
    assert bridge.tool_choice_required({"tool_choice": {"type": "any"}}) is True
    assert bridge.tool_choice_required({"tool_choice": "auto"}) is False
    assert bridge.tool_choice_required({}) is False


PWSH_TOOL = {
    "type": "function",
    "function": {
        "name": "pwsh",
        "description": "Run a PowerShell command and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]},
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def test_shell_flavor_detects_powershell_client():
    assert "PowerShell" in bridge.shell_flavor([PWSH_TOOL, READ_TOOL])
    assert "POSIX" in bridge.shell_flavor([BASH_TOOL])
    assert bridge.shell_flavor([READ_TOOL]) == ""


def test_tool_system_prompt_warns_about_heredocs_and_multiline():
    prompt = bridge.tool_system_prompt([PWSH_TOOL, READ_TOOL])
    assert "PowerShell syntax" in prompt
    assert "heredocs" in prompt
    assert "write the code to a file" in prompt


# --- 提示词 ----------------------------------------------------------------

def test_tool_system_prompt_uses_json_envelope():
    prompt = bridge.tool_system_prompt([SHELL_TOOL])
    assert "AVAILABLE_TOOLS" in prompt
    assert "run_shell" in prompt
    assert "Run a shell command" in prompt
    assert '"tool_calls"' in prompt
    assert "<tool_calls>" not in prompt
    assert bridge.tool_system_prompt([]) == ""


def _many_tools(count: int) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": "A tool that does a thing. " * 4,
                "parameters": {
                    "type": "object",
                    "properties": {"alpha": {"type": "string"}, "beta": {"type": "number"}, "gamma": {"type": "boolean"}},
                    "required": ["alpha", "beta"],
                },
            },
        }
        for index in range(count)
    ]


def test_few_tools_keep_full_json_schemas():
    prompt = bridge.tool_system_prompt([SHELL_TOOL, READ_TOOL])
    assert "Parameters: {" in prompt
    assert "compact" not in prompt


def test_many_tools_switch_to_compact_signatures():
    """DSH 声明 30+ 工具：完整 schema 会把提示词撑到近万字符，必须压缩。"""
    prompt = bridge.tool_system_prompt(_many_tools(30))
    assert "Parameters: {" not in prompt
    assert "tool_0({alpha:string, beta:number, gamma:boolean?})" in prompt
    assert "A trailing `?` means the parameter is optional" in prompt


def test_compact_listing_keeps_required_and_optional_information():
    summary = bridge._schema_summary({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "number"}}, "required": ["a"]})
    assert summary == "{a:string, b:number?}"


def test_schema_summary_handles_nested_and_array_types():
    schema = {
        "type": "object",
        "properties": {
            "files": {"type": "array", "items": {"type": "object"}},
            "options": {"type": "object", "properties": {"label": {"type": "string"}, "description": {"type": "string"}}},
            "plain": {},
        },
        "required": ["files"],
    }
    summary = bridge._schema_summary(schema)
    assert summary == "{files:array<object>, options:object{label, description}?, plain:string?}"


def test_tool_system_prompt_forbids_handing_work_to_the_user():
    prompt = bridge.tool_system_prompt([SHELL_TOOL])
    assert "Do not answer with shell commands" in prompt
    assert "Never claim a file was read" in prompt
    assert "only tool names listed" in prompt


def test_tool_system_prompt_marks_required_choice():
    assert "MUST" in bridge.tool_system_prompt([SHELL_TOOL], required=True)


def test_shell_tool_detects_command_parameter():
    assert bridge.shell_tool([DECLARED_SHELL]) == ("shell", "command")
    assert bridge.shell_tool([READ_TOOL]) is None


# --- 历史折叠 --------------------------------------------------------------

def test_to_upstream_messages_prepends_tool_prompt():
    messages = bridge.to_upstream_messages([{"role": "user", "content": "hi"}], [SHELL_TOOL])
    assert messages[0]["role"] == "system"
    assert "run_shell" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "hi"}


def test_to_upstream_messages_folds_assistant_tool_calls():
    messages = bridge.to_upstream_messages(
        [
            {"role": "user", "content": "git status"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_shell", "arguments": '{"command": "git status"}'},
                    }
                ],
            },
        ],
        [SHELL_TOOL],
    )
    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    assert "tool_calls" not in assistant
    assert parse_tool_calls(assistant["content"]) == [("run_shell", {"command": "git status"})]


def test_to_upstream_messages_folds_tool_role_results():
    messages = bridge.to_upstream_messages(
        [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": " M CHANGELOG.md"},
        ],
        [SHELL_TOOL],
    )
    result = messages[2]
    assert result["role"] == "user"
    assert " M CHANGELOG.md" in result["content"]
    assert "call_1" in result["content"]


def test_to_upstream_messages_folds_legacy_function_role():
    messages = bridge.to_upstream_messages(
        [{"role": "function", "name": "run_shell", "content": "ok"}],
        [SHELL_TOOL],
    )
    assert messages[1]["role"] == "user"
    assert "ok" in messages[1]["content"]


# --- JSON 信封解析 ---------------------------------------------------------

def test_parse_reply_extracts_json_envelope():
    text, calls = bridge.parse_reply(JSON_REPLY, [SHELL_TOOL])
    assert text == ""
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["id"]
    assert calls[0]["function"]["name"] == "run_shell"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_parse_reply_accepts_single_call_shape():
    raw = json.dumps({"tool": "run_shell", "arguments": '{"command": "ls"}'})
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


def test_parse_reply_accepts_fenced_json():
    raw = "```json\n" + JSON_REPLY + "\n```"
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert calls[0]["function"]["name"] == "run_shell"


def test_parse_reply_accepts_openai_shaped_envelope():
    raw = json.dumps({"tool_calls": [{"function": {"name": "run_shell", "arguments": {"command": "ls"}}}]})
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


def test_parse_reply_ignores_non_tool_json():
    text, calls = bridge.parse_reply('{"answer": 1}', [SHELL_TOOL])
    assert calls == []
    assert text == '{"answer": 1}'


def test_parse_reply_remaps_invented_tool_name():
    """上游倾向用它的内部名字 run_shell，必须映射回调用方声明的名字。"""
    _text, calls = bridge.parse_reply(JSON_REPLY, [DECLARED_SHELL])
    assert calls[0]["function"]["name"] == "shell"


def test_parse_reply_drops_unknown_tool_name():
    raw = json.dumps({"tool_calls": [{"name": "bogus_tool", "arguments": {}}]})
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL, READ_TOOL])
    assert calls == []


def test_unknown_tool_names_reports_rejected_names():
    raw = json.dumps({"tool_calls": [{"name": "bogus_tool", "arguments": {}}]})
    assert bridge.unknown_tool_names(raw, [SHELL_TOOL, READ_TOOL]) == ["bogus_tool"]


def test_parse_reply_passes_through_prose():
    text, calls = bridge.parse_reply("Hello there.", [SHELL_TOOL])
    assert text == "Hello there."
    assert calls == []


def test_parse_reply_never_invents_calls_without_tools():
    text, calls = bridge.parse_reply(JSON_REPLY, [])
    assert calls == []
    assert text == JSON_REPLY


# --- XML 与回声兜底 --------------------------------------------------------

def test_parse_reply_still_accepts_xml_fallback():
    _text, calls = bridge.parse_reply(SHELL_REPLY, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_parse_reply_handles_stray_closing_tag():
    raw = (
        "<tool_calls><tool_call><tool_name>read_file</tool_name>"
        "<parameters><path><![CDATA[D:\\cfg.json]]></path></parameters></parameters></tool_call></tool_calls>"
    )
    _text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "D:\\cfg.json"}


def test_parse_reply_converts_upstream_bash_echo():
    raw = "bash -lc git status --shortI couldn't find a Git repository in the current directory."
    text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert text == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status --short"}


def test_parse_reply_converts_glued_echo_before_code_fence():
    raw = "bash -lc uname -a; hostname; id -un; pwd```\nLinux x86_64 GNU/Linux\n```"
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "uname -a; hostname; id -un; pwd"}


def test_parse_reply_converts_only_first_echo_segment():
    raw = "bash -lc pwd && git status --short && git remote -vbash -lc pwd"
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd && git status --short && git remote -v"}


def test_parse_reply_echo_wins_over_empty_json_arguments():
    raw = EMPTY_ARGS_JSON + "\nbash -lc git log --oneline -3"
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git log --oneline -3"}


def test_parse_reply_keeps_empty_call_when_no_echo_available():
    _text, calls = bridge.parse_reply(EMPTY_ARGS_JSON, [SHELL_TOOL])
    assert calls[0]["function"]["arguments"] == "{}"


def test_parse_reply_strips_truncated_router_noise():
    """上游会吐被截断的路由头（如 {"paths":["skills://office-pptx/skill），不能当正文。"""
    for raw in (
        '{"paths":["skills://office-pptx/skill',
        '{"paths":["skills://office-pptx/skill"}',
        '{"paths":["skills://office-pptx/skill","x"],"query":null}',
    ):
        text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
        assert calls == []
        assert text == "", raw


def test_truncated_router_noise_detection_does_not_eat_tool_envelopes():
    assert bridge._looks_like_router_noise_text('{"paths":["x"]') is True
    assert bridge._looks_like_router_noise_text('{"tool_calls":[{"name":"a"') is False
    assert bridge._looks_like_router_noise_text('{"answer": 1}') is False


def test_parse_reply_strips_extra_arguments():
    """客户端会严格校验参数，多出的键会让整轮失败；上游习惯补一个 description。"""
    raw = json.dumps({"tool_calls": [{"name": "run_shell", "arguments": {"command": "git status", "description": "看看状态"}}]})
    _text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_strip_extra_arguments_keeps_declared_optional_params():
    """READ_TOOL 只声明 path，file_path 和 bogus 都不是它的参数。"""
    raw = json.dumps({"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py", "bogus": 1}}]})
    _text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}


def test_clean_text_treats_echo_shell_as_empty():
    """沙箱回声（含它执行的命令）不是可用的正文——要么转成调用，要么判空。"""
    for raw in ("bash -lc", "bash -lc ", "`bash -lc`", "bash -lc git status", "cmd /c echo hi"):
        assert bridge.clean_text(raw) == "", raw
    # 回声后面跟着真实正文时，只剥回声，正文保留。
    assert bridge.clean_text("bash -lc ls\n真正的回答在这里") == "真正的回答在这里"


def test_parse_reply_ignores_bare_echo_shell():
    text, calls = bridge.parse_reply("bash -lc", [SHELL_TOOL])
    assert calls == []
    assert text == ""


def test_tool_system_prompt_forbids_whole_file_edits():
    prompt = bridge.tool_system_prompt([PWSH_TOOL, READ_TOOL])
    assert "Never replace a whole file body through an edit call" in prompt
    assert "Do not rewrite a file you have not read in this session" in prompt


def test_calls_destructive_flags_emptying_edit():
    """实测事故：整文件替换式 edit 把脚本压成空壳。"""
    calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "edit", "arguments": json.dumps({"file_path": "a.ps1", "old_string": "$ErrorActionPreference='Stop'\n$out = 1\n", "new_string": ""})},
    }]
    assert bridge.calls_destructive(calls, [SHELL_TOOL]) != ""


def test_calls_destructive_allows_normal_edit():
    calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "edit", "arguments": json.dumps({"file_path": "a.ps1", "old_string": "old text", "new_string": "new text"})},
    }]
    assert bridge.calls_destructive(calls, [SHELL_TOOL]) == ""


def test_calls_destructive_allows_full_write():
    calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "write", "arguments": json.dumps({"file_path": "a.py", "content": "print('hello')\n" * 5})},
    }]
    assert bridge.calls_destructive(calls, [SHELL_TOOL]) == ""


def test_tool_turn_repairs_emptying_edit(monkeypatch):
    bad = json.dumps({"tool_calls": [{"name": "edit", "arguments": {"file_path": "a.ps1", "old_string": "real content here", "new_string": ""}}]})
    seen = []
    state = _fake_deltas(monkeypatch, [bad, JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL, EDIT_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    assert "would delete the matched content" in str(seen[1][-1].get("content") or "")


def test_tool_turn_raises_when_only_destructive_calls_survive(monkeypatch):
    bad = json.dumps({"tool_calls": [{"name": "edit", "arguments": {"file_path": "a.ps1", "old_string": "real content here", "new_string": ""}}]})
    _fake_deltas(monkeypatch, [bad])
    with pytest.raises(HTTPException) as excinfo:
        chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL, EDIT_TOOL])
    assert excinfo.value.status_code == 503


def test_parse_reply_echo_needs_declared_shell_tool():
    text, calls = bridge.parse_reply("bash -lc ls", [READ_TOOL])
    assert calls == []
    assert text == ""


# 上游真实输出：沙箱路由头 + 沙箱回声 + 字符串里含裸换行的 JSON 信封
NOISY_NEWLINE_ENVELOPE = (
    '{"paths":["x"],"query":"x","description":"noop"}bash -lc echo test'
    '{"tool_calls":[{"name":"write","arguments":{"file_path":"D:\\\\out.py","content":"line1\nline2\nline3"}}]}'
)


WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Create or fully replace a text file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Replace an exact string in a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]},
    },
}


def test_parse_reply_survives_noise_then_newline_bearing_envelope():
    text, calls = bridge.parse_reply(NOISY_NEWLINE_ENVELOPE, [SHELL_TOOL, WRITE_TOOL])
    assert text == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["file_path"] == "D:\\out.py"
    assert args["content"] == "line1\nline2\nline3"


def test_escape_control_chars_only_touches_string_literals():
    assert bridge._escape_control_chars('{"a":"x\ny"}') == '{"a":"x\\ny"}'
    assert bridge._escape_control_chars('{"a":1}\n{"b":2}') == '{"a":1}\n{"b":2}'


# --- 上游噪声清理 ----------------------------------------------------------

def test_parse_reply_strips_junk_json_prefix():
    text, calls = bridge.parse_reply(JUNK_REPLY, [SHELL_TOOL])
    assert text == "I cannot read files."
    assert calls == []


def test_parse_reply_strips_repeated_junk_prefixes():
    raw = '{"paths":["run_shell"],"query":""}{"paths":["run_shell"],"query":"command"}I can check the working tree.'
    text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert text == "I can check the working tree."
    assert calls == []


def test_parse_reply_drops_router_noise_alone():
    text, calls = bridge.parse_reply('{"paths":[],"query":""}', [SHELL_TOOL])
    assert text == ""
    assert calls == []


def test_parse_reply_drops_router_noise_before_prose():
    text, calls = bridge.parse_reply('{"paths":[],"query":""} Done.', [SHELL_TOOL])
    assert text == "Done."
    assert calls == []


# --- 调用可用性 ------------------------------------------------------------

def test_calls_missing_arguments_flags_empty_required_params():
    _text, calls = bridge.parse_reply(EMPTY_ARGS_REPLY, [SHELL_TOOL])
    assert bridge.calls_missing_arguments(calls, [SHELL_TOOL]) is True
    assert bridge.calls_usable(calls, [SHELL_TOOL]) is False


def test_calls_usable_accepts_filled_call():
    _text, calls = bridge.parse_reply(JSON_REPLY, [SHELL_TOOL])
    assert bridge.calls_usable(calls, [SHELL_TOOL]) is True


# --- 拒答/空承诺/甩锅识别 --------------------------------------------------

def test_is_refusal_detects_real_denials():
    for text in REAL_REFUSALS:
        assert bridge.is_refusal(text) is True, text


def test_is_refusal_ignores_ordinary_answers():
    for text in (
        "The working tree has two modified files.",
        "I cannot read files.",
        "I cannot help with that.",
        "You can't commit without a message.",
        "The build failed because it could not resolve a dependency.",
    ):
        assert bridge.is_refusal(text) is False, text


def test_announces_action_detected():
    assert bridge.announces_action("I'll check the current working directory now.") is True
    assert bridge.announces_action("Let me look at the repository state.") is True
    assert bridge.announces_action("The working directory is /app.") is False


def test_deflects_to_user_detected():
    deflected = "I need the repository state. Please run:\n```bash\ngit status\n```\nand paste the output."
    assert bridge.deflects_to_user(deflected) is True
    assert bridge.deflects_to_user("Here is an example:\n```bash\ngit commit\n```") is False
    assert bridge.deflects_to_user("The repository has two modified files.") is False


WEB_ARTIFACT_TEXTS = [
    "生成完成后我会直接给你一个可下载文件",
    "我已按“高端大气”方向设计为英文旅游展示 PPT，你可以直接打开生成的 PPT 文件",
    "文件已生成，下载链接在下方",
    "I created the deck for you and here is a download link",
]


def test_claims_web_artifact_detects_handoff_phrasing():
    for text in WEB_ARTIFACT_TEXTS:
        assert bridge.claims_web_artifact(text) is True, text


def test_tool_system_prompt_limits_skill_loading():
    prompt = bridge.tool_system_prompt([PWSH_TOOL, READ_TOOL])
    assert "Never load a skill as a substitute for performing the requested action" in prompt


def test_is_refusal_detects_chinese_denials():
    """实测：中文会话里的拒答是「无法调用工具」「没有可用的工具」这类措辞。"""
    for text in (
        "我目前无法在这个会话中调用可用的本地文件写入工具，因此不能确认已将内容写入文件。",
        "我当前会话中没有可用的工作区/文件搜索工具，因此我无法直接扫描你的工作区。",
        "抱歉，我无法访问你的本地文件。",
        "当前环境不支持执行命令。",
    ):
        assert bridge.is_refusal(text) is True, text


def test_is_refusal_ignores_chinese_ordinary_answers():
    for text in (
        "工作区有两个文件被修改了。",
        "HTTP 503 表示服务暂时不可用。",
        "项目里有 16 个 Python 文件。",
        "这个页面介绍了 Example Domain 的用途。",
    ):
        assert bridge.is_refusal(text) is False, text


def test_claims_unverified_lookup_detected():
    for text in (
        "我看了一下这个页面，它是一份示例文档。",
        "这个页面介绍了 Example Domain 的用途。",
        "I checked the page and it explains the purpose of example domains.",
        "The site contains a short explanation.",
    ):
        assert bridge.claims_unverified_lookup(text) is True, text


def test_claims_unverified_lookup_ignores_ordinary_answers():
    for text in (
        "The working tree has two modified files.",
        "HTTP 503 表示服务暂时不可用。",
        "I will fetch that page for you.",
    ):
        assert bridge.claims_unverified_lookup(text) is False, text


def test_cmd_powershell_echo_is_recognised():
    """上游沙箱回声的前缀会变成 `cmdpowershell` / `cmd /c`，也要能识别。"""
    assert bridge.extract_echo_command("cmdpowershell -NoProfile -Command Get-Content -Raw 'a.txt'") != ""
    assert bridge.clean_text("cmdpowershell -c") == ""
    assert bridge.clean_text("cmd /c echo hi") == ""


def test_echo_prefix_is_stripped_before_the_real_text():
    """`cmd /c echo hi` 与真实回答黏在一起时，回声要剥掉，正文要留下。"""
    assert bridge.clean_text("cmd /c echo hi计划已记录，接下来执行。") == "计划已记录，接下来执行。"
    assert bridge.clean_text("bash -lc ls\n真正的回答在这里") == "真正的回答在这里"


def test_router_noise_with_extra_description_key():
    """实测形态：{"paths":["skills"],"query":"office-pptx","description":"..."} 也是噪声。"""
    raw = '{"paths":["skills"],"query":"office-pptx","description":"生成 PPTX 需要技能说明"}'
    assert bridge._is_router_noise(raw) is True
    text, calls = bridge.parse_reply(raw, [SHELL_TOOL])
    assert text == ""
    assert calls == []


def test_ordinary_json_with_description_is_not_noise():
    assert bridge._is_router_noise('{"description": "a normal answer"}') is False


def test_tool_turn_repairs_unverified_lookup(monkeypatch):
    seen = []
    state = _fake_deltas(monkeypatch, ["这个页面介绍了 Example Domain 的用途。", JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "看看这个网页"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    assert "without calling a tool" in str(seen[1][-1].get("content") or "")


def test_tool_system_prompt_requires_fetching_live_resources():
    prompt = bridge.tool_system_prompt([PWSH_TOOL, READ_TOOL])
    assert "instead of answering from memory" in prompt


def test_explains_instead_of_acting_detected():
    for text in (
        "好的，我会先确认文档结构。",
        "第一步我会检查仓库状态。",
        "Here is my plan for the deck.",
        "First I will read the file, then edit it.",
        "我的计划是分三步完成。",
    ):
        assert bridge.explains_instead_of_acting(text) is True, text


def test_explains_instead_of_acting_ignores_ordinary_answers():
    for text in (
        "The working tree has two modified files.",
        "HTTP 503 表示服务暂时不可用。",
        "Changsha is known for Yuelu Academy.",
    ):
        assert bridge.explains_instead_of_acting(text) is False, text


def test_tool_turn_repairs_plan_only_answer(monkeypatch):
    seen = []
    state = _fake_deltas(monkeypatch, ["好的，我会先确认文档结构。", JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "做 ppt"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    assert "instead of doing it" in str(seen[1][-1].get("content") or "")


def test_compact_listing_forbids_extra_parameters():
    prompt = bridge.tool_system_prompt(_many_tools(30))
    assert "do not add any other key" in prompt


def test_claims_web_artifact_ignores_ordinary_answers():
    for text in (
        "The working tree has two modified files.",
        "I will now run git status to see the changes.",
        "HTTP 503 表示服务暂时不可用。",
        "The presentation covers Changsha landmarks and cuisine.",
    ):
        assert bridge.claims_web_artifact(text) is False, text


def test_tool_system_prompt_forbids_web_artifacts_and_download_links():
    prompt = bridge.tool_system_prompt([PWSH_TOOL, READ_TOOL])
    assert "Never produce the deliverable inside this chat" in prompt
    assert "do not offer a download link" in prompt
    assert "Never fall back to" in prompt


def test_python_hint_only_with_a_shell_tool():
    assert "python-pptx" in bridge.tool_system_prompt([PWSH_TOOL])
    assert "python-pptx" not in bridge.tool_system_prompt([READ_TOOL])


def test_python_hint_forbids_office_com():
    prompt = bridge.tool_system_prompt([PWSH_TOOL])
    assert "never use Office COM automation" in prompt
    assert "RPC_E_CALL_REJECTED" in prompt


def test_python_hint_prefers_loading_the_client_skill():
    prompt = bridge.tool_system_prompt([PWSH_TOOL])
    assert "office-pptx" in prompt
    assert "Load that skill first" in prompt


def test_python_hint_warns_about_paths_with_spaces():
    assert "quote executable paths" in bridge.tool_system_prompt([PWSH_TOOL])


def test_tool_turn_repairs_web_artifact_claim(monkeypatch):
    seen = []
    state = _fake_deltas(monkeypatch, [WEB_ARTIFACT_TEXTS[0], JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "做一份 ppt"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    correction = str(seen[1][-1].get("content") or "")
    assert "download" in correction
    assert "client machine" in correction


# --- 一轮工具调用（缓冲 + repair） -----------------------------------------

def test_tool_turn_drops_prose_when_a_usable_call_exists(monkeypatch):
    """调用可用时不能再带「已完成/给你下载链接」这类误导文字。"""
    raw = "完成后我会直接给你一个可下载文件。\n" + JSON_REPLY
    _fake_deltas(monkeypatch, [raw])
    text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert text == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_tool_turn_returns_tool_calls(monkeypatch):
    _fake_deltas(monkeypatch, [JSON_REPLY])
    text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert text == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_tool_turn_keeps_plain_answer_without_repair(monkeypatch):
    state = _fake_deltas(monkeypatch, ["The working tree has two modified files."])
    text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 1
    assert calls == []
    assert text == "The working tree has two modified files."


def test_tool_turn_repairs_after_refusal(monkeypatch):
    seen = []
    state = _fake_deltas(monkeypatch, [REFUSAL_TEXT, JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    assert any("DO have these tools" in str(item.get("content") or "") for item in seen[1])


def test_tool_turn_repair_message_names_the_problem(monkeypatch):
    seen = []
    raw = json.dumps({"tool_calls": [{"name": "bogus_tool", "arguments": {}}]})
    _fake_deltas(monkeypatch, [raw, JSON_REPLY], seen=seen)
    chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert "unknown tool name(s): bogus_tool" in str(seen[1][-1].get("content") or "")


def test_tool_turn_repairs_empty_arguments(monkeypatch):
    state = _fake_deltas(monkeypatch, [EMPTY_ARGS_JSON, JSON_REPLY])
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_tool_turn_raises_retryable_error_on_empty_reply(monkeypatch):
    state = _fake_deltas(monkeypatch, [""])
    with pytest.raises(HTTPException) as excinfo:
        chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 3  # 空回复要试满所有尝试次数，而不是一次就放弃
    assert excinfo.value.status_code == 503
    assert excinfo.value.headers["Retry-After"] == "20"
    assert excinfo.value.detail["error"]["code"] == "upstream_empty_response"


def test_tool_turn_recovers_when_a_later_attempt_succeeds(monkeypatch):
    state = _fake_deltas(monkeypatch, ["", "", JSON_REPLY])
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 3
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_empty_attempt_steers_away_from_internal_tools(monkeypatch):
    seen = []
    _fake_deltas(monkeypatch, ["", JSON_REPLY], seen=seen)
    chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    correction = str(seen[1][-1].get("content") or "")
    assert "empty" in correction
    assert "internal tool" in correction


def test_tool_turn_returns_prose_when_repair_also_refuses(monkeypatch):
    state = _fake_deltas(monkeypatch, [REFUSAL_TEXT])
    text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 3
    assert calls == []
    assert text


# --- 缓冲后的 SSE / 非流式响应 ---------------------------------------------

TRUNCATED_ENVELOPE = (
    '{"tool_calls":[{"name":"shell","arguments":{"command":"pwd && echo \'---\' && ls -la && git status --short --branch"}]}'
)


def test_parse_reply_does_not_leak_malformed_envelope_as_text():
    text, calls = bridge.parse_reply(TRUNCATED_ENVELOPE, [SHELL_TOOL])
    assert calls == []
    assert text == ""


def test_looks_like_tool_envelope_detection():
    assert bridge.looks_like_tool_envelope(TRUNCATED_ENVELOPE) is True
    assert bridge.looks_like_tool_envelope(JSON_REPLY) is True
    assert bridge.looks_like_tool_envelope("The working tree has two modified files.") is False


def test_tool_turn_repairs_truncated_envelope(monkeypatch):
    seen = []
    _fake_deltas(monkeypatch, [TRUNCATED_ENVELOPE, JSON_REPLY], seen=seen)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}
    assert "truncated or malformed" in str(seen[1][-1].get("content") or "")


def test_tool_turn_raises_on_truncated_envelope_after_repair(monkeypatch):
    state = _fake_deltas(monkeypatch, [TRUNCATED_ENVELOPE])
    with pytest.raises(HTTPException) as excinfo:
        chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 3
    assert excinfo.value.status_code == 503
    assert excinfo.value.headers["Retry-After"] == "20"
    assert excinfo.value.detail["error"]["code"] == "upstream_malformed_tool_envelope"


def test_tool_stream_chunks_emit_fake_sse():
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "run_shell", "arguments": '{"command": "git status"}'},
        }
    ]
    chunks = list(chat.tool_stream_chunks("gpt-6", "Checking now.", calls, "bot@example.com"))
    assert chat.collect_chat_content(chunks) == "Checking now."
    assert _joined_arguments(chunks) == {"command": "git status"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[0]["_account_email"] == "bot@example.com"


def test_tool_stream_chunks_finish_stop_without_calls():
    chunks = list(chat.tool_stream_chunks("gpt-6", "All done.", []))
    assert chat.collect_chat_content(chunks) == "All done."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_tool_completion_response_returns_tool_calls(monkeypatch):
    monkeypatch.setattr(chat, "text_backend", lambda: object())
    _fake_deltas(monkeypatch, [JSON_REPLY])
    response = chat.tool_completion_response("gpt-6", [{"role": "user", "content": "x"}], "", [SHELL_TOOL])
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "run_shell"


def test_tool_completion_response_keeps_plain_answer(monkeypatch):
    monkeypatch.setattr(chat, "text_backend", lambda: object())
    _fake_deltas(monkeypatch, ["Nothing to do."])
    response = chat.tool_completion_response("gpt-6", [{"role": "user", "content": "x"}], "", [SHELL_TOOL])
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == "Nothing to do."


# --- 无工具路径不受影响 ----------------------------------------------------

def test_stream_text_chat_completion_without_tools_is_incremental(monkeypatch):
    monkeypatch.setattr(chat, "stream_text_deltas", lambda backend, request: iter(["Hello", " world"]))
    chunks = list(chat.stream_text_chat_completion(object(), [{"role": "user", "content": "x"}], "gpt-6"))
    assert chat.collect_chat_content(chunks) == "Hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_text_chat_parts_wires_tools_and_drops_unavailable_notice():
    body = {
        "model": "gpt-6",
        "messages": [{"role": "user", "content": "git status"}],
        "tools": [SHELL_TOOL],
    }
    model, messages, tools = chat.text_chat_parts(body)
    assert model == "gpt-6"
    assert len(tools) == 1
    joined = "\n".join(str(item.get("content") or "") for item in messages)
    assert chat.TOOL_UNAVAILABLE_SYSTEM_MESSAGE not in joined
    assert "run_shell" in joined


def test_text_chat_parts_keeps_notice_for_untranslatable_tools():
    body = {
        "model": "gpt-6",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "some_future_tool"}],
    }
    _model, messages, tools = chat.text_chat_parts(body)
    assert tools == []
    assert chat.TOOL_UNAVAILABLE_SYSTEM_MESSAGE in str(messages[0].get("content") or "")


# --- 无工具轮次的空回复兜底 ------------------------------------------------

def test_plain_text_turn_retries_empty_reply(monkeypatch):
    state = _fake_deltas(monkeypatch, ["", "", "PONG"])
    text = chat.plain_text_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "")
    assert state["n"] == 3
    assert text == "PONG"


def test_plain_text_turn_raises_when_always_empty(monkeypatch):
    state = _fake_deltas(monkeypatch, [""])
    with pytest.raises(HTTPException) as excinfo:
        chat.plain_text_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "")
    assert state["n"] == 3
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["error"]["code"] == "upstream_empty_response"


def test_plain_text_turn_returns_first_non_empty(monkeypatch):
    state = _fake_deltas(monkeypatch, ["PONG"])
    assert chat.plain_text_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "") == "PONG"
    assert state["n"] == 1


def test_stream_text_chat_completion_retries_empty_then_streams(monkeypatch):
    state = _fake_deltas(monkeypatch, ["", "Hello there"])
    chunks = list(chat.stream_text_chat_completion(object(), [{"role": "user", "content": "x"}], "gpt-6"))
    assert state["n"] == 2
    assert chat.collect_chat_content(chunks) == "Hello there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_text_chat_completion_raises_when_always_empty(monkeypatch):
    _fake_deltas(monkeypatch, [""])
    with pytest.raises(HTTPException) as excinfo:
        list(chat.stream_text_chat_completion(object(), [{"role": "user", "content": "x"}], "gpt-6"))
    assert excinfo.value.detail["error"]["code"] == "upstream_empty_response"


# --- 传输层瞬时失败也要重试 ------------------------------------------------

TRANSPORT_ERROR = "Failed to perform, curl: (28) Failed to connect to chatgpt.com port 443 after 30000 ms"


def _raising_then_ok(monkeypatch, failures: int, reply: str):
    state = {"n": 0}

    def fake(backend, request):
        index = state["n"]
        state["n"] += 1
        if index < failures:
            raise RuntimeError(TRANSPORT_ERROR)
        return iter([reply])

    monkeypatch.setattr(chat, "stream_text_deltas", fake)
    return state


def test_tool_turn_retries_transport_failure(monkeypatch):
    state = _raising_then_ok(monkeypatch, 1, JSON_REPLY)
    _text, calls = chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "git status"}


def test_tool_turn_reports_unreachable_after_all_transport_failures(monkeypatch):
    state = _raising_then_ok(monkeypatch, 99, JSON_REPLY)
    with pytest.raises(HTTPException) as excinfo:
        chat.tool_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "", [SHELL_TOOL])
    assert state["n"] == 3
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["error"]["code"] == "upstream_unreachable"


def test_plain_text_turn_retries_transport_failure(monkeypatch):
    state = _raising_then_ok(monkeypatch, 1, "PONG")
    assert chat.plain_text_turn(object(), [{"role": "user", "content": "x"}], "gpt-6", "") == "PONG"
    assert state["n"] == 2


def test_is_transport_failure_classification():
    assert chat._is_transport_failure(RuntimeError(TRANSPORT_ERROR)) is True
    assert chat._is_transport_failure(RuntimeError("curl: (7) Failed to connect")) is True
    assert chat._is_transport_failure(RuntimeError("curl: (60) SSL: no alternative certificate subject")) is True
    assert chat._is_transport_failure(RuntimeError("SSL handshake failed")) is True
    assert chat._is_transport_failure(RuntimeError("invalid api key")) is False
    assert chat._is_transport_failure(ValueError("bad json")) is False


def test_conversation_layer_shares_the_same_classifier():
    """账号池的 failover 与桥接层必须用同一套判定，否则 TLS 抖动仍会直接 502。"""
    from services.protocol.conversation import is_transport_failure

    assert is_transport_failure(RuntimeError("curl: (60) SSL: no alternative certificate subject")) is True
    assert is_transport_failure(RuntimeError("SSL connection could not be established")) is True
    assert is_transport_failure(RuntimeError("no available text account")) is False
    assert chat._is_transport_failure(RuntimeError("curl: (60) SSL")) is is_transport_failure(RuntimeError("curl: (60) SSL"))


def test_stream_text_chat_completion_without_tools_is_incremental(monkeypatch):
    monkeypatch.setattr(chat, "stream_text_deltas", lambda backend, request: iter(["Hello", " world"]))
    chunks = list(chat.stream_text_chat_completion(object(), [{"role": "user", "content": "x"}], "gpt-6"))
    assert chat.collect_chat_content(chunks) == "Hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# --- Anthropic /v1/messages 走同一套 JSON 信封 ----------------------------

ANTHROPIC_TOOL = {
    "name": "read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
}


def test_anthropic_prompt_lists_input_schema_and_json_envelope():
    prompt = bridge.tool_system_prompt([ANTHROPIC_TOOL])
    assert "file_path" in prompt
    assert '"tool_calls"' in prompt
    assert "<tool_calls>" not in prompt


def test_anthropic_content_blocks_parses_json_envelope():
    raw = json.dumps({"tool_calls": [{"name": "read", "arguments": {"file_path": "x.txt"}}]})
    content, stop_reason = anthropic.content_blocks(raw, [ANTHROPIC_TOOL])
    assert stop_reason == "tool_use"
    assert content[-1]["type"] == "tool_use"
    assert content[-1]["name"] == "read"
    assert content[-1]["input"] == {"file_path": "x.txt"}


def test_anthropic_content_blocks_keeps_plain_text():
    content, stop_reason = anthropic.content_blocks("All done.", [ANTHROPIC_TOOL])
    assert stop_reason == "end_turn"
    assert content == [{"type": "text", "text": "All done."}]


def test_anthropic_content_blocks_never_leaks_malformed_envelope():
    content, stop_reason = anthropic.content_blocks('{"tool_calls":[{"name":"read","arguments":{"file_path":"x"', [ANTHROPIC_TOOL])
    assert stop_reason == "end_turn"
    assert content == [{"type": "text", "text": ""}]


def test_anthropic_tool_use_history_is_serialized_as_json():
    block = anthropic._preprocess_block({"type": "tool_use", "name": "read", "input": {"file_path": "x"}}, lambda text: text)
    assert json.loads(block["text"]) == {"tool_calls": [{"name": "read", "arguments": {"file_path": "x"}}]}


def test_anthropic_streamable_prefix_stops_at_json_envelope():
    assert anthropic.streamable_prefix('Let me check.\n{"tool_calls":[') == "Let me check."
    assert anthropic.streamable_prefix("Just prose.") == "Just prose."


# --- 工具标记在「解析不出调用」时也不能泄漏成正文 ---------------------------
# 对应 Chat2API PR #147 的 bug 类：清理逻辑只在找到有效调用时运行，
# 于是未知工具名 / 被截断的标记原样留在 content 里发给客户端。

UNKNOWN_TOOL_XML = (
    "<tool_calls><tool_call><tool_name>run_shell</tool_name>"
    "<parameters><command>ls</command></parameters></tool_call></tool_calls>"
)


def test_unknown_tool_xml_never_leaks_as_text():
    text, calls = bridge.parse_reply(UNKNOWN_TOOL_XML, [READ_TOOL])
    assert calls == []
    assert text == ""


def test_unknown_tool_xml_keeps_surrounding_prose():
    text, calls = bridge.parse_reply(f"Checking now.\n{UNKNOWN_TOOL_XML}", [READ_TOOL])
    assert calls == []
    assert text == "Checking now."


def test_truncated_tool_markup_never_leaks_as_text():
    raw = "Checking.\n<tool_call><tool_name>read_file</tool_name><parameters><path>a"
    text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert calls == []
    assert "<tool_call>" not in text
    assert text == "Checking."


def test_truncated_tool_markup_is_not_a_malformed_envelope_after_cleaning():
    # 剥掉标记后已无可见正文，_needs_repair 会靠空正文触发纠正重试。
    raw = "<tool_call><tool_name>read_file</tool_name><parameters><path>a"
    assert bridge.clean_text(raw) == ""


def test_complete_markup_is_still_stripped_by_clean_text():
    assert bridge.clean_text(f"Prose.\n{UNKNOWN_TOOL_XML}") == "Prose."


# --- 双重编码的 arguments（回归锁） -----------------------------------------
# 模型常把 arguments 写成「JSON 字符串里的 JSON」（对照 qwen2API 的
# NormalizeToolInput）。_arguments_dict 会解一层字符串，这条锁住该行为。
# 不采用 qwen2API 的 `{"input": v}` 兜底：那会给我们造出不存在的参数名。

def test_double_encoded_arguments_are_decoded():
    raw = '{"tool_calls":[{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}]}'
    _text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}


def test_double_encoded_arguments_survive_validation():
    raw = '{"tool_calls":[{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}]}'
    _text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert bridge.calls_usable(calls, [READ_TOOL]) is True


def test_plain_arguments_are_unchanged():
    raw = '{"tool_calls":[{"name":"read_file","arguments":{"path":"a.txt"}}]}'
    _text, calls = bridge.parse_reply(raw, [READ_TOOL])
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}



