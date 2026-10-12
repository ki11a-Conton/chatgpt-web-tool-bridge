# ChatGPT 网页端 → 本地 Agent 工具调用桥

把 ChatGPT 网页版（free 号）逆向成一个 **OpenAI 兼容接口**，让本地 Agent（DeepSeek Harness、Claude Code、Cline 等）能通过它**真正调用本机工具**（shell / 文件读写）。

上游**没有原生 `tools` 通道**，收到 `role:"tool"` 直接 500。所以整套工具调用是**纯提示词协议 + 双向翻译**实现的。

---

## 1. 为什么需要这一层

ChatGPT 网页后端（`chatgpt.com/backend-api`）和 OpenAI 官方 API 是两套东西：

| | 官方 API | 网页后端 |
|---|---|---|
| 工具调用 | 原生 `tools` 字段 | **没有** |
| 历史里的 `role:"tool"` | 支持 | **500 报错** |
| 流式 | SSE 增量 | SSE，但常一次性给完整消息 |
| 认证 | API Key | Access Token + Sentinel 三重校验 |

Agent 发过来的请求里有 `tools` 和 `role:"tool"`，网页后端全都不认。**桥接层要做的事就是双向翻译**：

```
Agent ──tools──▶ 桥接层 ──提示词注入──▶ ChatGPT 网页后端
                    ▲                        │
                    └──── 解析 JSON 信封 ─────┘
Agent ◀──tool_calls──┘
```

---

## 2. 核心设计：严格 JSON 信封

### 请求方向

`to_upstream_messages()` 把 OpenAI 格式的历史折成上游能吃的纯文本：

- `tools` 数组 → 压缩成 `AVAILABLE_TOOLS` 文本块，插到 system 最前面
- 历史里 assistant 的 `tool_calls` → `<tool_calls>` XML 文本（**上游只认文本**）
- 历史里 `role:"tool"` 的结果 → 折成 `role:"user"` 的 `Tool result ...:` 文本

工具多（>8 个）或 schema 大（>4000 字符）时**自动压缩**成 `name(param:type, opt:type?) - desc` 一行式，否则提示词会撑爆上下文。

### 响应方向

模型被要求**只输出一个 JSON 对象**：

```json
{"tool_calls":[{"name":"pwsh","arguments":{"command":"git status","description":"..."}}]}
```

`parse_reply()` 解析它，转成标准 `tool_calls` 返回给 Agent。

**多级兜底**（按优先级）：

1. 严格 JSON 信封（主路径）
2. XML `<tool_calls>` 标记（兼容模型跑偏）
3. 上游沙箱回声（见 §4）

### 三条安全约束

```python
validate_calls()        # 只保留本次请求声明过的工具名，未知名字直接丢弃
strip_extra_arguments() # schema 外的参数一律剥掉
calls_destructive()     # 整文件重写式的 edit 调用直接拒绝
```

**未知工具名绝不转发给 Agent** —— 否则模型幻觉出的 `rm_rf` 之类会真的被执行。

---

## 3. 工具轮为什么要「缓冲整轮」

普通聊天是**增量 SSE**（边生成边推），但**带工具的轮次必须缓冲整轮**：

- 工具调用必须**完整**才能解析，半截 JSON 毫无价值
- 解析失败要能**带原因重试**，而重试必须发生在**发出任何字节之前**
- 一旦先推了正文再发现是工具调用，客户端已经拿到错误内容了

所以 `tool_turn()` 的逻辑是：

```
上游整轮 ──▶ 解析 ──▶ 可用？──▶ 是 ──▶ 一次性发假 SSE
                │
                └─ 否 ──▶ 带原因纠正重试（最多 3 次）
                            │
                            └─ 全失败 ──▶ 503 + Retry-After: 20
```

重试时把失败原因明确告诉模型（`unknown tool name(s): xxx`、`the JSON was truncated or malformed`、`required parameters were missing`），比空泛地说"格式错了"有效得多。

**工具轮永远返回 `content: ""`**：模型常在给出可用调用的同时附带"已完成/给你下载链接"之类误导性文字，客户端会照发。有调用就一律不带正文。

---

## 4. 上游在它自己的沙箱里执行命令

这是最反直觉的一点：**ChatGPT 网页端会把 `bash -lc <cmd>` 真的跑在它自己的 Linux 容器里**，然后把回声漏进文本通道。

实测两次独立调用返回了**不同的 hostname**（`41d1db925e3e` / `d6b072379071`），但内核、uid、`/` 相同 —— 确认是它自己的沙箱。

对本地 Agent 来说这是**有害**的：Agent 以为命令在自己机器上跑了，实际跑在 OpenAI 的容器里。

处理方式：识别回声 → 转成**真实的工具调用**（把命令原样交给客户端执行），纯回声则判空触发重试。

---

## 5. 关键坑（都是实测踩出来的）

### 5.1 中文拒答识别

模型会说「我目前**无法**在这个会话中调用可用的本地文件写入工具」。早期拒答检测**全是英文正则**，这些被当正常回答返回了。已补中文模式，且确保不误伤「服务暂时不可用」这类正常回答。

### 5.2 工具标记截断会泄漏成正文

模型 XML 写一半被截断（流被掐 / 输出超限）时，**半截标记会原样发给客户端**。`_MARKUP_PATTERN` 只匹配**闭合**的块，未闭合的尾巴全漏。修法是再加一条 `_MARKUP_TAIL_PATTERN`，从第一个标记起截掉。

> 这个 bug 类来自 Chat2API PR #147。

### 5.3 别照抄别的项目的提示词规则

试过加「禁止使用 XML / `<tool_call>` / `[function_calls]` 等其它格式」（Chat2API #100 的建议），**行为验收从 10/10 掉到 6/10** —— 因为我们**自己就用 `<tool_call>` XML 序列化历史**，规则自相矛盾把模型搞懵了。

**跨项目的提示词规则不能照抄**，它们的前提和你的上下文不同。

### 5.4 推理块剥离是多余的

上游把 CoT 放在**独立的 `reasoning_recap` 消息**里（`cot_version: v5`，`reasoning_recap_type: hide_all`），而取正文的 `assistant_message_text` **只读 `content.parts`** —— 那条消息的 `content` 是字符串、没有 `parts`，**结构上就进不了正文通道**。实测模型 6/6 不发 `<think>` 标签。

### 5.5 sentinel token 是一次性的

实测复用同一个 sentinel token 连发：第 1 轮 200，第 2/3 轮 **403**。所以每轮都必须重新握手，**这部分无法压缩**。

### 5.6 bootstrap 可以缓存

每轮 `_bootstrap()` 都 GET 一次首页，只为拿 PoW 的脚本指纹。实测该指纹**跨账号完全一致**（3 个账号同为 `prod-d979090332d41904b008e` + 3 个脚本），跳过它 5/5 对话成功。改成进程级缓存后 **355ms → 0ms**（首轮付一次），带 6 小时 TTL（上游部署会换 build 哈希）。

---

## 6. 性能实测

### 每轮耗时构成

```
bootstrap (GET /)              355 ms   ← 已缓存到 ~0
sentinel prepare + finalize   1178 ms   ← 上游安全握手，省不掉
  ├─ prepare  (HTTP)           240 ms
  ├─ finalize (HTTP)           453 ms
  ├─ PoW 求解                    1 ms   ← 几乎免费
  └─ turnstile 求解              0 ms
POST 到首字节 (TTFB)          2202 ms   ← 模型思考，上游决定
整轮                          3919 ms
```

**PoW 只要 1 毫秒** —— 瓶颈不是计算，是那两次网络往返（纯 RTT 就 208ms）。

### 网络 RTT 基线

```
GET /                 157 ms
sentinel prepare      225 ms
```

### 每轮重建 session 的代价

```
每次新建 requests.Session:  396 ms
复用同一 session:           236 ms   ← 可省 ~160 ms/轮
```

**未采用**：session 与 access_token 绑定，复用需处理"换号时换 session"，改错会让整个号池轮换失效。收益 160ms 不值这个风险。

---

## 7. 文件清单

### 核心（我写的）

| 文件 | 作用 |
|---|---|
| `services/protocol/openai_tool_bridge.py` | **桥接层主体**：提示词构造、JSON 信封解析、校验、纠错重试、噪声清理 |
| `services/protocol/tool_markup.py` | XML 协议唯一定义处：解析 + 剥离（含截断尾巴处理） |

### 改动过的协议层

| 文件 | 改动 |
|---|---|
| `services/protocol/openai_v1_chat_complete.py` | `tool_turn()` 缓冲重试、假 SSE、错误分类（空回复/坏信封/不可达） |
| `services/protocol/anthropic_v1_messages.py` | 切到 JSON 信封，复用同一套解析 |
| `services/protocol/conversation.py` | 传输层失败识别 + 号池故障转移 |
| `services/openai_backend_api.py` | bootstrap 指纹缓存 |

### 上游客户端与支撑

`utils/pow.py`（PoW 求解）、`utils/turnstile.py`、`utils/helper.py`（SSE 解析）

### 测试与验证

- `tests/test_openai_tool_bridge.py` — 工具桥接全量测试
- `tests/test_pow_fingerprint_cache.py` — bootstrap 缓存
- `probes/` — 真实端到端验证脚本

---

## 8. 验证方式

```bash
# 单元测试
python -m pytest tests/test_openai_tool_bridge.py tests/test_pow_fingerprint_cache.py -q

# 行为验收（12 工具 + 真实 DSH system prompt，10 个任务）
python probes/behaviour_probe.py

# PPT 闭环（skill → write → pwsh → 真实落盘）
python probes/ppt_loop_probe.py

# 延迟拆解
python probes/latency_probe.py 3
```

行为验收的判定标准：**OK / MISS / BAD** 三类，其中 `BAD` 指选了一个**说得通但不是预期**的工具（例如用 `pwsh` 去改文件）。

---

## 9. 已知限制

- **模型会偶发选错工具**：把 `edit` 选成 `pwsh`（用 shell 改文件是个说得通的替代方案）。实测波动 8~10/10，通常下一轮自愈。**没有加启发式去拦** —— 那会过拟合。这个问题在 Chat2API、Toolify、CatGPT-Gateway 里同样存在，没人解决。
- **PoW 实现**：用 SHA3-512 + base64 编码 JSON，比网上流传的 `SHA-256(seed+counter)` 简化描述更接近真实算法。
- **账号要求**：free 号可用（走不了官方 Secure MCP Tunnel 那条路，那需要 Plus + 开发者模式）。
- **依赖**：本层不自带 HTTP 客户端，复用宿主仓库的 `curl_cffi` session 与账号池。

---

## 10. 未打包的依赖

为控制体积，以下**未包含**（需要时从原仓库取）：

- `services/account_service.py` — 账号池（AT/RT/session 生命周期、号池轮换）
- `services/image_failure.py` — 图片链路失败分类（`terminal_assistant_text` 在这里）
- `services/proxy_service.py` — 代理配置
- `utils/log.py`、`utils/diagnostics.py` — 日志与诊断

所以 `services/` 下的文件**不能直接独立运行**，它们是嵌入宿主仓库工作的。包内代码的价值在于**协议实现与踩坑结论**，不是开箱即用。
