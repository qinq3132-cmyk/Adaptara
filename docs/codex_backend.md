# AI Scientist 的 Codex 后端

项目通过独立 `codex exec` 子进程执行科研角色。当前默认配置为：

- 模型：`glm-5.3-flash`，网关：`http://39.105.190.188:3456/v1`。
- 密钥来源：OpenCode 的 `Lemon-Domestic` provider。不复制到 YAML、命令参数或研究产物。
- 认证：只给子进程设置 `SCIENTIST_CODEX_GATEWAY_KEY`，通过自定义 provider 的 `env_key` 使用。
- `requires_openai_auth=false`，避免使用全局 ChatGPT 登录；这是与官方登录并存的关键。
- `--ignore-user-config --ephemeral`，不修改全局 `~/.codex/config.toml` 或 `auth.json`。
- 假设角色关闭 shell、执行、应用、插件、hooks 和多 agent 功能。
- 本地阅读角色开启只读 shell；命令环境采用 `inherit=core` 并排除密钥变量。

全局配置中的 `requires_openai_auth=true` 适用于 Codex 自己的 API-key 登录路径。
用户希望继续使用官方 ChatGPT 登录，因此本项目默认采用独立 `env_key` 认证，而不运行 `codex login`。
可选 `codex_auth_mode: codex` 支持已有 API-key `auth.json` 的环境；它不会自动写入认证文件。

## 使用

从项目根目录执行：

```bash
python run_scientist.py --check-backend
python run_scientist.py --generate-hypotheses
python run_scientist.py --read-document examples/document_reading_smoke.md
```

`--generate-hypotheses` 仍通过 Python 校验 JSON、分配 H 编号并保存新的科学状态。
它不自动执行 E001/E002。后端失败时不会用 mock 内容替代，也不会提交科学状态。

`--read-document` 是本地文档阅读入口，支持 1–32000 字节的 UTF-8 `.txt/.md`。
文件由 Python 复制到临时目录，内容由 Codex 的 shell 工具读取，模型返回中文结构化摘要。
失败的工具调用或没有工具调用的摘要均不会保存为成功结果。
报告保存原文、SHA256、摘要与运行元数据，并生成校验清单。
还检查工具输出是否包含完整原文，避免把仅运行了无关命令视为已读文件。
这是文档阅读基础能力，尚未实现检索、PDF/OCR、文献拆分、来源核查或自动写入 Paper/Claim。

`--check-backend` 测试沙箱启动、结构化输出和“读文件 → 工具结果 → 模型最终回答”。
诊断报告在 `artifacts/backend_check_<uuid>/`，不创建科研实验记录。
仅 `/models` 返回成功不足以证明 Codex 兼容性。

保留 OpenCode 后端：

```bash
python run_scientist.py --generate-hypotheses --backend opencode --model Lemon-Domestic/glm-5.3
```

Codex 的可执行文件可通过 `--codex-bin` 覆盖。项目 `config/models.yaml` 指定的是此机器
已安装的独立 Codex 0.150.1，不替换 IDE 使用的 Codex。迁移机器时应调整此路径。
`codex_model_catalog` 可指向服务商提供的模型元数据，别名需与模型 ID 匹配；不自动假造元数据。

## 本机沙箱问题及解决办法

已测试的环境事实：

1. IDE 的 Codex `0.162.0-alpha.2`，默认只读沙箱报错：
   `bwrap: No permissions to create a new namespace`。
2. `/proc/sys/kernel/unprivileged_userns_clone` 为 `1`，`max_user_namespaces` 为 `8254889`，
   但 `unshare -Ur /bin/true` 仍报 `Operation not permitted`，进程 `Seccomp: 2`。
   这表明存在容器或宿主的额外限制；仅修改该 sysctl 不能解决已观测问题。
   当前容器内无法进一步确定是 seccomp、AppArmor 还是其他宿主策略，需查宿主日志/容器启动配置。
3. IDE 版本的 `use_legacy_landlock` 仍报：
   `filesystem-restricted execution requires bubblewrap to isolate app-server sockets`。
4. 本机独立 Codex `0.150.1` 配合 `use_legacy_landlock=true` 通过只读沙箱启动及真实文件读取。
   本项目明确选择这个安装，并只对其子进程开启 Landlock；未关闭沙箱或改动宿主配置。

当前可用方案是第 4 项。Landlock 开关已被标记 deprecated，未来版本可能移除；
不要假设任何旧版本或新版本都有相同兼容性，升级后先运行 `--check-backend`。

长期方案：在支持 Codex bubblewrap 的宿主或容器中运行，并检查 seccomp/AppArmor
对创建 user/mount/network namespaces 的限制。由容器维护者调整必要的隔离策略，
在同一运行用户下确认 `unshare -Ur /bin/true` 与 `codex sandbox -- /bin/true` 均成功。
随后设置 `codex_sandbox_backend: default`，选择目标 Codex 安装，再验证文件工具往返。

## API 兼容性与诊断

网关需要 Responses API 的 SSE 完成事件及 function call/result 续接。
仅支持 Chat Completions 的服务不能直接用于当前 Codex。
实测 `glm-5.3-flash` 偶尔在指定 `--output-schema` 后仍返回 Markdown 包装的 JSON。
程序只移除一个完整的 `json` 代码块，再严格校验内容；不依赖网关强制执行 schema。
读取请求也在 prompt 中明确三个必填字段及类型。文档输出无效时按 `validation_retries`
有界重试，所有尝试记录在成功报告中；最终不合格时不保存摘要，工具/网络错误不重试。

诊断代码与处理：

| 代码 | 含义与处理 |
| --- | --- |
| `credential_missing` | OpenCode 未配置 API key；或主动选择原生认证但 `auth.json` 仅有 ChatGPT 登录。保持原登录，优先使用默认 OpenCode 认证桥接。 |
| `provider_auth_failed` | API key 被拒绝，检查网关密钥及权限。 |
| `responses_endpoint_missing` | 检查 `/v1/responses` 路由，Chat Completions 兼容性不能替代。 |
| `sandbox_namespace_denied` | 容器限制命名空间，采用上面已验证的独立安装或调整宿主隔离。 |
| `landlock_incompatible` | 该 Codex 版本即使开启 Landlock 也要求 bubblewrap。 |
| `tool_read_not_verified` | 模型未成功使用文件工具，不能当成已读取文档。 |
| `runtime_or_provider_failed` | 检查流式/结构化输出/工具协议、代理与 Codex 版本；原始 provider 错误不直接输出，以免泄漏凭据。 |
| `timeout` | 整个调用及其工具进程组终止，不提交科研状态。 |

并非所有接入模型都具备相同的规划或工具使用能力。运行后端支持工具，不保证每个模型都能可靠使用工具。

官方参考：[非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode)、
[网关兼容性](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility)、
[自定义 provider](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers)、
[沙箱](https://learn.chatgpt.com/docs/sandboxing)。
