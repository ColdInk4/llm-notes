# AGENTS.md

本文件适用于整个 `llm-notes` 仓库。所有维护者和自动化 coding agent 在修改前都应先阅读这里的说明，并以 `STYLE.md` 的写作与排版约定为准。

AGENTS.md 是**流程规范**：回答前复核、修改前必读、审计循环、联网求证门槛、commit 约定。STYLE.md 是**写作规范**：Markdown 排版、提示块用法、术语口径、来源记录格式。两份文档在以下三处必须保持一致：(1) 禁用句式清单；(2) 提示块 5 种 alert 语义；(3) 章节末来源记录的 4 段结构。修改 `AGENTS.md` 后须 `cp AGENTS.md CLAUDE.md` 一并提交。

## 回答前复核

- 回答涉及仓库正文内容、章节技术解释、来源判断、修改建议或任何形式的审稿 / 审计 / fresh audit / Phase 1 finding 时，**必须**先完成至少一次相关网络搜索（WebSearch + WebFetch）。广查 / 盲查 / 章节级复核即使尚未动手修改任何文件，也属于本条触发范围；不只在实际改文件时才搜索。
- 联网求证后再结合本地目标章节、相邻小节和来源记录回答。若无法完成网络搜索（断网、API 限流、sandbox 拦截等），必须显式在回复中说明原因、影响范围和后续复核建议，并把相关 finding 标记为 `tentative`。
- 仅处理**纯本地机械问题**可以不联网：例如列文件、查看当前 diff、运行已有检查命令（rg / wc / git status）、读本地 PDF 抽出的纯文本。一旦涉及技术事实、外部来源、章节判断、来源映射、引用准确度或 sub-agent 给出的 verdict，仍按上一条执行。

## Token 不是停止工作的理由

在 ultracode 模式（`/effort` 设 xhigh + dynamic workflow orchestration）或任何明示"穷尽/完整"目标的会话下，**token 预算不是约束**。审计、Phase 1~3 finding、sub-agent 并行广查与 WebFetch/WebSearch 调用应做到：

- 计划阶段按 `Chapter × 问题域 × 一手源` 估出最大并行度，而不是先按"剩余 token 还够几次"压规模
- finding 数量大时**继续**分批 WebFetch 复核而不是停在"已经修了 14 处"的 commit 边界
- "受 token 预算约束未启动 Phase X / 章节 Y" 这种措辞**禁止**作为延期理由；要么启动，要么显式说明是"用户授权分批"或"技术失败"

报告里写"剩余 token 不够"等同于默认承认偷懒。正确说法要么是"已完成 X / Y / Z，未做 W 留到下一轮"，要么是"现在启动 Phase X"。本仓库维护目标是把每一轮能修的真错都修完，不是把"修了 14 条"包装成完整闭环。

## 修改前必读

- 每次修改任何文件前，**必须**先完成一次相关网络搜索（WebSearch + WebFetch），并阅读 `STYLE.md`，确认 Markdown、公式、图片、提示块、来源日期和文件命名约定。
- 再阅读 `README.md`，理解仓库定位、阅读入口、章节组织方式和维护说明。
- 修改具体章节前，先阅读目标文件及其相邻小节；如果章节末有来源记录，也要一起检查。
- 修改 CS336 相关内容前，阅读 `sources/cs336-2026.md` 中对应 lecture、slides、video 和论文映射。
- 涉及具体模型、硬件、benchmark、榜单、厂商声明等外部事实时，先定位最相关的章节或专题，并按 `STYLE.md` 的来源与日期约定记录。涉及 arXiv 编号、HF model id、官方文档链接时，**必须**先 WebFetch 一手页面验证标题/作者/数字，不允许凭印象写入。

## 修改后自检

修改 Markdown 正文后，必须对本次修改过的文件跑**三类自检**，每类命中即改写：

### 1. 禁用句式 rg（与 STYLE.md 同步）

```bash
rg -n "读法|看这类图时|工程读法|这张图告诉我们|当作起点|当成定律|不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是|未经一手|建议引用|待.*核对|缺口|以.*为准|本节以|属讲师口误|未能核|课件 Lecture|课件里|课件中|slides 第|字幕原话|视频中说|Lecture \d+ 字幕|讲者已在|笔记此前" <modified-markdown-files>
```

命中正文（章节末"来源与更新记录"除外）即改写为正向讲义表达：直接说明变量关系、成立条件、边界和工程后果。

### 2. 作者声音穿帮 rg（与 STYLE.md「修辞层规则」同步）

```bash
rg -n "这里使用.{1,30}(作为|当|来当|样例|为例)|本文采用|本文以.*为例|后续章节会|后文将|后文会|后面单独讨论|下一节会|下一节展开|下一节转到|不等于所有|不是说|并非主张|并不意味着|但愿不要|读者不要把本文|读者应以|本节以" <modified-markdown-files>
```

抓作者选择、免责声明、写作计划承诺漏到正文。命中即下沉到章节末来源记录或改写为正向事实陈述（如「§5.8 单列 B200 / Blackwell 的差异」）。注意 `下一节展开` / `下一节转到` 也属写作计划承诺，不只 `后文会` / `后续章节会`。

### 3. 提示块类型合理性（与 STYLE.md「提示块」同步）

如果改动涉及 alert 类型，对照 STYLE.md 的 5 种 alert 语义边界（NOTE / TIP / IMPORTANT / WARNING / CAUTION）核对用法是否合理——NOTE 不应承载作者声明或免责声明；IMPORTANT / WARNING / CAUTION 三档严格递进。

讲义正文**不要**出现下列元叙述（与 STYLE.md「修辞层规则」对齐）：

- "建议引用时直接查证 X"
- "以 Y 为准 / 未经 X 验证"
- "待 Y 披露后核对"
- "本节以 X 为准 / 属讲师口误"
- "这里使用 / 本文采用 / 后续章节会 / 后文将 / 不等于所有"

这类语句把审稿过程带到读者面前，破坏讲义"按权威源直接陈述"的风格。**正确做法**：要么按权威源给出事实（数字 + 出处），要么省略该数据点；来源记录统一写章节末"来源与更新记录"或来源映射区，正文不写 disclaimer。

## 审计与维护方法论

### 联网求证门槛（hard floor）

任何形式的"广查 / 盲查 / fresh audit / 三视角循环 / Phase 1~3 finding 收集"，**即使尚未动手修改任何文件**，按以下最小门槛执行：

1. 每个 sub-agent 在出 finding 之前，**必须**对涉及外部事实的部分（arXiv 编号、模型/硬件数字、benchmark 分数、版本号、日期、厂商声明、引用文献、HF model id）至少调用一次 WebSearch，并 WebFetch 一手页面（arXiv abstract / 官方 model card / HuggingFace `config.json` / 官方仓库 README / 技术报告 PDF）做实际验证。**不得**仅凭印象、训练数据记忆或课件文字给出 verdict。
2. finding 列表里每一条 `severity ≥ medium` 或 `needs_web_check=true` 的项目，必须在 evidence 字段附上：
   - 实际 WebSearch 查询字符串或 WebFetch URL；
   - 一手页面里的相关原文/数字摘录（≤ 2 行）；
   - 笔记正文里被质疑的那句话或数字（含 file:line）。
3. 没有附上述证据的 finding 视为 `tentative`；Phase 2 求证阶段按主题聚类 WebFetch 核证，仍无法被一手源支撑的应当降级或丢弃，不得计入最终 fix 列表。
4. sub-agent "建议加 X 链接 / 标注 Y 来源" 的建议，主 agent 在 commit 前**必须**自己再 WebFetch 核一次；历史经验：sub-agent 写出来的参考文献可用率约 70-80%，主 agent 必须复核。
5. 历史教训：曾出现 arXiv:1712.00409（实为 Hestness 等 Baidu 论文，并非"44× ImageNet 出处"）和 arXiv:2401.06865（实为天文学家 Osborne & Salim 的星系 SED 拟合论文）两例 sub-agent 幻觉引用。**任何 arXiv 编号进笔记前必须先 WebFetch abstract 页面确认标题与作者**，不接受"看起来对题"。

### 工具用法（WebSearch + WebFetch 并用）

- **WebSearch 与 WebFetch 必须并用**——只 WebFetch 不 WebSearch 是 sub-agent 的常见偷懒模式。每轮广查 / 审计每个 sub-agent 的硬性最小值：
  - WebSearch ≥ 5 次，覆盖"是不是真存在 / 是不是真作者 / 是不是真数字"三种目的
  - WebFetch ≥ 5 个不同 URL，其中至少 2 个为 `arxiv.org/pdf/{id}` 或 `/html/{id}` 的正文（非 abstract）
  - finding 的 `web_evidence.search_query` 必须实际跑过；不能写"未跑"
- WebSearch 关键词模板（必须包含具体名词）：
  - arXiv 复核：`"Hestness 2017 arxiv 1712.00409 ImageNet"` / `"Rafailov DPO arxiv 2305.18290 authors"`
  - 模型配置复核：`"LLaMA-2 7B num_key_value_heads config.json GQA"` / `"Qwen3-235B-A22B vocab_size huggingface"`
  - GPU 规格复核：`"H100 SXM HBM3 bandwidth 3.35 TB/s spec sheet"` / `"B200 SM count 144 GB100 whitepaper"`
  - benchmark 复核：`"Cybench first solve time arXiv 2408.08926 24h54m"` / `"MMLU-Pro 10 options CoT 33% arxiv 2406.01574"`
- WebFetch 优先 URL：`arxiv.org/abs/{id}`（标题/作者/日期）→ `arxiv.org/pdf/{id}` 或 `/html/{id}`（正文 Figure/Table）→ `huggingface.co/{org}/{model}/blob/main/config.json` → 官方仓库 README → 技术报告 PDF。
- 一次 WebFetch 拿不到信息就换关键词再 WebSearch；连续两次失败应切换一手 URL 而不是堆 WebSearch 数量。
- WebSearch/WebFetch 触达受 sandbox 限制时，回到「回答前复核」的降级路径：显式声明、影响范围、标记 `tentative`。

### 必须读原论文，不能只读 abstract

arXiv `abs/{id}` 页只有 abstract 与 metadata，绝大多数具体数字（样本量、训练 token 数、benchmark 分数、超参数表、Figure/Table 数据）都只在 PDF / HTML 正文里。任何 arXiv 引用进笔记前，WebFetch 的最小集是：

1. `arxiv.org/abs/{id}` 确认标题、作者、提交日期；
2. `arxiv.org/pdf/{id}` 或 `arxiv.org/html/{id}` 拿正文关键 Figure / Table 数字与公式；
3. 引用该论文的下游论文 / 官方 model card / 榜单页做交叉核对。

仅基于 abstract 出 finding 时，必须在 finding 证据字段显式标记 `source=abstract-only` 并降级为 `tentative`；不允许把 `abs/{id}` 的内容当成"已验证"结论写入 fix。

### sub-agent 必读课件与必读图

每条 finding 出 verdict 之前**必须**先读以下资料源：

1. 章节文件全文（含上一节 / 下一节）
2. 对应 lecture PDF 抽文 `sources/_extracted_pdfs/lecture_{NN}.txt`（权威技术内容来源）
3. 对应 lecture Python 代码讲义 `sources/lectures/lecture_{NN}.py`（代码级事实、reproducible trace 来源）
4. 章节末"来源与更新记录"与 `sources/cs336-2026.md` 里的 lecture 映射
5. **章节引用的所有本地 PNG**（用 `Read` 工具读图，agent 必须实际看图意，不是看 alt text 或文件名）

字幕（`sources/captions/lecture_{NN}.md`）只作为 logical flow 与上下文承接，不作为技术事实权威——字幕包含口误 / 转录错误，技术数字与公式必须以 lecture PDF 原文 + lecture_NN.py 代码为准。

finding 的 `evidence_in_lecture` 字段必须引用读到的具体原文段落或代码行（≤ 2 行）。**WebSearch + 读课件 + 读图三件并用**：WebSearch 解决"是不是真存在 / 数字精度"；读 lecture PDF + lecture_NN.py 解决"笔记 vs 课程口径是否一致"；读 PNG 解决"图意是否与正文描述一致"。只跑其中一两件都算 audit 不完整。

### 审计的 7 层覆盖（rg 是 sanity check，不是 audit 主体）

sub-agent 的审计主体是「通读全文 + 读本地 PNG + WebFetch 一手页面 + 跨章节对照」，`rg` 只在 commit 前做最后一遍机械兜底。审计范围覆盖 **7 个层面**，每个层面都必须由 agent 自己理解得出，不允许「rg 没命中就算 audit 通过」。`rg` 仅用于第 4 层「段落 / 修辞层」与第 7 层「跨章引用格式」。

**A. 章节内部**

- **1. 事实层**：arXiv 编号存在性 / 标题 / 作者 / 提交日期；模型 / 硬件 / benchmark 数字与一手源一致；论文 § 编号归属正确（§3.2 不写成 §2.1）；人名 / 公司名 / 法官名 / 论文首作者正确；论文方向 / 结论读反（如 Santurkar RLHF 反向）。
- **2. 公式账本层**：符号首次出现即定义（`$N_{\mathrm{param}}$` / `$B$` / `$d_{\mathrm{model}}$` / `$h$` / `$K_{\mathrm{kv}}$` / `$\hat{A}$` 等）；公式与代码示例（`lecture_NN.py`）数字一致；公式后立刻给承接句；长算式拆 `aligned` / 多行 display math；公式与表格算术链一致（Korthikanti 10 = 4+2+4）；不出现 snake_case 标识符。
- **3. 图意层**：核心图必读本地 PNG（不只看 alt text）；子图标题 / 坐标轴 / 图例被正确转述；正文数字与图上标注一致（>0.5 倍差距必核）；图说「机制解释」能在图中找到对应视觉证据；删图判定（两图承载同一份视觉信息即使 SHA 不同）；删图后按连号规则重排，跑 `rg 图 N-x` 验证无悬空引用。
- **4. 段落 / 修辞层**：长段 > 8 行拆段；单段承载动作 > 2 个拆段；公式 / 表格 / 代码块后承接句；作者声音穿帮（"这里使用 / 本文采用 / 后文会 / 不等于所有 / 下一节展开"等）；禁用句式（"读法 / 看这类图时 / 不是 X 而是 Y"）；提示块类型合理性（5 种 alert 语义边界）。

**B. 跨章节**

- **5. 术语 / 符号层**：同一概念在不同章节用同一英文术语；同一符号在不同章节指同一变量（ch3 `$d_k$/$h$` vs ch2 `$H$/$N$` 必须统一）。
- **6. 数字层**：同一硬件 / 模型 / benchmark 跨章一致（H100 BF16 dense 989.5 TFLOP/s、MMLU 12,032 题、Pile 334B tokens）；跨章引用的数字与那一章原文一致（Llama 3 70B tokens/param 215 vs ch10 15.6T → 223）；章节末「来源与更新记录」4 段结构（官方 / 课程 / 事实声明指向 / 不确定项）。
- **7. 引用 / 跨章层**：跨章引用格式统一（第 N 章 §X.Y 章节标题）；章节内 caption / 图注 / 正文交叉引用一致；别章引用的章节标题与目标章节实际 H2/H3 一致。

每层对应 sub-agent 工作流：

| 层 | 工具 | 硬约束 |
|---|---|---|
| 1 / 6 | WebSearch ×5 + WebFetch ×5 | `search_query_1/2/3` 三字段不同关键词 |
| 2 / 5 | Read 章节全文 + lecture_NN.py | 符号首次出现即定义 |
| 3 | Read 本地 PNG | 核心图必读，不看 alt text |
| 4 | rg 三类自检 | 禁用句式 / 作者声音 / 提示块 |
| 5 / 7 | Read 相邻小节 + 跨章引用目标章节 | 跨章引用格式「第 N 章 §X.Y 章节标题」 |
| 6 | 引用的别章 § 编号 + 章节号 | 与目标章原文核对 |
| 1-7 | 当场 Edit + 回读上下 5 行 | sibling 自洽检查 |

`introduced_by_commit` 字段是 fix verification 阶段的硬约束，跨章节 audit 必须用简化 schema（参见「工具使用经验」条目）。

### 三视角循环（Phase 1 / 2 / 3）

- **Phase 1（盲查）**——按 lecture 切分，sub-agent 报告 finding 列表（含 file、line、issue_type、severity、evidence、suggestion、web_evidence）。**每个 sub-agent 至少调用一次 WebSearch + WebFetch + Read PNG + 跨章节 Read**，对涉及外部事实的 finding 在 evidence 字段附 WebSearch 查询字符串 / WebFetch URL + 一手页面原文摘录（≤ 2 行）。
- **Phase 2（求证）**——仅针对 `needs_web_check=true` 或 `tentative` 的 finding 联网核证；可按主题聚类用 8-10 个 agent 并行 WebFetch / WebSearch。无法被一手源支撑的 finding 必须降级或丢弃。
- **Phase 3（章节级复核 fresh audit）**——从零重新读当前文件 + 对应课件 + 引用 PNG + 跨章节对照做 fresh audit，让每个章节独立得出 verdict；同样必须联网求证外部事实。这一轮最易发现前两轮漏掉的"原始事实"类错误（如 GPU 规格、KV cache 大小、样本数、vocab_size、训练 token 数）。**Phase 3 不传 Phase 1 finding 列表**，避免 anchoring bias。

**审计 verdict 三档规范（推荐默认 schema）**：

- `confirmed` — 笔记对得上权威源，**没有问题**，**不报**
- `refuted` — 笔记与权威源**直接冲突，是错的**，要修
- `tentative` — 一手源证据不足，可能是对也可能是错，留到下轮

prompt 要求 sub-agent 只报 `refuted + tentative`、**不再报 `confirmed`**，可显著减少低信号 confirmed informational finding，把篇幅留给可执行的 refuted 修复。

### 审计-fix 一体化（方案 A）

每个章节的 audit 与 fix 由**同一个 sub-agent** 在两次明确分开的 Phase 内完成，避免跨 agent 印象传染：

- **Phase 1（audit）**：读章节全文 + lecture PDF + lecture_NN.py + 引用 PNG + 跨章节对照（按上方硬约束）；WebFetch 一手页面（论文 PDF 原文 / HF model card / NVIDIA datasheet / config.json）；产出 finding 列表，**只列 wrong_facts 与 evidence_in_lecture，不写 suggested_fix**——避免 audit 的修复方案成为 fix agent 的"印象传染源"。
- **Phase 2（fix）**：每个 fix **必须重新 WebFetch 一手页面（不同 URL 或不同 search_query），不得复用 Phase 1 的 WebFetch 结果**。先读 lecture PDF + lecture_NN.py 原文与代码，**确认课件口径与一手页面一致后再写修复**。写完一个 fix 后**回读 fix 上下 5 行做上下文自洽检查**（sibling rows / sibling claims 是否仍成立），不一致则继续修。
- **不在 finding 里写 suggested_fix**：fix agent 不抄 audit 的修复建议，只看 wrong_facts + evidence_in_lecture + 一手 URL，自行 WebFetch 重新核验后写修复。这条切断"修复源头已污染"的链条——之前多次翻车的 Bartz 法官 / LIMO 800 vs 817 / LLaDA 2.0 "Tiwei Bie" 都是 audit 给的 suggested_fix 已经写错而被 fix 直接抄。
- **audit 与 fix 写在一个 agent 任务里**：避免 context 在两个 agent 之间序列化丢失；同一 agent 在 Phase 2 重新 WebFetch，避免 anchoring bias；Phase 1 与 Phase 2 之间必须**有清晰的任务结构与字段隔离**，agent 不能跳过 Phase 1 直接 Phase 2。

### fix verification round（次生复核）

单轮 audit + 单轮 fix 抓不住次生错误。曾出现 11 处 fix 留下的次生错误（Judge Illston / UltraFeedback 2310.01377 / Tulu 3 2411.15124 / SigLIP 5 天 / Mixtral TP=2 / DeepSeek-V4 残留等）直到下一轮 fix verification 才被抓到。

**推荐节奏**：每次 `fix:` commit 之后，下一轮 audit 必须包含一次专门的全章扫描式 fix verification round（**全章逐字读，不限于本 commit 引入的行**——之前的「本 commit 引入行」scope 漏掉了中段纯文字描述段，参见第 7 类反复错误）。每章 agent 强制带 `introduced_by_commit` 字段追溯次生错误来源，不要单纯依赖下一轮 fresh audit 顺带抓。

### 跨章节一致性 audit

单章节 audit-fix 抓不住「同一概念在不同章节被错记、术语不统一、跨章引用错位」这类问题。历史经验：H100 BF16 dense 989.5 TFLOP/s 跨章口径、MoE $d_c$ 数字、$d_k$/$h$ vs $H$/$N$ 符号反向、Llama 3 70B tokens/param 与 ch10 15.6T 错位等都是单章节审计漏掉的真错。

**主题组划分**（每轮跑 5–8 组，覆盖所有章节）：
- **GPU 硬件**：H100 / B200 / A100 数字、SM / HBM / Tensor Core 跨章口径
- **Transformer 架构**：$d_{model}$ / $d_k$ / $h$ / RoPE / RMSNorm 公式、$N$ 符号语义
- **MoE 路由**：top-k / shared-routed / capacity factor 跨章
- **数据-训练 token**：Pile / FineWeb / DCLM / DeepSeek-V3 / Llama 3 / Chinchilla 数字
- **评测基准**：MMLU / SWE-bench / HLE / Cybench / MLE-bench / AIME 数字
- **推理系统**：TTFT / latency / throughput / KV cache 公式
- **强化学习**：PPO / GRPO / DPO / RFT 公式

**跨章引用格式**（与 STYLE.md「跨章引用格式」同步）：跨章引用统一使用「第 N 章 §X.Y 章节标题」格式，不允许缩写为 `chapterN` 或省略章节标题。章节标题必须与目标章节实际 H2/H3 一致。

**Schema 简化**：跨章节 audit 的 finding 含 `files: [...]` / `line: {file: N}` 多文件嵌套对象，必须使用简化 schema（参见「工具使用经验」的 SO retry 超限 → 简化 schema 重跑）。

复核 round 也走方案 A：fix verification round 的每个 sub-agent 在自己任务里直接 Edit 修复自己抓到的次生错误，不把 finding 传给独立修复者——避免再次触发 impression 跨 agent 传染。**fresh audit 与 fix verification 两轮次都用方案 A**，agents 在各自轮次里同时 audit + 修复。

### 目标收敛判据

循环持续直到 `critical/high refuted 真错` 收敛到 0 条，且下一轮 fresh audit 独立 verdict 不再抓到新增 critical/high refuted。当两条标准同时满足、内容"逻辑自洽、出错符合各项要求"时停止。

任何 sub-agent "建议加 X 链接 / 标注 Y 来源"，自己 commit 前用 WebFetch 实际核一次；sub-agent 写出来的参考文献可用率约 70%-80%。

## 引用与求证经验（高频幻觉模式）

讲义正文出现 arXiv 编号、模型/硬件数字、benchmark 分数、版本号或日期时，**必须先 WebFetch 实际页面验证**，再写入。以下模式从多轮审计中沉淀，每条都对应真实发生过的错误：

- **主 agent 自己也会引入幻觉引用**：曾出现主 agent 主动写 Santurkar 论文引用时给了 arXiv:2306.12488（实为 GR-QC 黑洞论文；正确是 arXiv:2303.17548）。原"sub-agent 建议主 agent 复核"规则扩到**任何 arXiv 编号进笔记前必须 WebFetch abstract 确认标题与作者**，不分主 agent / sub-agent / 主动写 / 建议添加。
- **同一篇论文在 lecture 课件与原始论文中可能采用不同的符号约定**（如 linear attention state shape 取法）。遇到这类差异时，**讲义正文应当同时记下两种约定并说明自洽性**，不要凭印象挑一边"修正"。
- **模型、硬件、benchmark 等变化快的数据必须**在章节末"来源与更新记录"或来源映射中明确查阅日期（`YYYY-MM-DD`）。
- **abstract-only 复核漏 PDF body 数字**：abstract 只说"few examples"，PDF §2 实际是 "extract the top 800"；abstract 没给数值范围，PDF §5 才是。**训练 token 数 / 样本数 / 硬件带宽 / benchmark 分数 / 模型并行度等核心数字必须 PDF/HTML 正文复核**，abstract 仅用于核对标题、作者与提交日期。
- **arXiv 编号归属 hallucination 比"编号不存在"更常见**：编号真实存在（abstract / 作者都"对"），但论文主题与笔记归因不符（如把 FP8-LM 这篇 Microsoft 论文归到 NVIDIA Transformer Engine）。WebFetch abstract 时除了核对标题作者，还要**核对论文主题是否匹配笔记陈述**；必要时 fetch 正文 § 1 abstract / § 5 conclusion 二次确认。
- **arXiv ID 格式异常 = 幻觉信号**：标准格式 `YYMM.NNNNN`；任何 `YYYYMM.NNNNNN` 或其他异常前缀直接视为"未验证"或"幻觉"。**任何 arXiv 编号进笔记前肉眼检查 ID 格式合规**。
- **论文章节归属 hallucination**：曾出现 Llama 2 safety SFT 数字标在 §6.2，但实际数字在 §4.2.2（§6 是 Related Work）；Stiennon 标题写成 "with Human Feedback"，原文是 "from Human Feedback"。**引用具体数字 / 章节时必须 WebFetch 论文 § 编号 + 章节标题二次确认**，不能仅靠 abstract。
- **论文方向 / 结论读反是高频 sub-agent 错误**：曾出现 Santurkar 论文读反——论文说 RLHF 模型**削弱**对基督教主流群体的代表（朝"不属于任何宗教"群体靠拢），笔记却写成"偏向佛教 / 印度教 / 无神论者"。这种错误靠 abstract 复核抓不到，必须读论文 § 实验结论或 Figure 关键图。**对涉及 demographic / preference / comparison 类陈述，强制要求 PDF § 实验结论二次确认**。
- **法官 / 公司 / 人物名必须逐字核验**：早期 fix 曾把 Bartz v. Anthropic 案的法官写成 Judge Araceli Martínez-Olguín（实际 Judge Susan Illston, N.D. Cal.）；Llama 3 tech report 首作者在不同 arXiv 版本里是 Dubey et al. (v1/v2) → Grattafiori et al. (v3+，2024-11-23 起)；UltraFeedback 第一作者 Cui et al. 的 arXiv ID 是 2310.01377 不是 2310.01386。**任何具体人名 / 公司名 / 法官名 / 论文首作者 / arXiv ID 在写入前必须 WebFetch 一手页面核验作者列表与 ID 字面值**——abstract-only 复核或印象记忆都会写错。
- **量化口径"看起来小但本质大"**：
  - ViT-bigG：OpenCLIP 官方 2.54B vs Qwen-VL 论文 Table 1 报 1.9B（去掉 head 的非 embedding 部分）；两个数字都对，但缺一句话解释读者就会困惑
  - LIMO/R1 蒸馏：800k = 600k reasoning + 200k non-reasoning，不应笼统说"800k CoT traces"
  - Mistral 7B：官方未披露训练 tokens；按多家二手估计 ~8T（≈1,100+ tokens/param），不是早期"110 tokens/param"
  **修正方式**：保持所有数字 + 显式说明口径差异，不要强行二选一。
- **per-SM vs per-GPU / dense vs with-sparsity 是 GPU 规格常见口径混淆**：A100 FP64 19.5 TFLOP/s 实际是 with-sparsity（dense 是 9.7）。**引用 GPU datasheet 时必须区分 dense / sparsity / per-SM / per-GPU 四个维度**，并在表头明确标注。
- **教材级章节归属幻觉（sub-agent 凭印象扩写论文细节）**：RFT 写成 "Reinforcing Fine-Tuning" 而 DeepSeekMath §5.2.1 原文是 "Rejection Sampling Fine-tuning"；Dr. GRPO baseline 改成 leave-one-out（实际论文 §3.2 只删 std 与 1/|o_i| 两个分母）；R1 §2.3.1 冷启动写"温度 1.0 + DeepSeek-V3 精炼 + LLM 批量扩展"（实际论文 §2.3.1 列出的是 long CoT few-shot / 直接生成详细答案 / R1-Zero 输出整理 / 人工后处理 四种）。**对近期（≤ 2024）论文涉及实验细节的引述，必须读 PDF § 正文，不能凭 abstract 或课件 PPT 一句话扩写**。
- **图意核对 sub-task 模板**（与 STYLE.md「图意核对」同步）：写作核心图时必须实际打开本地 PNG 核对图意，不能只看 alt text / 文件名 / 路径。checklist：(1) 子图标题 / 坐标轴 / 图例是否被正文正确转述；(2) 正文数字与图上标注是否一致；(3) 图说的「机制解释」是否能在图中找到对应视觉证据；(4) alt text / 图注 / 正文交叉引用是否使用同一个图号。
- **删图判定**：两图承载同一份视觉信息（即使 SHA256 不同，如课件截图 vs 论文原图）即视为重复；保留论文原图、删课件截图（lecture 装饰元素属元叙述）；删图后必须按 STYLE 连号规则重排同节其他图号，并跑 `rg 图 N-x` 全仓库验证没有悬空引用。
- **tokenization 实证数据必须实际跑 tokenizer**：写 `tiktoken` / `sentencepiece` / `transformers` 类的"X 字符串切成 N 个 token id"示例时，**必须实际运行一遍 tokenizer**，把真实 id 序列与每个 id 对应的字符串写进正文 / 图说。印象记忆几乎一定会写错：ch1 §1.1 图 1.1-2 描述「年份 1885 作为 4 位数字整体成为 id 13096」错误，tiktoken 实际切分是 `[93447 Stan, 9201 ford, 673 ` was`, 24303 ` founded`, 306 ` in`, 220 ` `, 13096 `188`, 20 `5`, 13 `.`]`，1885 是 3+1 位两段而非 4 位整体。**所有 tokenizer / BPE / 词表实证段必须以 `python3 -c "import tiktoken; ..."` 或 Jupyter notebook 实际输出为准**。
- **fix verification round 抓到的 8 类反复错误**：每次 audit + fix 之后跑一轮 `fix verification` **全章扫描**（不是仅审本 commit 引入的行），下面 8 类反复出现：
  1. **跨章引用错位**：引「§X.Y 章节标题」时 X.Y 与目标章节实际 H2/H3 不对齐（ch8 L11 引「§8.6.8」陈旧但内容已前移；ch8 L1081 引「§8.4.3」但实际在 §8.4.2；ch14 L249 引「章节末总结」但实际在 §14.3 NOTE；topics L338 把 §3 RLVR 写成 §2）。
  2. **章节结构与节首描述矛盾**：节首说「四个递进的例子」但实际只有三个（ch6 L256 §6.4 matmul 在 §6.5）；节首说「三阶段」但表格只有两行（ch7 L1287 §7.11）。
  3. **模型名 / 数据错** typo：「Qwen3-Next Coder」（不存在）→「Qwen3-Coder-Next」（ch13 L1209）；「FineWeb 用 Cuckoo 哈希」（sub-agent 误植，论文 §3.4 仅写 MinHash + LSH）→MinHash（ch10 L399）。
  4. **节首描述与一手论文术语不一致 / 与正文自相矛盾**：两个子类：
     a) **节首描述与一手论文术语不一致**（Phase 9 ch4 L1003 抓的 `device-level routing` 实际论文术语是 `device-limited routing`）。fix verification agent 必须对每条节首描述断言性内容逐条 WebFetch 一手论文核证。
     b) **节首描述与正文自相矛盾**：ch7 L439「§7.3 NCCL 报的 GB/s」与正文「NCCL 并不回报这个数字」直接矛盾。
  5. **章节号归属错位**：方法 1/2/3 都在 §8.4.2 不在 §8.4.3（图 8.6-12 错引）；Method X 写在 §8.6.8 但实际在 §8.4.2（图说）。
  6. **元叙述 / 写作计划承诺残留**：「§1.3 接下来会再次出现」（forward-looking 承诺，ch1 L132）；「每章开头的『本章学习目标』会标出」但 14 章无一标注（不成立，preface L23）；「使用前应核对...再决定是否照搬」（搜证元句，ch4 L1069）。
  8. **跨章引用缩写违规**（Phase 9 ch4 L112/537/547 抓的 `EP / ETP / EDP`）：ch4 章节内部多次缩写 `Expert Parallelism` 为 `EP`，但跨章引用应使用完整 H3 标题「第 N 章 §X.Y 章节标题」，不能用 2-3 字母缩写。fix verification agent 必须对每个跨章引用 grep「ch[0-9]+」、「第 N 章 §」+ 缩写模式，识别缩写后改成完整标题。
  7. **中段纯文字描述段漏扫**（fix verification 默认 scope 偏章首/图说/章末，**§X.2 / §X.3 / §X.4 中没有图、表格、引用的纯文字断言段容易被跳过**）。Phase 8 ch1 §1.3 L175 教训：b8773b4 写「GPT-2 预分词正则大致是... DeepSeek 系列则使用 `\p{L}+|...`」三处错（"大致是"作者声音 / openai_public.py 引用错 / "在词内或数字串中间发生 merge"反了 `\p{N}{1,3}` 的实际行为），fix verification 当时只抓了 §1.3-1 / §1.3-2 / §1.3-2 三处图说（同一节内但都属「带图」段），**完全漏掉 L175 这段没有图的纯文字描述**。fix verification prompt 必须显式要求 agent 逐字读 §X.2 / §X.3 / §X.4 全段，对每条断言性内容（"X 预分词正则 / Y 算法用 Z 公式 / N 模型用 M 配置"等）逐条 WebFetch 一手核证；agent 默认扫描容易跳过「没有图、没有表、只有一两段纯文字」的节中段。
  11. **结构性内容必走第一性原理**（Phase 11 ch8 教训）：改图说 / 长公式 / 长算法描述等结构性内容时，**不能只做「去冗」**——要从第一性原理问 4 个问题：(1) 这段内容回答什么工程决策或论述节点？(2) 为什么放在这一节（哪段论述需要它才能继续往前推）？(3) 不放这段内容，论述会怎么断？(4) 正文（不是图说）现在有没有真正调用这段内容？如果 4 个问题的答案都是「不需要」，则可以删除或大幅压缩；如果任何一个问题答案是「需要」，则需要按「第一性原理」重写而不是简单去冗。**修正**：fix verification round 的 prompt 强制要求 agent 对结构性内容按这 4 个问题做 audit，发现问题 3 / 4 的答案是否定的原位改进。
  9. **sub-agent 报告机制漏洞**（Phase 9 v2 教训）：当 audit-fix agent 报告 "改前 / 改后" 时，notes 里的「旧值」可能是**已经改过的最终值**而不是 git HEAD 改前值。Phase 9 ch1 agent 报告「L288 来源记录 `\x8f` 应为 `\x8d`」但实际 L288 在 Phase 9 v2 之前（d35c1cf）就是 `\x8f`（错），Phase 9 v2 改成了 `\x8d`（对）——agent notes 的「改前/改后」描述与 git HEAD diff 一致，但 audit 阶段读 notes 看不到**「Phase 9 v2 改的旧值」**这条线索（agent 自我修订已经把 notes 写成"已修复后"）。**修正**：fix verification round 的 prompt 必须显式要求 agent 在 notes 字段附「`git show <commit-hash>:<file>:<line>` 改前值」与「`git show HEAD:<file>:<line>` 改后值」，让外部审稿人能从 git ref 重建完整改前/改后链路。
  10. **结构性内容改动必读课程材料**（Phase 11 ch13 教训）：当 audit-fix agent 改**结构性内容**（长公式推导 / 长算法描述 / 长代码块）而非单点事实数字时，**必须**先 Read 课程材料（`sources/_extracted_pdfs/lecture_{NN}.txt` 与 `sources/lectures/lecture_{NN}.py`）做第一手依据。Phase 11 ch13 agent 改了 80 行 PPO 详细推导（policy gradient → REINFORCE → TRPO → PPO-clip），但 transcript 显示 **0 次 Read lecture 抽文 / lecture_13.py**——agent 凭印象压缩，notes 写「Read ch12 §12.4 验证 PPO 推导已讲过」也是凭印象编的。**修正**：fix verification round 的 prompt 强制要求 agent 改结构性内容前必须 `Read` 课程材料，并把 read 路径写进 notes 字段「`lecture read: [路径列表]`」让外部审稿人能验证第一手依据存在。
  **每条都按方案 A 就地 Edit 修复**（不归到独立 round），并跑禁用句式 + 元叙述两类 rg 自检零命中才算完成。
- **多 agent 并行 fix verification 经验**：单章节 fix verification 单独跑一次 agent，16 章节并行启动 16 个 agent，每个 agent 做**全章扫描**（用 `git log` 看最近 commit 历史做参考，但不限于本 commit 引入的行——参见第 7 类反复错误，「中段纯文字描述段漏扫」就是限定本 commit 引入行时漏掉的真实案例）。Schema 仍用简化版（findings_count / edits_applied / fixes / notes），避免 SO retry 超限。16 agent 并行总耗时约 13 分钟，比串行快 16×；findings 数量比单章节 fresh audit 多（因为每条 finding 都覆盖中段纯文字 + 图说 + 表格），但每条都是「本轮可修复的真错」而非历史包袱（agent 必须用 `git log` 过滤历史问题、只对当前 commit 之后或本 fix 阶段引入的行下 Edit）。

## 工具使用经验

- **sub-agent 偏好 WebFetch 而忽略 WebSearch** 是持续性问题——但有解法。早期 prompt 要求"WebSearch ≥ 5~8 次"实际只跑 ~2 次 / agent；改 schema 把 WebSearch 拆为 finding 必填的 `search_query_1` / `search_query_2` / `search_query_3` 三字段后，平均跃升到 ~12 次 / agent（约 6×），并一次性抓到 11 处 fix 留下的次生错误（arXiv ID 归属、论文小节归属、法官名、数字精度等）。**结论**：单纯在 prompt 写"WebSearch ≥ N 次"无效，必须把 WebSearch 列到结构化输出必填字段才能强制执行。三个 search_query 还要求**不同关键词**（不能 3 次都搜同一个词）。
- **fix 本身可能错**：每轮 audit 抓到真错并修复后，**fix 本身也可能错**。曾出现：修复 Santurkar 引用时写对 arXiv ID 但同 commit 引入 ch10:154 Bartz v. Anthropic "Judge Araceli Martínez-Olguín"（实际 Judge Susan Illston, N.D. Cal.）；改 OpenHermes 错分时写 ch12:281 "UltraFeedback arXiv:2310.01386"（实际 2310.01377）、"Tulu 3 arXiv:2411.19484"（实际 2411.15124）；写 ch14:51 SigLIP "5 天 73.4% 是 from-scratch 微调"（实际论文 Table 1 caption 明确 from-scratch + 5 天行也是随机初始化）。发现模式：**单轮 audit + 单轮 fix 不够**——必须跑"**次生复核 round（fix verification）**"专门审上次 commit 引入的行，专项找 typo / 错行号 / 错归属 / 错编号 / 旧幻觉残留。每轮 audit 后额外跑一次 fix verification，agent schema 强制带 `introduced_by_commit` 字段追溯次生错误来源。
- **Schema 复杂度 → agent 失败率**：曾出现 ch5 agent 因 StructuredOutput retry cap 5 次超限而失败（Phase 1 ch12 也是同一模式）。**结论**：finding 字段不要塞太多嵌套对象，能合并就合并；web_evidence 简化为 `search_query + fetch_url + fetch_snippet + verdict` 四字段就够，不要加 `source_kind` / 多个 evidence 对象。但 `search_query_1/2/3` 三字段是必须的例外——这是强制 WebSearch 的有效手段。
- **SO retry 超限 → 立即简化 schema 重跑**：跨章节 audit 与其他多文件 / 多来源 finding 类任务的 schema 含 `files: [...]` / `line: {file: N}` / `web_evidence` / 多文件嵌套对象时，agent 容易 SO retry 5 次失败，但 Edit 已经落地、finding 列表未返回。**结论**：这类任务 schema 必须简化到 ≤ 4 个顶层字段。简化版本用三字段：`findings_count` / `edits_applied` / `notes`，足够驱动 commit 与进度报告。Edit 不需要被 schema 验证，仅最后 SO 必须返回。**重跑触发条件**：journal 报 `StructuredOutput retry cap (5) exceeded` 或 `subagent completed without calling StructuredOutput`。
- **次生修复三类模式**（每轮 audit 后必须再跑 fix verification round）：
  - **数字回归 revert**：上次 commit 把对的数字改成错的（如 Singhal LPPO Stack 59%→58%、Stiennon 标题小写→Title Case），下次 audit revert 之前要确认原值与论文原文一致，不是把错的二次 revert 回对的。
  - **casing 归一化漏扫**：上次 commit 修一处 casing（如 CyBench→Cybench），其他章节残留（SWEBench / MLEBench / SWE-Bench Verified）要在同一 commit 内 `rg` 全仓库扫一遍。
  - **修复引入新错**：上次 commit修一个低阶项时把另一个数字写错（如 8×→4× 时把 `5MN+M` 漏掉 `+2M` 低阶项），下次 audit 必须回读上下 5 行做 sibling 自洽检查。
- **图片 SHA256 不一致不一定是不同图**：SHA 不同但视觉内容相同的图（如课件截图 vs 论文原图、同一 PDF 不同页面截图）都是「同一份视觉信息的不同版本」，应判为重复；删图判定用视觉内容，不只用 SHA。
- 课件 PDF 在 agent 上下文里读会触发图片上限（一次 Read 可能塞入几十张 page-image），先用 `pdftotext -layout` 抽成纯文本到 `sources/_extracted_pdfs/lecture_NN.txt` 供 agent 线性读取；该目录作为维护副产物，不纳入 git（已在 `.gitignore` 或不入库）。
- Workflow 脚本读取运行参数时，harness 把数组 args 包成 `{item: [...]}`，pick-args 函数要同时处理 string、数组、`{item}` 和 `{cluster}` 四种 shape，否则会出现"9 个 agent 都退化成对同一文件 audit"的浪费。

## 资料位置

- 课件、slides、讲义整理和相关原始材料放在 `sources/lectures/`。
- 字幕、转写和相关文本材料放在 `sources/captions/`，按 `lecture_{NN}.md` 与 `lecture_{NN}.{py,pdf}` 一一对应；时间戳只在原 `.txt` 抓取阶段保留一次，不再随仓库发布。
- 写章节或补来源时，先到这两个目录找原始材料，再回到 `docs/` 和 `sources/` 落笔。

## 当前文件架构

- `README.md`：仓库总入口，说明笔记定位、阅读入口、目录结构和维护原则。
- `STYLE.md`：全仓库写作与维护规范，是修改 Markdown 内容前必须遵守的主规范。
- `LICENSE.md`、`NOTICE.md`：许可证与改编来源说明。
- `docs/00_前言.md`：前言和整体阅读铺垫。
- `docs/chapterN/chapterN_标题.md`：稳定章节笔记。当前覆盖第 1-14 章，包括 tokenizer、PyTorch 与资源核算、模型架构、MoE、GPU、分布式训练、Scaling Laws、推理系统、数据工程、评估、训练流程、强化学习和多模态模型。
- `docs/topics/`：不纳入主章节编号的专题补充，目前包含 `reasoning_behavior.md`。
- `sources/`：公开来源映射和课程材料对应关系；`cs336-2026.md` 是映射文档，`captions/` 收录 `lecture_{NN}.md` 字幕整理稿，`lectures/` 是课程原始材料（独立 git 仓库）。

## 维护约定

- 章节采用专题组织，不按 lecture 逐讲重排；逐 lecture 对应关系放在 `sources/`。
- 稳定、可反复复核的概念、公式、代码解释和工程经验优先进入 `docs/`。
- 尚在校准中的信息仍写入最相关章节或专题，但作为可被未来数据修正的案例或观察，避免写成永久排名、稳定常数或无条件结论。
- 新增图片放在对应章节的本地 `images/` 目录，并遵守 `STYLE.md` 的图号、文件名、图注和来源记录规则。
- 不提交站点构建产物、未引用旧资源、过时导版 PDF 或与笔记维护无关的临时文件。

## 提交约定

- 提交标题采用 conventional commits：`docs:` / `chore:` / `fix:` / `refactor:` 等小写前缀 + 冒号空格 + 祈使句动作（如 `docs: refine inference systems chapter`）。
- 正文（commit body）可选；非平凡改动应简要列修改重点或风险提示。
- **不**附加 `Co-Authored-By: Claude Code ...` / `Generated with ...` / `🤖 Generated with ...` 之类 trailer；本仓库以维护者本人为唯一署名。
- 一个 commit 只对应一类改动；如果同时动 `docs/` 和 `sources/`，按"先 docs 与内容，后 chore 与构建产物"拆 commit。
- 修改后跑「修改后自检」一节的 3 类 rg 与检查，确保新提交无命中。
- `CLAUDE.md` 与 `AGENTS.md` 内容保持一致；修改 `AGENTS.md` 后须 `cp AGENTS.md CLAUDE.md` 一并提交。

## 与 STYLE.md 的同步锚点

| AGENTS.md 节 | 对应 STYLE.md 节 | 同步关系 |
|---|---|---|
| 修改后自检 1（禁用句式 rg） | STYLE.md「验证流程不进正文」+「全局原则」+「修辞层规则」 | rg 关键字清单与 STYLE.md 反例一一对应 |
| 修改后自检 2（作者声音穿帮 rg） | STYLE.md「修辞层规则」 | rg 关键字清单来自 STYLE.md 反例（含「这里使用」「下一节展开」等） |
| 修改后自检 3（提示块合理性） | STYLE.md「提示块」 | 5 种 alert 语义边界与 STYLE.md 对齐（IMPORTANT/WARNING/CAUTION 三档严格递进，NOTE 不承载作者声明） |
| 审计 7 层第 1 / 6 层（事实 / 数字） | STYLE.md「公式与排版」+「公式账本」 | 公式符号、跨章节数字一致性 |
| 审计 7 层第 2 层（公式账本） | STYLE.md「公式账本」 | 符号首次出现即定义、公式与代码对齐、公式后承接句、长算式拆分、不允许 snake_case 进数学环境 |
| 审计 7 层第 3 层（图意） | STYLE.md「图意核对」+「图片」 | 核心图必读本地 PNG、6 类图文不符、删图判定 |
| 审计 7 层第 4 层（段落 / 修辞） | STYLE.md「段落与承接」+「修辞层规则」+「修辞层规则」反例清单 | 长段拆段、单段承载动作 > 2 拆段、公式 / 表格 / 代码块后承接句、作者声音穿帮 |
| 审计 7 层第 5 层（术语 / 符号） | STYLE.md「术语表」+「全局原则」 | 同一概念用同一英文术语、同一符号指同一变量 |
| 审计 7 层第 6 层（数字） | STYLE.md「来源与日期」+「章节末『来源与更新记录』约定」 | 同一硬件 / 模型 / benchmark 跨章一致；4 段结构（官方 / 课程 / 事实声明指向 / 不确定项） |
| 审计 7 层第 7 层（引用 / 跨章） | STYLE.md「跨章引用格式」+「图片」alt text 一致 | 跨章引用「第 N 章 §X.Y 章节标题」、caption / 图注 / 正文交叉引用一致 |
| 跨章节一致性 audit | STYLE.md「学习式讲义」「术语表」 | 主题组（GPU 硬件 / Transformer / MoE / 数据 / 评测 / 推理 / 强化学习）的术语对照 |
| 引用与求证经验（章节归属 / 人物名 / arXiv ID） | STYLE.md「术语表」「来源与日期」 | sub-agent 在命名 / 引用 / 章节号层面遵守同一标准 |
| 图意核对 sub-task 模板 / 删图判定 | STYLE.md「图意核对」「图片」 | 4 个检查项 + 两图视觉内容一致即可删 |
| 修改后自检 1（禁用句式 rg）与第 7 层 | STYLE.md「跨章引用格式」 | rg pattern 抓「第 N 章 §X.Y」与「chapterN」缩写 |

AGENTS.md 与 STYLE.md 任何一侧调整规则时，另一侧必须同步引用对应章节；同一 commit 内完成。