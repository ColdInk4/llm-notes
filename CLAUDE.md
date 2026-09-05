# AGENTS.md

本文件适用于整个 `llm-notes` 仓库。所有维护者和自动化 coding agent 在修改前都应先阅读这里的说明，并以 `STYLE.md` 的写作与排版约定为准。

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

- 修改 Markdown 正文后，必须对本次修改过的正文文件运行禁用句式检查，至少覆盖 `STYLE.md` 中明确列出的元提示和否定转折句式：

  ```bash
  rg -n "读法|看这类图时|工程读法|这张图告诉我们|当作起点|当成定律|不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是" <modified-markdown-files>
  ```

- 若检查命中正文，先改成正向讲义表达：直接说明变量关系、成立条件、边界和工程后果。只有代码、真实引用、规范原文或必须保留的搜索模式可以命中，并在最终回复里说明。
- 讲义正文**不要**出现下列元叙述：
  - "建议引用时直接查证 X"
  - "以 Y 为准 / 未经 X 验证"
  - "待 Y 披露后核对"
  - "本节以 X 为准 / 属讲师口误"
  - 任何把"我不确定 / 我没核实 / 读者该做什么"暴露给读者的免责语

  这类语句把审稿过程带到读者面前，破坏讲义"按权威源直接陈述"的风格。**正确做法**：要么按权威源给出事实（数字 + 出处），要么省略该数据点；来源记录统一写章节末"来源与更新记录"或来源映射区，正文不写 disclaimer。
- 自查 grep 除 `STYLE.md` 列出的句式外，再加查：`未经一手`, `建议引用`, `待 .* 核对`, `缺口`, `以 .* 为准`, `未能核`, `本节以`。

## 资料位置

- 课件、slides、讲义整理和相关原始材料放在 `sources/lectures/`。
- 字幕、转写和相关文本材料放在 `sources/captions/`，按 `lecture_{NN}.md` 与 `lecture_{NN}.{py,pdf}` 一一对应；时间戳只在原 `.txt` 抓取阶段保留一次，不再随仓库发布。
- 写章节或补来源时，先到这两个目录找原始材料，再回到 `docs/` 和 `sources/` 落笔。

## 广查与审计必须联网（强制门槛）

任何形式的"广查 / 盲查 / fresh audit / 三视角循环 / Phase 1~3 finding 收集"，**即使尚未动手修改任何文件**，也按以下最小门槛执行；这是对上文"回答前复核"的延伸与硬约束：

1. 每个 sub-agent 在出 finding 之前，**必须**对涉及外部事实的部分（arXiv 编号、模型/硬件数字、benchmark 分数、版本号、日期、厂商声明、引用文献、HF model id）至少调用一次 WebSearch，并 WebFetch 一手页面（arXiv abstract / 官方 model card / HuggingFace `config.json` / 官方仓库 README / 技术报告 PDF）做实际验证。**不得**仅凭印象、训练数据记忆或课件文字给出 verdict。
2. finding 列表里每一条 `severity ≥ medium` 或 `needs_web_check=true` 的项目，必须在 evidence 字段附上：
   - 实际 WebSearch 查询字符串或 WebFetch URL；
   - 一手页面里的相关原文/数字摘录（≤ 2 行）；
   - 笔记正文里被质疑的那句话或数字（含 file:line）。
3. 没有附上述证据的 finding 视为 `tentative`；Phase 2 求证阶段按主题聚类 WebFetch 核证，仍无法被一手源支撑的应当降级或丢弃，不得计入最终 fix 列表。
4. sub-agent "建议加 X 链接 / 标注 Y 来源" 的建议，主 agent 在 commit 前**必须**自己再 WebFetch 核一次；历史经验：sub-agent 写出来的参考文献可用率约 70-80%，主 agent 必须复核。
5. 历史教训（必须记住）：曾出现 arXiv:1712.00409（实为 Hestness 等 Baidu 论文，并非"44× ImageNet 出处"）和 arXiv:2401.06865（实为天文学家 Osborne & Salim 的星系 SED 拟合论文）两例 sub-agent 幻觉引用。**任何 arXiv 编号进笔记前必须先 WebFetch abstract 页面确认标题与作者**，不接受"看起来对题"。

### 联网求证工具用法

- **WebSearch 与 WebFetch 必须并用**——只 WebFetch 不 WebSearch 是 sub-agent 的常见偷懒模式，会导致"对已存在但 fetch 不到的来源也放过"或"对子 agent 印象里的 URL 做 round-trip 求证"。每轮广查 / 审计每个 sub-agent 的硬性最小值：
  - WebSearch ≥ 5 次，覆盖"是不是真存在 / 是不是真作者 / 是不是真数字"三种目的
  - WebFetch ≥ 5 个不同 URL，其中至少 2 个为 `arxiv.org/pdf/{id}` 或 `/html/{id}` 的正文（非 abstract）
  - finding 的 `web_evidence.search_query` 必须实际跑过；不能写"未跑"
- WebSearch 关键词模板（必须包含具体名词，不要泛查询）：
  - arXiv 复核：`"Hestness 2017 arxiv 1712.00409 ImageNet"` / `"Rafailov DPO arxiv 2305.18290 authors"`
  - 模型配置复核：`"LLaMA-2 7B num_key_value_heads config.json GQA"` / `"Qwen3-235B-A22B vocab_size huggingface"`
  - GPU 规格复核：`"H100 SXM HBM3 bandwidth 3.35 TB/s spec sheet"` / `"B200 SM count 144 GB100 whitepaper"`
  - benchmark 复核：`"Cybench first solve time arXiv 2408.08926 24h54m"` / `"MMLU-Pro 10 options CoT 33% arxiv 2406.01574"`
- WebFetch 优先 URL：`arxiv.org/abs/{id}`（标题/作者/日期）→ `arxiv.org/pdf/{id}` 或 `/html/{id}`（正文 Figure/Table）→ `huggingface.co/{org}/{model}/blob/main/config.json` → 官方仓库 README → 技术报告 PDF。
- 一次 WebFetch 拿不到信息就换关键词再 WebSearch；连续两次失败应切换一手 URL 而不是堆 WebSearch 数量。
- WebSearch/WebFetch 触达受 sandbox 限制时，回到 `回答前复核` 的降级路径：显式声明、影响范围、标记 `tentative`。

### 必须读原论文，不能只读 abstract

arXiv `abs/{id}` 页只有 abstract 与 metadata，绝大多数具体数字（样本量、训练 token 数、benchmark 分数、超参数表、Figure/Table 数据）都只在 PDF / HTML 正文里。任何 arXiv 引用进笔记前，WebFetch 的最小集是：

1. `arxiv.org/abs/{id}` 确认标题、作者、提交日期；
2. `arxiv.org/pdf/{id}` 或 `arxiv.org/html/{id}` 拿正文关键 Figure / Table 数字与公式；
3. 引用该论文的下游论文 / 官方 model card / 榜单页做交叉核对。

仅基于 abstract 出 finding 时，必须在 finding 证据字段显式标记 `source=abstract-only` 并降级为 `tentative`；不允许把 `abs/{id}` 的内容当成"已验证"结论写入 fix。

### 搜证过程不进讲义正文

下列短语属于"搜证过程元句"，**禁止**出现在 `docs/` 任何正文（包括章节末"来源与更新记录"之外的来源记录小段）。出现即视为把审计过程带到读者面前：

- "abstract 仅给出…"
- "正文表述以论文原表为准"
- "需按目标章节复核"
- "建议引用时直接查证 X"
- "未经 X 验证"
- "待 Y 披露后核对"
- "以 Y 为准 / 未经 X 验证"
- "本节以 X 为准 / 属讲师口误"
- "未能核"
- "缺口"

这些语句只在 `AGENTS.md` / `STYLE.md` 维护审计流程用，不进入讲义正文。讲义正文要么直接给出事实 + 出处，要么省略该数据点，不留搜证过程的痕迹。

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

## 维护实践经验（从审计和修复循环中沉淀）

以下经验来自多轮"课件比对 + 网络求证 + 全量修复"循环，作为后续维护的默认动作。每条经验仅描述问题与对策，不附轮次编号。

### 引用与求证

- 讲义正文出现 arXiv 编号、模型/硬件数字、benchmark 分数、版本号或日期时，**必须先 WebFetch 实际页面验证**，再写入。sub-agent 给出的"看起来对题"的 verdict 不可信 — 历史观察中曾出现 arXiv:1712.00409（实为 Hestness 等 Baidu 论文，并非"44× ImageNet 出处"）和 arXiv:2401.06865（实为天文学家 Osborne & Salim 的星系 SED 拟合论文）两例幻觉。
- **主 agent 自己也会引入幻觉引用**：曾出现主 agent 主动写 Santurkar 论文引用时给了 arXiv:2306.12488（实为 GR-QC 黑洞论文；正确是 arXiv:2303.17548）。原"sub-agent 建议主 agent 复核"规则扩到**任何 arXiv 编号进笔记前必须 WebFetch abstract 确认标题与作者**，不分主 agent / sub-agent / 主动写 / 建议添加。
- **同一篇论文在 lecture 课件与原始论文中可能采用不同的符号约定**（如 linear attention state shape 取法）。遇到这类差异时，**讲义正文应当同时记下两种约定并说明自洽性**，不要凭印象挑一边"修正"。
- **模型、硬件、benchmark 等变化快的数据必须**在章节末"来源与更新记录"或来源映射中明确查阅日期（`YYYY-MM-DD`）。
- **abstract-only 复核漏 PDF body 数字**是常见漏报：abstract 只说"few examples"，PDF §2 实际是 "extract the top 800"；abstract 没给数值范围，PDF §5 才是。**训练 token 数 / 样本数 / 硬件带宽 / benchmark 分数 / 模型并行度等核心数字必须 PDF/HTML 正文复核**，abstract 仅用于核对标题、作者与提交日期。
- **arXiv 编号归属 hallucination 比"编号不存在"更常见**：编号真实存在（abstract / 作者都"对"），但论文主题与笔记归因不符（如把 FP8-LM 这篇 Microsoft 论文归到 NVIDIA Transformer Engine）。WebFetch abstract 时除了核对标题作者，还要**核对论文主题是否匹配笔记陈述**；必要时 fetch 正文 § 1 abstract / § 5 conclusion 二次确认。
- **arXiv ID 格式异常 = 幻觉信号**：标准格式 `YYMM.NNNNN`；任何 `YYYYMM.NNNNNN` 或其他异常前缀直接视为"未验证"或"幻觉"。**任何 arXiv 编号进笔记前肉眼检查 ID 格式合规**。
- **论文章节归属 hallucination**：曾出现 Llama 2 safety SFT 数字标在 §6.2，但实际数字在 §4.2.2（§6 是 Related Work）；Stiennon 标题写成 "with Human Feedback"，原文是 "from Human Feedback"。**引用具体数字 / 章节时必须 WebFetch 论文 § 编号 + 章节标题二次确认**，不能仅靠 abstract。
- **论文方向 / 结论读反是高频 sub-agent 错误**：曾出现 Santurkar 论文读反——论文说 RLHF 模型**削弱**对基督教主流群体的代表（朝"不属于任何宗教"群体靠拢），笔记却写成"偏向佛教 / 印度教 / 无神论者"。这种错误靠 abstract 复核抓不到，必须读论文 § 实验结论或 Figure 关键图。**对涉及 demographic / preference / comparison 类陈述，强制要求 PDF § 实验结论二次确认**。
- **法官 / 公司 / 人物名必须逐字核验**：早期 fix 曾把 Bartz v. Anthropic 案的法官写成 Judge Araceli Martínez-Olguín（实际 Judge Susan Illston, N.D. Cal.）；Llama 3 tech report 首作者在不同 arXiv 版本里是 Dubey et al. (v1/v2) → Grattafiori et al. (v3+，2024-11-23 起)；UltraFeedback 第一作者 Cui et al. 的 arXiv ID 是 2310.01377 不是 2310.01386。**任何具体人名 / 公司名 / 法官名 / 论文首作者 / arXiv ID 在写入前必须 WebFetch 一手页面核验作者列表与 ID 字面值**——abstract-only 复核或印象记忆都会写错。
- **量化口径"看起来小但本质大"**：曾出现
  - ViT-bigG：OpenCLIP 官方 2.54B vs Qwen-VL 论文 Table 1 报 1.9B（去掉 head 的非 embedding 部分）；两个数字都对，但缺一句话解释读者就会困惑
  - LIMO/R1 蒸馏：800k = 600k reasoning + 200k non-reasoning，不应笼统说"800k CoT traces"
  - Mistral 7B：官方未披露训练 tokens；按多家二手估计 ~8T（≈1,100+ tokens/param），不是早期"110 tokens/param"
  **修正方式**：保持所有数字 + 显式说明口径差异，不要强行二选一。
- **per-SM vs per-GPU / dense vs with-sparsity 是 GPU 规格常见口径混淆**：A100 FP64 19.5 TFLOP/s 实际是 with-sparsity（dense 是 9.7）。**引用 GPU datasheet 时必须区分 dense / sparsity / per-SM / per-GPU 四个维度**，并在表头明确标注。

### 审计与修复循环

- 审稿查错建议分三视角循环，避免单一 agent 视角盲点；每一阶段都受上文"广查与审计必须联网"约束：
  - Phase 1（盲查）——按 lecture 切分，sub-agent 报告 finding 列表（含 file、line、issue_type、severity、evidence、suggestion、web_evidence）。**每个 sub-agent 至少调用一次 WebSearch + WebFetch**，对涉及外部事实的 finding 在 evidence 字段附 WebSearch 查询字符串 / WebFetch URL + 一手页面原文摘录（≤ 2 行）。
  - Phase 2（求证）——仅针对 `needs_web_check=true` 或 `tentative` 的 finding 联网核证；可按主题聚类用 8-10 个 agent 并行 WebFetch / WebSearch。无法被一手源支撑的 finding 必须降级或丢弃。
  - Phase 3（章节级复核）——从零重新读当前文件 + 对应课件做 fresh audit，让每个章节独立得出 verdict；同样必须联网求证外部事实。这一轮最易发现前两轮漏掉的"原始事实"类错误（如 GPU 规格、KV cache 大小、样本数、vocab_size、训练 token 数）。
- **审计 verdict 三档规范（推荐默认 schema）**：
  - `confirmed` — 笔记对得上权威源，**没有问题**，**不报**
  - `refuted` — 笔记与权威源**直接冲突，是错的**，要修
  - `tentative` — 一手源证据不足，可能是对也可能是错，留到下轮
  prompt 要求 sub-agent 只报 `refuted + tentative`、**不再报 `confirmed`**，可显著减少低信号 confirmed informational finding，把篇幅留给可执行的 refuted 修复。
- **Phase 3 fresh audit 的真正价值是不传 Phase 1 finding 列表的独立 verdict**：Phase 1 报告的 finding 中有相当一部分在 Phase 3 fresh verdict 里被推翻或加严；Phase 3 又会新抓到一批 PDF body 真错。**审计循环不能省掉 fresh audit 这一步**，否则 Phase 1 的"误判"会在 commit 链里累积成"事实污染"。
- **章节组织 / heading level / 元叙述 是审计盲区**。Phase 1/2/3 都集中查 arXiv 与数字，但章节重复 `## 标题`、`###` 与 `##` 层级不一、大量"课件 Lecture X 字幕原话"类元叙述——这些都不在数值复核的视野里。**审计流程必须显式跑两轮 rg**：
  - 第一轮：`rg -n "课件 Lecture|课件里|课件中|slides 第|字幕原话|视频中说|Lecture \d+ 字幕|讲者已在"` 检查元叙述
  - 第二轮：`rg -n "读法|看这类图时|工程读法|这张图告诉我们|当作起点|当成定律|不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是|未经一手|建议引用|待.*核对|缺口|以.*为准|本节以|笔记此前"` 检查禁用句式
- **同一份笔记每轮审计都在抓之前漏掉的真错**。多轮维护下来，每一轮都会抓到上一轮漏掉的真错；写完一份笔记不意味着 audit 完成，应把"每轮 audit 都能挖出新错"作为预期而不是问题。
- **每轮 commit body 应包含"复盘"小节**，写明本轮自错与被推翻的 sub-agent verdict；这种 commit-time 复盘是后续 audit 的最直接 checklist。
- **任何 fix 提交前**，对修改过的正文文件跑一次 `rg -n "读法|看这类图时|工程读法|这张图告诉我们|当作起点|当成定律|不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是|未经一手|建议引用|待.*核对|缺口|以.*为准|本节以" <modified>`，命中应改写。
- **sub-agent 必读课件才能下 verdict**：每条 finding 出 verdict 之前**必须**先读四个资料源——(1) 章节文件全文；(2) 对应 lecture PDF 抽文 `sources/_extracted_pdfs/lecture_{NN}.txt`（权威技术内容来源）；(3) 对应 lecture Python 代码讲义 `sources/lectures/lecture_{NN}.py`（代码级事实、reproducible trace 来源）；(4) 章节末"来源与更新记录"与 `sources/cs336-2026.md` 里的 lecture 映射。**字幕（`sources/captions/lecture_{NN}.md`）只作为 logical flow 与上下文承接**，不作为技术事实权威——字幕包含口误 / 转录错误，技术数字与公式必须以 lecture PDF 原文 + lecture_NN.py 代码为准。finding 的 `evidence_in_lecture` 字段必须引用读到的具体原文段落或代码行（≤ 2 行）。**WebSearch + 读课件并用**：WebSearch 解决"是不是真存在 / 数字精度"；读 lecture PDF + lecture_NN.py 解决"笔记 vs 课程口径是否一致 / 课件讲的是不是更新版"。只有 WebSearch 没有读课件、或只读课件没有 WebSearch，都算 audit 不完整。
- **audit-fix 一体化 agent 硬约束（方案 A）**：每个章节的 audit 与 fix 由**同一个 sub-agent** 在两次明确分开的 Phase 内完成，避免跨 agent 印象传染。
  - **Phase 1（audit）**：读章节全文 + lecture PDF + lecture_NN.py（按上方"必读课件"规则）；WebFetch 一手页面（论文 PDF 原文 / HF model card / NVIDIA datasheet / config.json）；产出 finding 列表，**只列 wrong_facts 与 evidence_in_lecture，**不写 suggested_fix****——避免 audit 的修复方案成为 fix agent 的"印象传染源"。
  - **Phase 2（fix）**：每个 fix **必须重新 WebFetch 一手页面（不同 URL 或不同 search_query），不得复用 Phase 1 的 WebFetch 结果**。先读 lecture PDF + lecture_NN.py 原文与代码，**确认课件口径与一手页面一致后再写修复**。写完一个 fix 后**回读 fix 上下 5 行做上下文自洽检查**（sibling rows / sibling claims 是否仍成立），不一致则继续修。
  - **不在 finding 里写 suggested_fix**：fix agent 不抄 audit 的修复建议，只看 wrong_facts + evidence_in_lecture + 一手 URL，自行 WebFetch 重新核验后写修复。这条切断"修复源头已污染"的链条——之前多次翻车的 Bartz 法官 / LIMO 800 vs 817 / LLaDA 2.0 "Tiwei Bie" 都是 audit 给的 suggested_fix 已经写错而被 fix 直接抄。
  - **audit 与 fix 写在一个 agent 任务里**：避免 context 在两个 agent 之间序列化丢失；同一 agent 在 Phase 2 重新 WebFetch，避免 anchoring bias；Phase 1 与 Phase 2 之间必须**有清晰的任务结构与字段隔离**，agent 不能跳过 Phase 1 直接 Phase 2。
- **commit 后必须有次生复核 round**：单轮 audit + 单轮 fix 抓不住次生错误。曾出现 11 处 fix 留下的次生错误（Judge Illston / UltraFeedback 2310.01377 / Tulu 3 2411.15124 / SigLIP 5 天 / Mixtral TP=2 / DeepSeek-V4 残留等）直到下一轮 fix verification 才被抓到。**推荐节奏**：每次 `fix:` commit 之后，下一轮 audit 必须包含一次专门审 "上次 commit 修改行"的 fix verification round，每章 agent 强制带 `introduced_by_commit` 字段追溯次生错误来源，不要单纯依赖下一轮 fresh audit 顺带抓。
- **复核 round 也走方案 A**：fix verification round 的每个 sub-agent 在自己任务里直接 Edit 修复自己抓到的次生错误，不把 finding 传给独立修复者——避免再次触发 impression 跨 agent 传染。**fresh audit 与 fix verification 两轮次都用方案 A**（audit-fix 一体化），agents 在各自轮次里同时 audit + 修复。次生复核轮不再只列 finding 而不修。
- **目标收敛判据**：循环持续直到 `critical/high refuted 真错` 收敛到 0 条，且下一轮 fresh audit 独立 verdict 不再抓到新增 critical/high refuted。当两条标准同时满足、内容"逻辑自洽、出错符合各项要求"时停止。
- **任何 sub-agent "建议加 X 链接 / 标注 Y 来源"**，自己 commit 前用 WebFetch 实际核一次；sub-agent 写出来的参考文献可用率约 70%-80%。

### 工具使用

- **sub-agent 偏好 WebFetch 而忽略 WebSearch** 是持续性问题——但有解法。早期 prompt 要求"WebSearch ≥ 5~8 次"实际只跑 ~2 次 / agent；改 schema 把 WebSearch 拆为 finding 必填的 `search_query_1` / `search_query_2` / `search_query_3` 三字段后，平均跃升到 ~12 次 / agent（约 6×），并一次性抓到 11 处 fix 留下的次生错误（arXiv ID 归属、论文小节归属、法官名、数字精度等）。**结论**：单纯在 prompt 写"WebSearch ≥ N 次"无效，必须把 WebSearch 列到结构化输出必填字段才能强制执行。三个 search_query 还要求**不同关键词**（不能 3 次都搜同一个词）。
- **fix 本身可能错**：每轮 audit 抓到真错并修复后，**fix 本身也可能错**。曾出现：修复 Santurkar 引用时写对 arXiv ID 但同 commit 引入 ch10:154 Bartz v. Anthropic "Judge Araceli Martínez-Olguín"（实际 Judge Susan Illston, N.D. Cal.）；改 OpenHermes 错分时写 ch12:281 "UltraFeedback arXiv:2310.01386"（实际 2310.01377）、"Tulu 3 arXiv:2411.19484"（实际 2411.15124）；写 ch14:51 SigLIP "5 天 73.4% 是 from-scratch 微调"（实际论文 Table 1 caption 明确 from-scratch + 5 天行也是随机初始化）。发现模式：**单轮 audit + 单轮 fix 不够**——必须跑"**次生复核 round（fix verification）**"专门审上次 commit 引入的行，专项找 typo / 错行号 / 错归属 / 错编号 / 旧幻觉残留。每轮 audit 后额外跑一次 fix verification，agent schema 强制带 `introduced_by_commit` 字段追溯次生错误来源。
- **Schema 复杂度 → agent 失败率**：曾出现 ch5 agent 因 StructuredOutput retry cap 5 次超限而失败（Phase 1 ch12 也是同一模式）。**结论**：finding 字段不要塞太多嵌套对象，能合并就合并；web_evidence 简化为 `search_query + fetch_url + fetch_snippet + verdict` 四字段就够，不要加 `source_kind` / 多个 evidence 对象。但 `search_query_1/2/3` 三字段是必须的例外——这是强制 WebSearch 的有效手段。
- 课件 PDF 在 agent 上下文里读会触发图片上限（一次 Read 可能塞入几十张 page-image），先用 `pdftotext -layout` 抽成纯文本到 `sources/_extracted_pdfs/lecture_NN.txt` 供 agent 线性读取；该目录作为维护副产物，不纳入 git（已在 `.gitignore` 或不入库）。
- Workflow 脚本读取运行参数时，harness 把数组 args 包成 `{item: [...]}`，pick-args 函数要同时处理 string、数组、`{item}` 和 `{cluster}` 四种 shape，否则会出现"9 个 agent 都退化成对同一文件 audit"的浪费。

## 提交约定

- 提交标题采用 conventional commits：`docs:` / `chore:` / `fix:` / `refactor:` 等小写前缀 + 冒号空格 + 祈使句动作（如 `docs: refine inference systems chapter`）。
- 正文（commit body）可选；非平凡改动应简要列修改重点或风险提示。
- **不**附加 `Co-Authored-By: Claude Code ...` / `Generated with ...` / `🤖 Generated with ...` 之类 trailer；本仓库以维护者本人为唯一署名。
- 一个 commit 只对应一类改动；如果同时动 `docs/` 和 `sources/`，按"先 docs 与内容，后 chore 与构建产物"拆 commit。
- 修改后跑前文禁用句式检查（`rg -n "读法|不是.*而是|..." <modified-markdown-files>`），确保新提交无命中。
- `CLAUDE.md` 与 `AGENTS.md` 内容保持一致；修改 `AGENTS.md` 后须 `cp AGENTS.md CLAUDE.md` 一并提交。
