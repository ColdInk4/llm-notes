# AGENTS.md

本文件适用于整个 `llm-notes` 仓库。所有维护者和自动化 coding agent 在修改前都应先阅读这里的说明，并以 `STYLE.md` 的写作与排版约定为准。

AGENTS.md 是**流程规范**：回答前复核、修改前必读、审计循环、联网求证门槛、commit 约定。STYLE.md 是**写作规范**：Markdown 排版、提示块用法、术语口径、来源记录格式。两份文档在以下五处必须保持一致：(1) 禁用句式清单；(2) 提示块 5 种 alert 语义；(3) 章节末来源记录的 3 段结构；(4) GitHub 渲染安全规则（8 种公式渲染失败模式与自检 rg）；(5) 第一性原理三条硬约束与逻辑审计 logic_finding 字段口径。修改 `AGENTS.md` 后须 `cp AGENTS.md CLAUDE.md` 一并提交。

## 回答前复核

- 回答涉及仓库正文内容、章节技术解释、来源判断、修改建议或任何形式的审稿 / 审计 / fresh audit / 盲查阶段 finding 时，**必须**先完成至少一次相关网络搜索（WebSearch + WebFetch）。广查 / 盲查 / 章节级复核即使尚未动手修改任何文件，也属于本条触发范围；不只在实际改文件时才搜索。
- 联网求证后再结合本地目标章节、相邻小节和来源记录回答。若无法完成网络搜索（断网、API 限流、sandbox 拦截等），必须显式在回复中说明原因、影响范围和后续复核建议，并把相关 finding 标记为 `tentative`。
- 仅处理**纯本地机械问题**可以不联网：例如列文件、查看当前 diff、运行已有检查命令（rg / wc / git status）、读本地 PDF 抽出的纯文本。一旦涉及技术事实、外部来源、章节判断、来源映射、引用准确度或 sub-agent 给出的 verdict，仍按上一条执行。

## Token 不是停止工作的理由

在 ultracode 模式（`/effort` 设 xhigh + dynamic workflow orchestration）或任何明示"穷尽/完整"目标的会话下，**token 预算不是约束**。审计 / 盲查 / 求证 / 复核各阶段 finding、sub-agent 并行广查与 WebFetch/WebSearch 调用应做到：

- 计划阶段按 `Chapter × 问题域 × 一手源` 估出最大并行度，而不是先按"剩余 token 还够几次"压规模
- finding 数量大时**继续**分批 WebFetch 复核而不是停在"已经修了 14 处"的 commit 边界
- "受 token 预算约束未启动该阶段 / 章节 Y" 这种措辞**禁止**作为延期理由；要么启动，要么显式说明是"用户授权分批"或"技术失败"

报告里写"剩余 token 不够"等同于默认承认偷懒。正确说法要么是"已完成 X / Y / Z，未做 W 留到下一轮"，要么是"现在启动该阶段"。本仓库维护目标是把每一轮能修的真错都修完，不是把"修了 14 条"包装成完整闭环。

## 修改前必读

- 每次修改任何文件前，**必须**先完成一次相关网络搜索（WebSearch + WebFetch），并 `Read STYLE.md` 全文（覆盖 Markdown、公式、图片、提示块、来源日期、文件命名，以及修辞 / 格式 / 元叙述层约束），再 `Read AGENTS.md`「与 STYLE.md 的同步锚点」节刷新锚点。
- 再阅读 `README.md`，理解仓库定位、阅读入口、章节组织方式和维护说明。
- 修改具体章节前，先阅读目标文件及其相邻小节；如果章节末有来源记录，也要一起检查。
- 修改 CS336 相关内容前，阅读 `sources/cs336-2026.md` 中对应 lecture、slides、video 和论文映射。
- 涉及具体模型、硬件、benchmark、榜单、厂商声明等外部事实时，先定位最相关的章节或专题，并按 `STYLE.md` 的来源与日期约定记录。涉及 arXiv 编号、HF model id、官方文档链接时，**必须**先 WebFetch 一手页面验证标题/作者/数字，不允许凭印象写入。

## 修改后自检

修改 Markdown 正文后，必须对本次修改过的文件跑**四类自检**，每类命中即改写：

### 1. 禁用句式 rg（与 STYLE.md 同步）

```bash
rg -n "读法|看这类图时|工程读法|这张图告诉我们|当作起点|当成定律|不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是|未经一手|建议引用|待.*核对|缺口|以.*为准|本节以|属讲师口误|未能核|课件 Lecture|课件里|课件中|slides 第|字幕原话|视频中说|Lecture \d+ 字幕|讲者已在|笔记此前" <modified-markdown-files>
```

命中正文（章节末"来源与更新记录"除外）即改写为正向讲义表达：直接说明变量关系、成立条件、边界和工程后果。

### 2. 作者声音穿帮 rg（与 STYLE.md「修辞层规则」同步）

```bash
rg -n "这里使用.{1,30}(作为|当|来当|样例|为例)|本文采用|本文以.*为例|后续章节会|后文将|后文会|后面单独讨论|下一节会|下一节展开|下一节转到|不等于所有|不是说|并非主张|并不意味着|但愿不要|读者不要把本文|读者应以|本节以" <modified-markdown-files>
```

抓作者选择、免责声明、写作计划承诺漏到正文。命中即下沉到章节末来源记录或改写为正向事实陈述（如「§5.8 单列 B200 / Blackwell 的差异」）。注意 `下一节展开` / `下一节转到` 也属写作计划承诺，不只 `后文会` / `后续章节会`。audit finding 涉及作者声音穿帮时用 `author_voice_break` 字段报告（与 STYLE.md「修辞层规则」同步）。

### 3. 提示块类型合理性（与 STYLE.md「提示块」同步）

如果改动涉及 alert 类型，对照 STYLE.md 的 5 种 alert 语义边界（NOTE / TIP / IMPORTANT / WARNING / CAUTION）核对用法是否合理——NOTE 不应承载作者声明或免责声明；IMPORTANT / WARNING / CAUTION 三档严格递进。同时核对**节首学习目标与学习检查收束不用 alert**（一律普通段落表达），并跑节首模板 rg 兜底：

```bash
rg -n -A1 "^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]" docs/ | rg "本节(介绍|覆盖|聚焦|回答)|读完后"
```

命中即改写为普通段落。

元叙述与 disclaimer（审计元句、作者声音、写作计划承诺）的反例清单由自检 1「禁用句式 rg」与自检 2「作者声音 rg」覆盖，此处不重复列举。命中即改写：要么按权威源给出事实（数字 + 出处写章节末「来源与更新记录」或来源映射区），要么省略该数据点，正文不写 disclaimer。

### 4. GitHub 渲染安全 rg（与 STYLE.md「公式与排版」+「HTML」同步）

修改公式或含 `$` 的正文后，跑以下检测（8 种失败模式），命中即按 STYLE.md 规则修复：

```bash
# 模式1：行内开 $ 紧贴非空白字符（汉字 / 全角标点 / 字母），GitHub 整个公式不渲染
rg -n '[^\s*$\\`]\$[A-Za-z\\(]' <modified-markdown-files>
# 模式3：} 或 | 紧贴 _ 的公式——行内会切碎；display 块内凑齐开、关一对同样切碎整块
# （单向 flanking 无配对者的历史命中经核查可幸存，非本次引入可不动）
rg -n '\}_[A-Za-z{]|\|_[A-Za-z]' <modified-markdown-files>
# 模式4：字面美元金额未包 span（人工排除 $10000$ 这类已闭合公式；无 PCRE2 的 rg 用 python re 等价扫）
rg -nP '(?<![/\w>])\$\d[\d,]+(\.\d+)?\b' <modified-markdown-files>
# 模式5：math 内裸 ^* / _*（被 CommonMark 配成强调、公式变 ^_；排除 code span 内字面通配符如 <|reserved_*|>、lecture_*.md）
rg -n '\^\*|_\*' <modified-markdown-files>
# 模式6：math 内下划线（正确形式为双反斜杠 \\_；正文 prose 的 vocab\_size 合法；命中逐条区分单/双反斜杠）
#         标点前缀 _ 开强调有两种形态，都须做段落级配对检查（\\_ 可开强调、同段 _{ 下标可闭强调，配对则两条公式都被切碎）：
#         \\_ 反斜杠前缀（本 rg 命中）；prime 下标 '_{ 撇号前缀（本 rg 不命中，用下一条 rg 补扫，命中后逐段确认有无闭者配对）
rg -n '\\_' <modified-markdown-files>
rg -n "'_\{" <modified-markdown-files>
# 模式7：display 块内裸 <（行内 < 实测安全）
python3 - <modified-markdown-files> <<'PY'
import re, pathlib, sys
for p in sys.argv[1:]:
    t = pathlib.Path(p).read_text()
    for m in re.finditer(r'\$\$(.+?)\$\$', t, re.S):
        body = m.group(1)
        for lm in re.finditer(r'<', body):
            if '\\lt' in body[max(0,lm.start()-10):lm.start()+10]:
                continue
            print(f"{p}:{t[:m.start()+lm.start()].count(chr(10))+1}")
PY
# 模式8：\operatorname 不在 GitHub 客户端 MathJax 宏白名单
rg -n '\\operatorname' <modified-markdown-files>
```

模式 2（斜体 `*…*` 图注内的裸 `$expr$`）rg 扫不到，目检图注。用 curl 抓 GitHub 页面数裸 `$` 时，排除 `aria-label="Permalink:` 属性——标题内公式的 permalink 会带字面 `$`，属页面模板自动生成的不可见属性，公式本身渲染正常，不算失败。

命中处理：模式 1 在开 `$` 前补半角空格；模式 2 改官方 `` `` $`expr`$ `` `` 形式；模式 3 改写为单参数命令省花括号形式（`$\hat{R}_t$` → `$\hat R_t$`）；模式 4 包 `<span>$</span>`；模式 5 星号改 `\ast`（`$h^*$` → `$h^\ast$`、 `$\|W_l\|_*$` → `$\|W_l\|_\ast$`）；模式 6 单反斜杠 `\_` 改双反斜杠 `\\_`，或改用 dollar-backtick 形式——双反斜杠命中后按段落做 CommonMark 强调配对检查（标点前缀 `_` 的两种开者形态 `\\_` 反斜杠、`'_{` 撇号都可开强调，同段 `_{` 型下标可闭强调，配对则 `<em>` 切碎两条公式），存在配对时 `\\_` 优先改写为连字符形式（`fan-in`、`fan-out`），`'_{` prime 下标把 prime 后移为 `$g_{i,t}'$`（`_` 变 close-only，LaTeX 渲染不变）；模式 7 裸 `<` 改 `\lt`（`x_{<t}` → `x_{\lt t}`）；模式 8 改 `\mathrm{...}`。修改图注时注意：斜体内公式必失败，粗体内公式可渲染。

## 审计与维护方法论

### 联网求证门槛（hard floor）

任何形式的"广查 / 盲查 / fresh audit / 三阶段循环 / 章节级复核 finding 收集"，**即使尚未动手修改任何文件**，按以下最小门槛执行：

1. 每个 sub-agent 在出 finding 之前，**必须**对涉及外部事实的部分（arXiv 编号、模型/硬件数字、benchmark 分数、版本号、日期、厂商声明、引用文献、HF model id）至少调用一次 WebSearch，并 WebFetch 一手页面（arXiv abstract / 官方 model card / HuggingFace `config.json` / 官方仓库 README / 技术报告 PDF）做实际验证。**不得**仅凭印象、训练数据记忆或课件文字给出 verdict。
2. finding 列表里每一条 `severity ≥ medium` 或 `needs_web_check=true` 的项目，必须在 evidence 字段附上：
   - 实际 WebSearch 查询字符串或 WebFetch URL；
   - 一手页面里的相关原文/数字摘录（≤ 2 行）；
   - 笔记正文里被质疑的那句话或数字（含 file:line）。
3. 没有附上述证据的 finding 视为 `tentative`；求证阶段按主题聚类 WebFetch 核证，仍无法被一手源支撑的应当降级或丢弃，不得计入最终 fix 列表。
4. sub-agent "建议加 X 链接 / 标注 Y 来源" 的建议，主 agent 在 commit 前**必须**自己再 WebFetch 核一次；历史经验：sub-agent 写出来的参考文献可用率约 70-80%，主 agent 必须复核。
5. 历史教训：曾出现 arXiv:1712.00409（实为 Hestness 等 Baidu 论文，并非"44× ImageNet 出处"）和 arXiv:2401.06865（实为天文学家 Osborne & Salim 的星系 SED 拟合论文）两例 sub-agent 幻觉引用。**任何 arXiv 编号进笔记前必须先 WebFetch abstract 页面确认标题与作者**，不接受"看起来对题"。

### 工具用法（WebSearch + WebFetch 并用）

- **WebSearch 与 WebFetch 必须并用**——只 WebFetch 不 WebSearch 是 sub-agent 的常见偷懒模式。每轮广查 / 审计每个 sub-agent 的硬性最小值：
  - WebSearch ≥ 5 次，覆盖"是不是真存在 / 是不是真作者 / 是不是真数字"三种目的
  - WebFetch ≥ 5 个不同 URL，其中至少 2 个为 `arxiv.org/pdf/{id}` 或 `/html/{id}` 的正文（非 abstract）；sandbox 对 `arxiv.org` abs 路径常返回「Unable to verify domain」、对 `/pdf/` 返回压缩对象流（FlateDecode streams，不可直接解析），实际可读正文只能走 `/html/{id}`；html 截断（附录 / Table / Figure 数值）或 sandbox 拦截时退到 `/pdf/{id}` 配合 `pdftotext -layout` 本地抽目标句前后 ≥6 行
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

4. **WebFetch 缺引文或被截断（`truncated` / `not visible`）时，该 finding 保持未核**，换 PDF 路径 `curl -sL arxiv.org/pdf/{id} -o /tmp/{id}.pdf && pdftotext -layout /tmp/{id}.pdf` 抽目标句前后 ≥6 行；附录与 Table 数字常在 html 截断区，直接上 PDF。完整实操模板见 `memory/webfetch-full-text-required.md`。摘要式 WebFetch 返回「XX 是 YY」但不给原文引文时，与 abstract-only 同等待遇——保持 `tentative`、补 PDF 抽文后才能下 verdict。

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
- **4. 段落 / 修辞层**：长段 > 8 行拆段；单段承载动作 > 2 个拆段；公式 / 表格 / 代码块后承接句；作者声音穿帮（"这里使用 / 本文采用 / 后文会 / 不等于所有 / 下一节展开"等）；禁用句式（"读法 / 看这类图时 / 不是 X 而是 Y"）；提示块类型合理性（5 种 alert 语义边界；节首学习目标与学习检查收束不用 alert，节首模板 rg 兜底见「修改后自检 3」）。

**B. 跨章节**

- **5. 术语 / 符号层**：同一概念在不同章节用同一英文术语；同一符号在不同章节指同一变量（ch3 `$d_k$/$h$` vs ch2 `$H$/$N$` 必须统一）。
- **6. 数字层**：同一硬件 / 模型 / benchmark 跨章一致（H100 BF16 dense 989.5 TFLOP/s、MMLU 12,032 题、Pile 334B tokens）；跨章引用的数字与那一章原文一致（Llama 3 70B tokens/param 215 vs ch10 15.6T → 223）；章节末「来源与更新记录」3 段结构（官方 / 课程 / 事实声明指向）。
- **7. 引用 / 跨章层**：跨章引用格式统一（第 N 章 §X.Y 章节标题）；章节内 caption / 图注 / 正文交叉引用一致；别章引用的章节标题与目标章节实际 H2/H3 一致。

每层对应 sub-agent 工作流：

| 层 | 工具 | 硬约束 |
|---|---|---|
| 1 / 6 | WebSearch ×5 + WebFetch ×5 | `search_query_1/2/3` 三字段不同关键词 |
| 2 / 5 | Read 章节全文 + lecture_NN.py | 符号首次出现即定义 |
| 3 | Read 本地 PNG | 核心图必读，不看 alt text |
| 4 | 修改后自检四类 rg | 禁用句式 / 作者声音 / 提示块 / 渲染安全 |
| 5 / 7 | Read 相邻小节 + 跨章引用目标章节 | 跨章引用格式「第 N 章 §X.Y 章节标题」 |
| 6 | 引用的别章 § 编号 + 章节号 | 与目标章原文核对 |
| 1-7 | 当场 Edit + 回读上下 5 行 | sibling 自洽检查 |

`introduced_by_commit` 字段是 fix verification 阶段的硬约束。

### 三阶段循环（盲查 / 求证 / 复核）

- **盲查阶段**——按 lecture 切分，sub-agent 报告 finding 列表（含 file、line、issue_type、severity、evidence、suggestion、web_evidence）。**每个 sub-agent 至少调用一次 WebSearch + WebFetch + Read PNG + 跨章节 Read**，对涉及外部事实的 finding 在 evidence 字段附 WebSearch 查询字符串 / WebFetch URL + 一手页面原文摘录（≤ 2 行）。
- **求证阶段**——仅针对 `needs_web_check=true` 或 `tentative` 的 finding 联网核证；可按主题聚类用 8-10 个 agent 并行 WebFetch / WebSearch。无法被一手源支撑的 finding 必须降级或丢弃。
- **复核阶段（fresh audit）**——从零重新读当前文件 + 对应课件 + 引用 PNG + 跨章节对照做 fresh audit，让每个章节独立得出 verdict；同样必须联网求证外部事实。这一轮最易发现前两轮漏掉的"原始事实"类错误（如 GPU 规格、KV cache 大小、样本数、vocab_size、训练 token 数）。**复核阶段不传盲查 finding 列表**，避免 anchoring bias。

**审计 verdict 三档规范（默认 schema）**：

- `confirmed` — 笔记对得上权威源，**没有问题**，**不报**
- `refuted` — 笔记与权威源**直接冲突，是错的**，要修
- `tentative` — 一手源证据不足，可能是对也可能是错，留到下轮

prompt 要求 sub-agent 只报 `refuted + tentative`、**不再报 `confirmed`**，可显著减少低信号 confirmed informational finding，把篇幅留给可执行的 refuted 修复。

**逻辑审计 finding schema（logic_finding，与 verdict 三档并用）**：

`rg` 只能抓句式骨架（禁用句式 / 作者声音 / 元叙述），抓不到推导跳步、经验伪装推导、结论无证据；逻辑质量由结构化 finding 字段强制，与 `search_query_1/2/3` 强制 WebSearch 同理——写在 prompt 里的方法论倡议无效，StructuredOutput 必填字段才执行。凡 finding 涉及论证质量（对应 STYLE.md「第一性原理方法论」三条硬约束），必须按 logic_finding 三字段报告：

- `claim`：被质疑的论断原文 + `file:line`；
- `axiom_source`：该论断的公理来源（公式 / 定理 / 物理约束 / 代码接口 / lecture 原文）；无推导路径、属经验或类比则标 `vibes`；
- `gap`：从公理到结论跳过的中间步骤，或经验判断被写成推导的原句位置。

logic_finding 只报 `refuted + tentative`，与 verdict 三档对齐。fix 阶段按方案 A 重新 WebFetch 一手源 + 重读 lecture 抽文 / `lecture_NN.py` 后就地修复，notes 附 `first-principle derivation: [路径]` 与 `lecture read: [路径列表]`。

### 审计-fix 一体化（方案 A）

每个章节的 audit 与 fix 由**同一个 sub-agent** 在两次明确分开的小阶段内完成，避免跨 agent 印象传染：

- **audit 阶段**：读章节全文 + lecture PDF + lecture_NN.py + 引用 PNG + 跨章节对照（按上方硬约束）；WebFetch 一手页面（论文 PDF 原文 / HF model card / NVIDIA datasheet / config.json）；产出 finding 列表，**只列 wrong_facts 与 evidence_in_lecture，不写 suggested_fix**——避免 audit 的修复方案成为 fix agent 的"印象传染源"。
- **fix 阶段**：每个 fix **必须重新 WebFetch 一手页面（不同 URL 或不同 search_query），不得复用 audit 阶段的 WebFetch 结果**。先读 lecture PDF + lecture_NN.py 原文与代码，**确认课件口径与一手页面一致后再写修复**。写完一个 fix 后**回读 fix 上下 5 行做上下文自洽检查**（sibling rows / sibling claims 是否仍成立），不一致则继续修。
- **不在 finding 里写 suggested_fix**：fix agent 不抄 audit 的修复建议，只看 wrong_facts + evidence_in_lecture + 一手 URL，自行 WebFetch 重新核验后写修复。这条切断"修复源头已污染"的链条——之前多次翻车的 Bartz 法官 / LIMO 800 vs 817 / LLaDA 2.0 "Tiwei Bie" 都是 audit 给的 suggested_fix 已经写错而被 fix 直接抄。
- **audit 与 fix 写在一个 agent 任务里**：避免 context 在两个 agent 之间序列化丢失；同一 agent 在 fix 阶段重新 WebFetch，避免 anchoring bias；audit 与 fix 之间必须**有清晰的任务结构与字段隔离**，agent 不能跳过 audit 直接 fix。

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
- **法官 / 公司 / 人物名必须逐字核验**：早期 fix 曾把 Bartz v. Anthropic 案承办法官凭印象写成「Judge Susan Illston, N.D. Cal.」（Illston 是 N.D. Cal. 真法官但不是本案承办法官；正确序列是 Judge William Alsup → Judge Araceli Martínez-Olguín，2025-12 Alsup 退休后案号由 3:24-cv-05417-WHA reassign 为 4:24-cv-05417-AMO，案件主体不变）；Llama 3 tech report 首作者在不同 arXiv 版本里是 Dubey et al. (v1/v2) → Grattafiori et al. (v3+，2024-11-23 起)；UltraFeedback 第一作者 Cui et al. 的 arXiv ID 是 2310.01377 不是 2310.01386。**任何具体人名 / 公司名 / 法官名 / 论文首作者 / arXiv ID 在写入前必须 WebFetch 一手页面核验作者列表与 ID 字面值**——abstract-only 复核或印象记忆都会写错，凭印象写「N.D. Cal. 法官」也能撞对真法官但错案号。
- **量化口径"看起来小但本质大"**：
  - ViT-bigG：OpenCLIP 官方 2.54B vs Qwen-VL 论文 Table 1 报 1.9B（去掉 head 的非 embedding 部分）；两个数字都对，但缺一句话解释读者就会困惑
  - LIMO/R1 蒸馏：800k = 600k reasoning + 200k non-reasoning，不应笼统说"800k CoT traces"
  - Mistral 7B：官方未披露训练 tokens；按多家二手估计 ~8T（≈1,100+ tokens/param），不是早期"110 tokens/param"
  **修正方式**：保持所有数字 + 显式说明口径差异，不要强行二选一。
- **per-SM vs per-GPU / dense vs with-sparsity 是 GPU 规格常见口径混淆**：A100 FP64 19.5 TFLOP/s 实际是 with-sparsity（dense 是 9.7）。**引用 GPU datasheet 时必须区分 dense / sparsity / per-SM / per-GPU 四个维度**，并在表头明确标注。
- **教材级章节归属幻觉（sub-agent 凭印象扩写论文细节）**：RFT 写成 "Reinforcing Fine-Tuning" 而 DeepSeekMath §5.2.1 原文是 "Rejection Sampling Fine-tuning"；Dr. GRPO baseline 改成 leave-one-out（实际论文 §3.2 只删 std 与 1/|o_i| 两个分母）；R1 §2.3.1 冷启动写"温度 1.0 + DeepSeek-V3 精炼 + LLM 批量扩展"（实际论文 §2.3.1 列出的是 long CoT few-shot / 直接生成详细答案 / R1-Zero 输出整理 / 人工后处理 四种）。**对近期（≤ 2024）论文涉及实验细节的引述，必须读 PDF § 正文，不能凭 abstract 或课件 PPT 一句话扩写**。
- **图意核对与删图判定**：核心图必须实际 `Read` 本地 PNG，按 STYLE.md「图意核对」4 项检查执行；删图判定、连号重排与 `rg 图 N-x` 悬空引用验证按 STYLE.md「图片」节执行（权威文本在 STYLE.md，此处不重复清单）。
- **tokenization 实证数据必须实际跑 tokenizer**：写 `tiktoken` / `sentencepiece` / `transformers` 类的"X 字符串切成 N 个 token id"示例时，**必须实际运行一遍 tokenizer**，把真实 id 序列与每个 id 对应的字符串写进正文 / 图说。印象记忆几乎一定会写错：ch1 §1.1 图 1.1-2 描述「年份 1885 作为 4 位数字整体成为 id 13096」错误，tiktoken 实际切分是 `[93447 Stan, 9201 ford, 673 ` was`, 24303 ` founded`, 306 ` in`, 220 ` `, 13096 `188`, 20 `5`, 13 `.`]`，1885 是 3+1 位两段而非 4 位整体。**所有 tokenizer / BPE / 词表实证段必须以 `python3 -c "import tiktoken; ..."` 或 Jupyter notebook 实际输出为准**。
- **fix verification round 抓到的 12 类反复错误**：每次 audit + fix 之后跑一轮 `fix verification` **全章扫描**（不是仅审本 commit 引入的行），下面 12 类反复出现（4a / 4b 同属第 4 类）：

| # | 类别 | 典型反例 | 检测方法 |
|---|---|---|---|
| 1 | **跨章引用错位** | 引「§X.Y 章节标题」时 X.Y 与目标章节实际 H2/H3 不对齐（ch8 L11 引「§8.6.8」陈旧但内容已前移；ch8 L1081 引「§8.4.3」但实际在 §8.4.2；ch14 L249 引「章节末总结」但实际在 §14.3 NOTE；topics L338 把 §3 RLVR 写成 §2） | grep「第 N 章 §」+ 对照目标章节实际 H2/H3 |
| 2 | **章节结构与节首描述矛盾** | 节首说「四个递进的例子」但实际只有三个（ch6 L256 §6.4 matmul 在 §6.5）；节首说「三阶段」但表格只有两行（ch7 L1287 §7.11）| 抓节首描述数字 vs 实际 H3 子节数 |
| 3 | **模型名 / 数据错 typo** | 「Qwen3-Next Coder」（不存在）→「Qwen3-Coder-Next」（ch13 L1209）；「FineWeb 用 Cuckoo 哈希」（sub-agent 误植，论文 §3.4 仅写 MinHash + LSH）→MinHash（ch10 L399）| grep 错误拼写 + WebFetch 一手论文核证 |
| 4a | **节首描述与一手论文术语不一致** | ch4 L1003 抓的 `device-level routing` 实际论文术语是 `device-limited routing` | 逐条 WebFetch 一手论文核证 |
| 4b | **节首描述与正文自相矛盾** | ch7 L439「§7.3 NCCL 报的 GB/s」与正文「NCCL 并不回报这个数字」直接矛盾 | 逐段对照节首 vs 正文 |
| 5 | **章节号归属错位** | 方法 1/2/3 都在 §8.4.2 不在 §8.4.3（图 8.6-12 错引）；Method X 写在 §8.6.8 但实际在 §8.4.2（图说）| 抓「§X.Y」+ 对照目标章节实际 H3 |
| 6 | **元叙述 / 写作计划承诺残留** | 「§1.3 接下来会再次出现」（forward-looking 承诺，ch1 L132）；「每章开头的『本章学习目标』会标出」但 14 章无一标注（不成立，preface L23）；「使用前应核对...再决定是否照搬」（搜证元句，ch4 L1069） | rg 抓「后续章节会 / 后文将 / 下一节会 / 下一节展开 / 下一节转到 / 本文采用 / 这里使用」 |
| 7 | **中段纯文字描述段漏扫** | ch1 §1.3 L175 教训：b8773b4 写「GPT-2 预分词正则大致是... DeepSeek 系列则使用 `\p{L}+|...`」三处错（"大致是"作者声音 / openai_public.py 引用错 / "在词内或数字串中间发生 merge"反了 `\p{N}{1,3}` 的实际行为），fix verification 当时只抓了 §1.3-1 / §1.3-2 / §1.3-2 三处图说（同一节内但都属「带图」段），**完全漏掉 L175 这段没有图的纯文字描述** | prompt 强制 agent 逐字读 §X.2 / §X.3 / §X.4 全段，对每条断言性内容逐条 WebFetch 一手核证 |
| 8 | **跨章引用缩写违规** | ch4 L112/537/547 写 `EP / ETP / EDP`，跨章引用应使用完整 H3 标题「第 N 章 §X.Y 章节标题」 | grep「ch[0-9]+」、「第 N 章 §」+ 缩写模式 |
| 9 | **sub-agent 报告机制漏洞** | ch1 L288 教训：audit-fix agent 报告「改前/改后」时，notes 里的「旧值」可能是已经改过的最终值而不是 git HEAD 改前值 | prompt 强制 agent 在 notes 字段附「`git show <commit-hash>:<file>:<line>` 改前值」+「`git show HEAD:<file>:<line>` 改后值」 |
| 10 | **结构性内容改动必读课程材料** | ch13 教训：agent 改了 80 行 PPO 详细推导（policy gradient → REINFORCE → TRPO → PPO-clip），但 transcript 显示 **0 次 Read lecture 抽文 / lecture_13.py**——agent 凭印象压缩，notes 写「Read ch12 §12.4 验证 PPO 推导已讲过」也是凭印象编的 | prompt 强制 agent 改结构性内容前必须 `Read` 课程材料，并把 read 路径写进 notes 字段「`lecture read: [路径列表]`」 |
| 11 | **论证必走第一性原理（Aristotle 框架）** | 每段论证必须满足三条硬约束（公理起点明确 / 推导链完整 / 经验 vs 推导清楚区分），权威文本、典型公理起点与 vibes 论域清单见 STYLE.md「第一性原理方法论（Aristotle 框架）」，此处不重复全文；第一性原理关心「论证是否从公理推导」，结构审计 5 问关心「位置是否成立」，两者并列互不替代 | 写章节或做章节级调整前按 STYLE.md 三条硬约束列出「推导来源」，notes 附「first-principle derivation: [路径]」与「vibes / 待推导: [路径]」；audit finding 按「逻辑审计 finding schema」报 claim / axiom_source / gap 三字段（见「三阶段循环」节） |
| 12 | **fix verification 误判「字段未公开」而整段删除** | Phase 11.3 ch4 agent 把「DeepSeek V4-Pro 演进」整段删除：表行「总参数 / 激活参数」两列写「未公开」→ agent 推断「不完整就别列」，连带把 §4.3 节首描述、§4.3.2「DeepSeek V4 的改进」整节、§4.6 表头与 V4-Pro 行、章节末来源记录的 V4-Pro config 全部删除；但 ch3 §3.2.5.7.3 与 ch14 都在引用 `DeepSeek-V4-Pro/config.json` 的 `index_topk: 1024` / `compress_ratios` / `swiglu_limit: 10.0`——删除破坏跨章一致性 | sub-agent 删前必须 `rg "<key>"` 全仓库扫引用方 + `git log -S "<key>" -- <file>` 看历史上是否有完整版本可恢复 + WebFetch 一手 config.json / 论文确认「未公开」字段是否真的不可得；不要把「数据稀疏」与「数据错误」混为一谈；同样警惕把 swiglu_limit / num_hash_layers / scoring_func 等「字段非主流」当成「应替换为更主流来源」 |

**每条都按方案 A 就地 Edit 修复**（不归到独立 round），并跑「修改后自检」四类 rg 零命中才算完成。

### 结构审计 5 问（与「第一性原理」并列、互不重复）

这是「每个结构单元是否挣到自己的位置」的工程审计框架（ch8 教训：图说 / 长公式 / 长算法描述改动时不能只做「去冗」），**与 class 11 第一性原理不同**——前者关心「位置是否成立」，后者关心「论证是否从公理推导」。两者并列使用：先用第一性原理确认论证来源（class 11），再用结构审计确认每个单元的必要性（5 问）。

5 问：

1. **它回答什么工程决策或论述节点？**（不是「展示什么」——是「读者看完能做哪个原本做不了的判断」）
2. **为什么放在这一节？**（哪段论述需要它才能继续往前推）
3. **不放这段内容，论述会怎么断？**（如果答案是「不会断」，可能它就是冗余的）
4. **正文（不是图说）现在有没有真正调用它？**（还是只贴了图说 / 表格就过去了）
5. **它是不是其他章节的引用目标？**（class 12 新增：如果其他章节在引这个 config 字段 / 模型名 / 数字，删前必须先同步更新引用方）

判定规则：4 / 5 个「不需要」→ 可删 / 大幅压缩；任一是「需要」→ 按第一性原理重写（不是「去冗」），让第一句直接答工程问题、中间写关键元素与正文对应、最后指向章节链下一步。

检测方法：删前必须 `rg "<key>" docs/` + `git log -S "<key>" -- <file>` 看历史。Phase 11.3 ch4 V4-Pro 反例：ch3 / ch14 在引 V4-Pro config，但 ch4 agent 仅扫自己负责的 ch4，没看其他章节怎么引用 V4-Pro。

### Workflow 完成后必须先 diff 复核再 commit

- **多 agent 并行 fix verification 经验**：单章节 fix verification 单独跑一次 agent，16 章节并行启动 16 个 agent，每个 agent 做**全章扫描**（用 `git log` 看最近 commit 历史做参考，但不限于本 commit 引入的行——参见第 7 类反复错误，「中段纯文字描述段漏扫」就是限定本 commit 引入行时漏掉的真实案例）。Schema 仍用简化版（findings_count / edits_applied / fixes / notes），避免 SO retry 超限。16 agent 并行总耗时约 13 分钟，比串行快 16×；findings 数量比单章节 fresh audit 多（因为每条 finding 都覆盖中段纯文字 + 图说 + 表格），但每条都是「本轮可修复的真错」而非历史包袱（agent 必须用 `git log` 过滤历史问题、只对当前 commit 之后或本 fix 阶段引入的行下 Edit）。

## 工具使用经验

- **sub-agent 偏好 WebFetch 而忽略 WebSearch** 是持续性问题——但有解法。早期 prompt 要求"WebSearch ≥ 5~8 次"实际只跑 ~2 次 / agent；改 schema 把 WebSearch 拆为 finding 必填的 `search_query_1` / `search_query_2` / `search_query_3` 三字段后，平均跃升到 ~12 次 / agent（约 6×），并一次性抓到 11 处 fix 留下的次生错误（arXiv ID 归属、论文小节归属、法官名、数字精度等）。**结论**：单纯在 prompt 写"WebSearch ≥ N 次"无效，必须把 WebSearch 列到结构化输出必填字段才能强制执行。三个 search_query 还要求**不同关键词**（不能 3 次都搜同一个词）。
- **fix 本身可能错**：次生错误模式与权威流程见「fix verification round（次生复核）」节及下方「次生修复三类模式」；每轮 audit 后必须再跑 fix verification，schema 强制带 `introduced_by_commit` 字段。
- **Schema 复杂度 → agent 失败率**：曾出现 ch5 agent 因 StructuredOutput retry cap 5 次超限而失败（跨章节 audit 也多次出现同一模式）。**结论**：finding 字段不要塞太多嵌套对象，能合并就合并；web_evidence 简化为 `search_query + fetch_url + fetch_snippet + verdict` 四字段就够，不要加 `source_kind` / 多个 evidence 对象。但 `search_query_1/2/3` 三字段是必须的例外——这是强制 WebSearch 的有效手段。
- **SO retry 超限 → 立即简化 schema 重跑**：跨章节 audit 与其他多文件 / 多来源 finding 类任务的 schema 含 `files: [...]` / `line: {file: N}` / `web_evidence` / 多文件嵌套对象时，agent 容易 SO retry 5 次失败，但 Edit 已经落地、finding 列表未返回。**结论**：这类任务 schema 必须简化到 ≤ 4 个顶层字段。简化版本用三字段：`findings_count` / `edits_applied` / `notes`，足够驱动 commit 与进度报告。Edit 不需要被 schema 验证，仅最后 SO 必须返回。**重跑触发条件**：journal 报 `StructuredOutput retry cap (5) exceeded` 或 `subagent completed without calling StructuredOutput`。
- **次生修复三类模式**（每轮 audit 后必须再跑 fix verification round）：
  - **数字回归 revert**：上次 commit 把对的数字改成错的（如 Singhal LPPO Stack 59%→58%、Stiennon 标题小写→Title Case），下次 audit revert 之前要确认原值与论文原文一致，不是把错的二次 revert 回对的。
  - **casing 归一化漏扫**：上次 commit 修一处 casing（如 CyBench→Cybench），其他章节残留（SWEBench / MLEBench / SWE-Bench Verified）要在同一 commit 内 `rg` 全仓库扫一遍。
  - **修复引入新错**：上次 commit修一个低阶项时把另一个数字写错（如 8×→4× 时把 `5MN+M` 漏掉 `+2M` 低阶项），下次 audit 必须回读上下 5 行做 sibling 自洽检查。
- **图片 SHA256 不一致不一定是不同图**：SHA 不同但视觉内容相同的图（如课件截图 vs 论文原图、同一 PDF 不同页面截图）都是「同一份视觉信息的不同版本」，应判为重复；删图判定用视觉内容，不只用 SHA。
- 课件 PDF 在 agent 上下文里读会触发图片上限（一次 Read 可能塞入几十张 page-image），先用 `pdftotext -layout` 抽成纯文本到 `sources/_extracted_pdfs/lecture_NN.txt` 供 agent 线性读取；该目录作为维护副产物，不纳入 git（已在 `.gitignore` 或不入库）。
- Workflow 脚本读取运行参数时，harness 把数组 args 包成 `{item: [...]}`，pick-args 函数要同时处理 string、数组、`{item}` 和 `{cluster}` 四种 shape，否则会出现"9 个 agent 都退化成对同一文件 audit"的浪费。
- **arXiv 论文 WebFetch 优先 `/html/{id}`，不要 `/abs/` 或 `/pdf/`**。sandbox 对 `arxiv.org` 域名的 `abs` 路径返回 "Unable to verify if domain arxiv.org is safe to fetch"；对 `/pdf/{id}` 返回 binary "FlateDecode streams"（压缩对象流）而非可读文本；只有 `/html/{id}` 能返回标题、作者、abstract、§ 章节标题与原文句子。所有求证一手 arXiv 内容（标题、作者、提交日期、§ 编号归属、原文引用句）都先 `WebFetch https://arxiv.org/html/{id}`；html 解析失败或需要 Figure/Table 数值时再退回 `/pdf/{id}` 配合本地 `pdftotext` 抽文。WebSearch 在 sandbox 拒 arXiv 域时仍能返回关键句摘录，但只能用来定位 URL / 确认存在性 / 拿关键词，不能替代 `WebFetch /html/{id}` 作为一手引用。
- **Workflow 工具不接受自定义顶层参数（如 `scope: 'single-task'`）**。Workflow 工具的合法顶层字段只有 `script` / `name` / `description` / `title` / `args` / `scriptPath` / `resumeFromRunId`；任何额外字段（如 `scope`、`task_type` 等不在 schema 内的字段）会触发 `InputValidationError: Workflow failed due to the following issue: An unexpected parameter ... was provided` 整条调用被拒，需要去掉多余字段后重发。**记忆**：调用 Workflow 前如果想加并行度控制（如「single-task」「fan-out」「sequential」等）应当写到 `meta.phases` 与脚本内的 `phase()` / `parallel()` / `pipeline()` 编排，而不是加 Workflow 工具顶层参数。
- **fix agent 不要并行跑同一文件**（ch3 全量调整会话教训，2026-09-15）。原本设计 5 个并行 fix agent 按主题切片（§3.1 / §3.2.5 / §3.2.1-3.2.4 / §3.2.5.8.1 / 图号重排），其中 4 个 fix agent 都涉及 `docs/chapter3/chapter3_语言模型架构和训练技术细节.md`，F1 还涉及 ch4/ch14——多个 agent 同时 Edit 同一文件会导致：(a) Edit 操作的行号在另一个 agent 改完后漂移、anchor 失效；(b) sibling rows 自洽检查不可靠（一个 agent 看到的 sibling rows 已被另一个 agent 修改）；(c) rg 自检的命中行号在不同 agent 之间错乱；(d) commit 拆分困难（一类改动只能对应一个 commit）。**正确做法**：fix 阶段按文件隔离（每个 agent 只负责一个文件的所有 fix）或完全串行跑（`phase('Fix A') → agent → rg 兜底 → phase('Fix B') → agent → ...`）。**Workflow 编排时**，`phase()` 串行 + 中间跑 rg 兜底比 `parallel([fixA, fixB, fixC])` 更稳。**Audit 阶段并行可以**（每个 sub-agent 只读不改，无写冲突），**fix 阶段必须串行或按文件严格隔离**。
- **audit/fix sub-agent 与主 agent Edit 前必须先 Read STYLE.md 全文 + AGENTS.md「与 STYLE.md 的同步锚点」节**（ch3 全量调整会话教训，2026-09-15）。prompt 必须显式要求 `Read /home/jiepengjin/Study/llm-notes/STYLE.md` 并按锚点表逐项刷新（禁用句式 rg / 作者声音 rg / 5 种 alert 语义 / 3 段来源结构 / 「第 N 章 §X.Y 章节标题」/ 图号规则 / 第一性原理框架 / 图意核对 4 项 / tentative 与待核验类元叙述不进笔记）；只写「按 AGENTS.md 规范」不够——AGENTS.md 是流程规范，STYLE.md 才是写作规范本体，二者需同时读。无论 Edit 大小 / 范围，此步是必要条件，不可跳过。
- **sandbox 拦截一手 URL 时不要自己 hallucinate，交给用户**（ch3 fresh audit 会话教训，2026-09-15）。本会话中 A3 audit agent 把 sandbox 拦截的 `huggingface.co / arxiv.org /html/` 当作「无法核证」，随后自己推测「MiniMax-01 真实 arXiv ID 是 2501.05402」并把笔记里正确的 `arXiv:2501.08313` 报为 hallucinated——实际 `arXiv:2501.05402` 是物理论文「Edge modes in modulated metamaterials」，MiniMax-01 真实 arXiv ID 就是 `2501.08313`（标题「MiniMax-01: Scaling Foundation Models with Lightning Attention」，作者 Li, Aonian 等，2025-01-14，MiniMax 团队）。**正确做法**：当 `WebFetch arxiv.org / huggingface.co / github.com` 等一手 URL 被 sandbox 返回 `Unable to verify if domain ... is safe to fetch` 或 curl 超时时，sub-agent 与主 agent 都应：(a) **不要自己推测 / hallucinate** 一手内容；(b) 直接告诉用户「需要复核 URL X，请帮我复制页面内容（或标题/作者/相关数字）」；(c) 让用户在 IDE 浏览器或 curl 外部环境拿到一手页面后粘贴进来，再继续核证。fallback 顺序：用户手动提供 > 仓库已有 `sources/_extracted_pdfs/lecture_NN.txt` 抽文 > 本地 HuggingFace cache > 留 tentative 并显式标 `web_evidence_fetched: false` 而非 hallucinate。**反例**：A3 报「A3 critical: 把模型名改正为 X / arXiv ID 改正为 Y」，但实际 Y 是另一篇完全无关的论文。
- **主 agent 自己 Edit 后也必须强制跑四类 rg 自检**（ch3 全量调整会话教训，2026-09-15）。`d9558aa` 主 agent Edit 后没跑自检直接 commit，引入「不是公理而是」骨架，被用户在 IDE 选中才发现；`d2b57c8` 才补删。**正确做法**：主 agent 任何 Edit 后强制跑「修改后自检」四类 rg（禁用句式 + 作者声音 + 提示块 + 渲染安全）+ 回读上下 5 行做 sibling 自洽检查再 commit，与 sub-agent fix agent 同纪律。写「经验 vs 推导」标注时下意识出现「不是 X 而是 Y」骨架的概率更高；Edit 后第一件事就是 `rg -n "不是.*而是|是.*而不是|不是.*而非|不是简单的.*而是" <file>` 兜底。
- **主 agent commit 前必须对每条 sub-agent refuted finding 做全量 WebFetch 复核（不是抽样）**（Phase 14.1 教训，2026-09-22）。审计-fix 一体化（方案 A）让 sub-agent 自行负责核证，但 AGENTS.md L115 明确「sub-agent 写出来的参考文献可用率约 70-80%」，主 agent 必须在 commit 前亲自复核——**不能抽样**。本轮 Phase 14.1 dispatch 15 个 sub-agent，主 agent 抽样复核后发现 3 处 sub-agent「修复引入新错」：(1) ch4 A4 把 V2 shared expert 数从 2 改为 1（DeepSeek-V2 论文 §2.1.2 明确写「2 shared experts」，A4 误用 V3 / V4-Pro 的口径套到 V2），commit `6436ec2` revert；(2) ch8 A8 把 TP V µP Table 3 中 SGD / Adam 在 hidden / output matrix 上的 LR factor 翻转（A8 fix 后「SGD hidden = 1/n_{l-1}」实际应为 Θ(1)），commit `30b7efd` revert；(3) topics A15 互换 image path 互换错——原笔记 caption 与文件内容全对（reasoning-01 = 9.169 vs 9.6；reasoning-02 = last-letter concat），A15 fix 后 caption 与 path 全错，commit `71fb596` revert。**正确做法**：主 agent 对每条 refuted finding 的关键 web_evidence（不是抽样）做独立 WebFetch / WebSearch 复核一手 URL；(a) 涉及 arXiv ID 时按「标题 / 作者 / 提交日期 / § 数字 / config.json 字段」五项核对，**禁止凭「sub-agent 引用了 arXiv 编号」就放过**（sub-agent 可能记错归属）；(b) 涉及 image path 时按 Read PNG 实看图意核对（PNG 内容 vs caption vs 文件名三方一致）；(c) 涉及跨章数字 / 引用时在所有 touched files 间扫一致（`rg "<key>" docs/`）。发现 sub-agent 引入新错则立即 revert 该 fix commit + 打 patch（不依赖 fix verification round 在下一轮抓）。Phase 14.2 跨章一致性与 Phase 14.3 fix verification round 是兜底机制，不是 commit 前纪律的替代。
- **Sub-agent 不应自己 commit；改动留在 working tree 由主 agent commit + 复核**（Phase 14.1v2 教训，2026-09-22）。sub-agent 在 audit-fix 一体化方案下会做大量 Edit 并触发 commit（Phase 14.1v2 中 A1 / A2 / A3 / A4 / A5 / A6 / A8 / A9 / A10 / A11 / A12 / A13 / A14 / A15 共 14 个 sub-agent 中，**A8 ch8 自己 commit `a52b573`**，A1 ch1 self-commit `ddffb82`，A2 ch2 self-commit（含 2 个 image 文件 rename 落在 `f812969` commit）。**问题**：self-commit 跳过主 agent 在 commit 前必做的全量 WebFetch 复核——主 agent 必须主动 `git log` + `git show <sha>` 找回这些 commit 复核，否则 sub-agent 的「修复引入新错」会溜进 main。本轮 Phase 14.1v2 主 agent 在 self-commit 后才复核，发现：(1) **A8 ch8 `a52b573` 把 TP V §8.6.6 Adam hidden LR 写成 `1/fan_in`**——实际 Table 3 原文是 Adam hidden LR = 1 (Θ(1))，与已 revert 的 v1 A8 错位同源；(2) **A1 ch1 `ddffb82` 图 regen 内容**(实际 tiktoken 输出与 image 内容一致，✓ 通过)；(3) **A8 ch8 `a52b573` Mistral 7B 8T+ tokens 声称三源核证**——实际主 agent WebFetch arxiv abstract + HF model card 都查不到 8T 数字，可能是 hallucinated 或引错模型（**待核证**）。**正确做法**：(a) sub-agent prompt 必须显式说明「**不要自己 commit / git commit / git push**；所有改动留在 working tree，由主 agent 统一复核 + commit」；(b) 主 agent 必须在每轮 audit-fix 完成后跑 `git log` + `git status --short` 列出**所有**新 commit（包括 sub-agent self-commit 与主 agent commit）+ `git show <sha>` 复核每个 diff；(c) 发现 sub-agent 引入新错则立即 revert + 打 patch（与「全量 WebFetch 复核」节同流程）。**禁止妥协**：sub-agent 不应跳到 commit 阶段；主 agent 不应跳过对 self-commit diff 的复核直接进入下一阶段 dispatch。
- **并行 Read PNG 可能异步错位返回**（Phase 14.1 ch13 教训，2026-09-22）。Read 工具在同一消息中并行调用多张 PNG 时，返回的图像内容可能与所请求的文件名不对应——本轮 Phase 14.1 中，主 agent 在单条消息里并行 Read `13-4-6 / 13-4-7 / 13-4-10 / 13-4-12` 四张 PNG，返回的第一张实际是 s1 Table 1（应该属于 13-4-12）而非 R1 训练流水线（应该是 13-4-6）。A13 也基于类似的并行 Read 错位报告「图 13.4-6 = 简化 4 步 + Key differences」并据此改了 4 张图 caption——全部反向错位（A13 fix 后 4 张图 caption 与文件实拍图意全错）。**正确做法**——**严禁任何 PNG 并行 Read**：(a) 同一张图只 Read 一次，不重复；(b) 一条消息里**只能 Read 1 张 PNG**，等 transcript 拿到结果并做文件名 vs 图意二次核对后，再起下一条消息 Read 下一张；(c) 每张图 Read 后立即在 transcript 上做**文件名 vs 实际图意**二次核对，不靠 memory；(d) 对图意与 caption 不一致的图片，必须**再单独 Read 一次**才能下结论，避免并行错位引发的 caption 错位 fix。**反例**：本轮 1 条消息里并行 Read 4 张 PNG → 主 agent 与 A13 都错位 → A13 据此改了 4 张图 caption → 4 处全部反向（最终 revert 到 HEAD 原版）。**禁止妥协**：哪怕「2-3 张也不并行」，因为无法保证 2 张以上不异步错位。
- **笔记任何位置不留「待核验 / 不确定 / 需复核 / tentative」类元叙述**（ch3 fresh audit 教训，2026-09-15）。核验工作外移到 chat / sub-agent transcript；sandbox 拦截一手 URL 时由主 agent 交用户在 IDE 浏览器或 curl 取回内容再核验；未核到的字段不写入笔记，已写入的撤回。章节末「来源与更新记录」只有 3 段结构（官方 / 课程 / 事实声明指向），无「不确定项」子段（曾用 4 段结构 + tentative 子段，已废除）。commit 前 `rg "待.*核|待核|不确定|需核|需复核|缺口|以.*为准|本节以|未能核"` 命中清零。

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

- 默认直接在 main 上 commit + push，不开分支，包括大批量改动与长期 feature；单次改动、跨章改动、流程规范更新都直接落到 main。
- 提交标题采用 conventional commits：`docs:` / `chore:` / `fix:` / `refactor:` 等小写前缀 + 冒号空格 + 祈使句动作（如 `docs: refine inference systems chapter`）。
- 正文（commit body）可选；非平凡改动应简要列修改重点或风险提示。
- **不**附加 `Co-Authored-By: Claude Code ...` / `Generated with ...` / `🤖 Generated with ...` 之类 trailer；本仓库以维护者本人为唯一署名。
- 一个 commit 只对应一类改动；如果同时动 `docs/` 和 `sources/`，按"先 docs 与内容，后 chore 与构建产物"拆 commit。
- 修改后跑「修改后自检」一节的 4 类 rg 与检查，确保新提交无命中。
- `CLAUDE.md` 与 `AGENTS.md` 内容保持一致；修改 `AGENTS.md` 后须 `cp AGENTS.md CLAUDE.md` 一并提交。

## 与 STYLE.md 的同步锚点

| AGENTS.md 节 | 对应 STYLE.md 节 | 同步关系 |
|---|---|---|
| 修改后自检 1（禁用句式 rg） | STYLE.md「验证流程不进正文」+「全局原则」+「修辞层规则」 | rg 关键字清单与 STYLE.md 反例一一对应 |
| 修改后自检 2（作者声音穿帮 rg） | STYLE.md「修辞层规则」 | rg 关键字清单来自 STYLE.md 反例（含「这里使用」「下一节展开」等）；audit finding 用 `author_voice_break` 字段报告 |
| 修改后自检 3（提示块合理性） | STYLE.md「提示块」 | 5 种 alert 语义边界与 STYLE.md 对齐（IMPORTANT/WARNING/CAUTION 三档严格递进，NOTE 不承载作者声明）；节首学习目标 / 学习检查不用 alert 与节首模板 rg 兜底与 STYLE.md「提示块」补充规则一致 |
| 审计 7 层第 1 / 6 层（事实 / 数字） | STYLE.md「公式与排版」+「公式账本」 | 公式符号、跨章节数字一致性 |
| 审计 7 层第 2 层（公式账本） | STYLE.md「公式账本」 | 符号首次出现即定义、公式与代码对齐、公式后承接句、长算式拆分、不允许 snake_case 进数学环境 |
| 审计 7 层第 3 层（图意） | STYLE.md「图意核对」+「图片」 | 核心图必读本地 PNG、6 类图文不符、删图判定 |
| 审计 7 层第 4 层（段落 / 修辞） | STYLE.md「段落与承接」+「修辞层规则」+「修辞层规则」反例清单 | 长段拆段、单段承载动作 > 2 拆段、公式 / 表格 / 代码块后承接句、作者声音穿帮 |
| 审计 7 层第 5 层（术语 / 符号） | STYLE.md「术语表」+「全局原则」 | 同一概念用同一英文术语、同一符号指同一变量 |
| 审计 7 层第 6 层（数字） | STYLE.md「来源与日期」+「章节末『来源与更新记录』约定」 | 同一硬件 / 模型 / benchmark 跨章一致；3 段结构（官方 / 课程 / 事实声明指向；不确定项不留在笔记，由 chat 流程核验） |
| 审计 7 层第 7 层（引用 / 跨章） | STYLE.md「跨章引用格式」+「图片」alt text 一致 | 跨章引用「第 N 章 §X.Y 章节标题」、caption / 图注 / 正文交叉引用一致 |
| 跨章节一致性 audit | STYLE.md「学习式讲义」「术语表」 | 主题组（GPU 硬件 / Transformer / MoE / 数据 / 评测 / 推理 / 强化学习）的术语对照 |
| 引用与求证经验（章节归属 / 人物名 / arXiv ID） | STYLE.md「术语表」「来源与日期」 | sub-agent 在命名 / 引用 / 章节号层面遵守同一标准 |
| 图意核对 / 删图判定 | STYLE.md「图意核对」「图片」 | 4 个检查项 + 两图视觉内容一致即可删 |
| 修改后自检 1（禁用句式 rg）与第 7 层 | STYLE.md「跨章引用格式」 | rg pattern 抓「第 N 章 §X.Y」与「chapterN」缩写 |
| 修改后自检 4（GitHub 渲染安全 rg） | STYLE.md「公式与排版」渲染条目 +「HTML」金额 span 例外 | 8 种渲染失败模式（开 `$` 紧贴 / 斜体图注 / `}_`·`|_` 切碎 / 字面金额 / math 内裸 `^*`·`_*` / math 内 `\_` 单反斜杠 / display 内裸 `<` / `\operatorname` 宏禁用）的 rg 与修复规则一一对应；图注用 `` `` $`expr`$ `` ``、金额用 `<span>$</span>`、星号用 `\ast`、下划线用 `\\_`（标点前缀 `_` 含 `\\_` 与 prime `'_{` 两种开者，同段 `_{` 配对时 `\\_` 改连字符 `fan-in`、prime 后移 `g_{i,t}'`）、小于号用 `\lt`、函数名用 `\mathrm` |
| 逻辑审计 finding schema（logic_finding） | STYLE.md「第一性原理方法论（Aristotle 框架）」三条硬约束 | `claim` / `axiom_source` / `gap` 必填字段与「公理起点明确 / 推导链完整 / 经验 vs 推导清楚区分」一一对应；`rg` 只作句式兜底，不承担逻辑审计 |
| 章节末「来源与更新记录」段外部来源 URL 完整链接 | STYLE.md「章节末『来源与更新记录』约定」新增 hard rule | 来源段每条 arXiv ID / HF ID / 官方 URL / DOI / 案件号都挂完整 markdown 链接，格式与正文一致；裸 ID 仅当紧邻前一行已挂同一链接时可省；混用裸 ID 与完整 URL 视为不一致 |

AGENTS.md 与 STYLE.md 任何一侧调整规则时，另一侧必须同步引用对应章节；同一 commit 内完成。