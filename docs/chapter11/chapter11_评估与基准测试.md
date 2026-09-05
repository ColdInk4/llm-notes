# 第 11 章 评估与基准测试

## 本章学习目标

读完本章，读者应能：

- 区分四类常见的"好模型"定义（绝对能力 / 相对排序 / 任务得分 / 安全性）及其适用场景。
- 描述 perplexity 作为通用语言建模评估的优缺点：可重复、与训练目标对齐，但与下游任务相关性弱。
- 列出 MMLU / GPQA / Humanity's Last Exam 等 exam 基准考察的能力维度，并说明 contamination 对它们的冲击。
- 区分聊天基准（Chatbot Arena / AlpacaEval / WildBench / IFEval）与 agent 基准（`SWE-bench` / Terminal-Bench / Cybench / MLE-Bench）在输入分布和评分规则上的差异。
- 解释 evaluation rules of the game：评分规则、prompt 范围、采样参数、工具与预算不同，分数不能直接横比。

评估的核心问题是**给定一个固定的模型，它到底有多“好”？** 这个问题看似只是打分，实际需要先定义使用目标、输入分布、调用方式和评分规则。评估不仅衡量当前模型，也会反过来影响训练目标、数据选择和产品决策。

评估可以看成一个映射问题：把“好模型”这类抽象目标转化为可执行、可比较、可复核的具体指标。困难也正在这里：不同使用者关心的“好”并不相同，采购模型、改进训练、研究能力边界和评估社会风险，需要的评估设计往往不是同一套。

因此，本章把评估拆成一条可复查的链路：先定义目标行为，再选择输入分布、调用方式、评分规则和解释边界，最后判断这个分数是否能代表真实使用场景。榜单截图和基准案例都服务于这条链路。

## 本章主线

本章按 evaluation 主线组织：先问“好模型”这个抽象构念到底指什么，再看 perplexity、考试基准、聊天基准、agent 基准、纯推理、安全、真实性和有效性。榜单图保留为教学材料，用来拆解分数、成本、真实使用和评估失真之间的关系。

一个评估在比较之前要先定义 rules of the game：比较的是裸模型、模型加工具、agent scaffold，还是完整产品系统；允许哪些 prompt、采样参数、工具和预算；分数如何聚合。规则不同，榜单分数和案例结论就不能直接横向比较。

下面这张速查图把本章七类评估与对应小节对齐，方便阅读时随时定位当前段落。

```text
                  ┌────────────────────────────────────────────────┐
                  │  第 11 章 评估与基准测试 — 七类评估速查            │
                  └────────────────────────────────────────────────┘

  ┌─ 抽象层：什么是「好」              ┌─ 指标层：用什么分数衡量
  │  § 11.1  四类「好模型」定义        │  § 11.3  perplexity（分布拟合）
  │  § 11.2  difficulty / realism /   │  § 11.4  exam（MMLU / MMLU-Pro /
  │           validity 三维度          │           GPQA / HLE）
  │  § 11.11 method vs model/system   │  § 11.5  chat（Arena / AlpacaEval /
  │           rules of the game       │           IFEval / WildBench）
  └───────────────────────────────────┴────────────────────────────────────
                                       │
                                       ▼
  ┌─ 任务层：模型在做什么              ┌─ 风险层：评估本身的可信度
  │  § 11.6  agent（SWE-bench /      │  § 11.8  safety（HarmBench /
  │           Terminal-Bench /        │           AIR-Bench / GCG jailbreak）
  │           Cybench / MLE-bench）   │  § 11.9  realism（GDPval / Clio /
  │  § 11.7  纯推理（ARC-AGI）        │           MedHELM）
  └───────────────────────────────────┴────────────────────────────────────
                                       │
                                       └─→ § 11.10  contamination / 标注噪声
                                              跨七类的共同失真来源
```

阅读顺序建议：先读 §11.1-§11.2 建立 evaluation 的目标与三维度框架，再沿指标层（§11.3-§11.5）逐类看分数如何被构造，最后看任务层（§11.6-§11.7）和风险层（§11.8-§11.9）；§11.10 与 §11.11 放在最后，是跨七类评估的共同约束。

## 11.1 简介

> [!NOTE]
> 本节解决三个前置问题：「好模型」有几种常见定义、公开榜单截图能在多大程度上传递信息、社区案例观察为什么不能替代系统评估。读完后，读者应能在看到任何一份 LLM 评估材料时，先定位它属于哪一类，再追问它测到了什么。

当你打开任何一个大语言模型（LLM）的评测网站或论文时，最容易看到的是一串分数、一张排行榜或几条社交媒体上的惊艳案例。本节先把这些表象拆成四类问题，再追问它们各自到底测到了什么。

### 11.1.1 四种常见的“好模型”定义

第一种定义是**基准测试分数高**。这类图把多个 benchmark 汇总成一个能力指数，方便快速比较，但也容易隐藏 benchmark 选择、权重和饱和问题。

![图 11.1-1 Artificial Analysis 能力榜单](images/11-1-1-artificial-analysis-capability.png)

*图 11.1-1 Artificial Analysis 能力榜单*

第二种定义是**能力高且便宜**。同样的准确率如果需要更高价格、更高延迟或更难部署，对很多产品场景并不是更好的选择。

![图 11.1-2 Artificial Analysis 能力与成本对比](images/11-1-2-artificial-analysis-cost.png)

*图 11.1-2 Artificial Analysis 能力与成本对比*

第三种定义是**人类更偏好它的回答**。Chatbot Arena（现改名 LMArena）通过匿名两两比较来估计偏好，但偏好会混合正确性、风格、长度、礼貌程度和用户分布。

![图 11.1-3 LM Arena 人类偏好榜单](images/11-1-3-lmarena-preference-leaderboard.png)

*图 11.1-3 LM Arena 人类偏好榜单*

第四种定义是**真实用户愿意持续使用或付费**。OpenRouter 这类使用量排名更接近市场反馈，但它同时受价格、可得性、生态集成、默认路由和社区热度影响。

![图 11.1-4 OpenRouter 使用量排名](images/11-1-4-openrouter-usage-ranking.png)

*图 11.1-4 OpenRouter 使用量排名*

> [!NOTE]
> 这四类定义都合理，但没有任何一类等同于“模型本身绝对更好”。评估要先说明目标：比较研究方法、采购产品模型、评估 agent 系统，还是衡量社会风险。目标不同，输入分布、调用方式、评分规则和解释边界都会变化。

### 11.1.2 静态榜单截图的使用边界

下面这些截图仍然保留在本章中，用来说明公开评估材料会混合多个层次的信息。静态名次只作为当时快照。分析这类材料时，先看 benchmark 组合，再看模型调用规则，最后看分数是否包含成本、延迟和真实场景约束。

![图 11.1-5 DeepSeek-R1 基准性能](images/11-1-5-deepseek-r1-benchmarks.png)

*图 11.1-5 DeepSeek-R1 基准性能*

[DeepSeek-R1 论文 Figure 1 报告的基准性能](https://arxiv.org/pdf/2501.12948)，展示了 DeepSeek-R1 在 AIME 2024、Codeforces、GPQA Diamond、MATH-500 等基准上的性能。

![图 11.1-6 Llama 4 基准性能](images/11-1-6-llama4-benchmarks.png)

*图 11.1-6 Llama 4 基准性能*

[Llama 4 Behemoth 指令微调版本的基准性能](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，展示了其在 MMLU-Pro、MATH-500、GPQA 等多个基准上的表现。

![图 11.1-7 OLMo-2 基准性能](images/11-1-7-olmo2-benchmarks.png)

*图 11.1-7 OLMo-2 基准性能*

[Allen AI 团队在技术报告中展示的 OLMo-2-32B 模型基准性能](https://allenai.org/blog/olmo2-32B)，主要覆盖 MATH、MMLU、DROP 和 GSM8k 等基准。

下面是 [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)、OpenCompass、SuperCLUE、Artificial Analysis、OpenRouter 和 Chatbot Arena 等平台的截图例子。它们有的强调能力覆盖，有的强调中文或区域性任务，有的强调成本或用户偏好。

![图 11.1-8 HELM capabilities 排行榜](images/11-1-8-helm-capabilities-leaderboard.png)

*图 11.1-8 HELM capabilities 排行榜*

![图 11.1-9 OpenCompass 排行榜](images/11-1-9-opencompass-leaderboard.png)

*图 11.1-9 OpenCompass 排行榜*

![图 11.1-10 SuperCLUE 排行榜](images/11-1-10-superclue-leaderboard.png)

*图 11.1-10 SuperCLUE 排行榜*

![图 11.1-11 Artificial Analysis 排行榜](images/11-1-11-artificial-analysis-leaderboard.png)

*图 11.1-11 Artificial Analysis 排行榜*

![图 11.1-12 Artificial Analysis 性能与成本对比](images/11-1-12-artificial-analysis-performance-cost.png)

*图 11.1-12 Artificial Analysis 性能与成本对比*

![图 11.1-13 不同厂商 token 使用排行](images/11-1-13-openrouter-token-usage.png)

*图 11.1-13 不同厂商 token 使用排行*

![图 11.1-14 Chatbot Arena 分数排行榜](images/11-1-14-chatbot-arena-leaderboard.png)

*图 11.1-14 Chatbot Arena 分数排行榜*

> [!WARNING]
> 这组静态榜单截图的用途是拆解榜单分数：先看 benchmark 组合和模型调用规则，再看分数是否包含成本、延迟、工具权限和 agent scaffold。模型版本、提示模板、系统 prompt、采样参数和计价方式都会改变分数含义。

### 11.1.3 评估危机与社区案例观察

还有一种非正式评估来自社交媒体和社区口碑：人们会分享模型令人惊艳或翻车的案例。这些案例有启发性，但样本量小、选择偏差强，不能替代系统评估。

![图 11.1-15 评估危机观点](images/11-1-15-evaluation-crisis-comment.png)

*图 11.1-15 评估危机观点*

Karpathy 对“评估危机”的担忧可以概括为三点：常见基准会饱和，公开榜单会被过拟合，社区案例观察又容易受到确认偏误和小样本影响。因此，本章后续会逐类拆解 benchmark，并反复检查 difficulty、realism 和 validity。

## 11.2 如何看待评估

> [!NOTE]
> 本节回答一个总框架问题：给定一个评估结果，应该按哪三个维度反复审视它？读完后，读者应能用 difficulty / realism / validity 三维度直接拆解任一 benchmark 的可信度，并理解四类使用场景（用户 / 研究者 / 政策 / 开发者）对评估设计的不同要求。

评估是一个服务明确目标的设计框架。它从要测量的抽象构念出发，再选择足够贴近该构念的输入、调用方式、评分规则和解释方式；输入提示词、得到输出、计算平均分只是执行层面的最后一步。

> [!NOTE]
> 核心原则：没有唯一的“正确”评估。只有和目标、输入分布、调用规则、评分规则绑定在一起的评估。

每个 benchmark 都可以用三个维度反复检查：

- **Difficulty**：任务是否已经饱和，是否能区分当前模型之间的差异。
- **Realism**：输入、工具、交互方式和评分标准是否接近真实使用。
- **Validity**：分数是否真的测到了声称要测的能力，是否排除了数据污染、格式投机、裁判偏见或 scaffold 差异。

这三个维度经常互相牵制。越真实的任务通常越贵、越难复现；越干净的学术基准又可能离真实产品较远。因此评估设计本身就是取舍。

一个常见误区是把 realism 当成 validity 的充分条件。真实任务确实更接近用户场景，但如果评分规则噪声很大、工具 scaffold 差异没有控制、测试集被污染，真实任务也会给出无效信号。反过来，抽象基准虽然不够真实，却可能更适合隔离某种能力边界。

评估的意义取决于你想回答的问题：

- 对于终端用户/企业：我应该采购哪个模型来满足我的具体需求（如客户服务聊天机器人）？
- 对于研究人员：我们是否在模型的原始能力（如“智能”）上取得了科学进步？
- 对于政策制定者/安全机构：当前模型带来的益处和风险分别是什么？现状如何？
- 对于模型开发者：我该如何改进我的模型？哪些干预措施有效？

为了将抽象目标转化为具体评估方案，我们需要思考四个关键环节。

### 11.2.1 输入是什么？

- 提示词从何而来？覆盖了哪些使用场景？
- 我们是否在长尾分布（即那些出现频率很低、不常见但真实存在的输入情况，如在语言模型中，可能包括罕见领域的问题、复杂推理任务、边缘语言现象、对抗性或歧义性较强的语句等）中包含了具有代表性的困难输入样本？
- 输入是否需要适配模型？例如，在多轮对话中，用户的下一条输入依赖于模型的上一条回复，这使得评估变得动态化。

### 11.2.2 如何调用模型？

- 如何提示模型？是零样本、少样本还是思维链（CoT）？
- 是否允许模型使用外部工具（如代码解释器、RAG 检索增强生成）？
- 我们是在评估语言模型本身，还是评估一个由模型和代理框架（agent scaffolding）组成的完整系统？开发者关心前者，而用户关心后者。

### 11.2.3 如何评估输出？

- 参考答案是否准确无误？
- 使用什么指标？对于代码生成，是 pass@1（模型生成一个回答，输出回答和标准答案一样即为正确）还是 pass@10（模型一次生成十个回答，只要十个输出回答里有一个和标准答案一样即为正确）？
- 如何考量成本？模型性能不应只看准确率，还要考虑计算成本。例如：一个更大但仅略优的模型，是否值得其高推理延迟或训练开销？在资源受限场景（如移动端、低延迟服务），成本是关键约束。
- 如何处理不对称错误？某些错误的代价远高于其他错误。在医疗、金融、法律等高风险领域中，幻觉（hallucination）的代价远高于其他错误。
- 如何评估开放域生成任务？当没有标准答案时（如“写一个关于斯坦福的精彩故事”），评估变得极其困难。常见做法：
  - 使用人工评估（成本高但可靠）；
  - 用 LLM-as-a-judge（让另一个大模型打分）；
  - 基于多样性、流畅性、相关性等维度设计 Proxy metrics（代理指标，是指在无法直接测量目标量（target quantity）时，用来间接反映或近似该目标的可测量指标）；
  - 通过用户反馈或 A/B 测试衡量实际效用。

### 11.2.4 如何解读结果？

- 一个 91% 的分数意味着什么？它是否足够好，可以部署给真实用户？
- 如何判断模型掌握的是可迁移能力，还是训练数据记忆？
- 最重要的是，我们到底在评估什么？是最终的产品模型、一个完整的系统，还是底层的研究方法？

忽略这些问题，仅凭一个分数做判断，是评估中最大的误区。

## 11.3 困惑度

> [!NOTE]
> 本节介绍评估语言模型最基础的指标 perplexity：它如何定义、为什么需要在测试集上测量、经典数据集的历史作用、为什么今天仍然是预训练阶段的连续监控信号，以及何时不该用 perplexity。读完后，读者应能在看到任何 perplexity 数字时，准确说出它的口径、置信区间和局限。

### 11.3.1 什么是 Perplexity（困惑度）？

在深入具体的下游任务基准之前，我们必须理解一个基础且重要的度量标准：困惑度（Perplexity）。

语言模型定义了一个序列概率分布 $p(x)$，它对任意一段 token 序列 $x$ 给出概率值，也就是这个序列在模型分布下有多自然。困惑度衡量模型对某个数据集分配高概率的能力。在预训练阶段，模型的目标就是最小化训练集上的困惑度。数值越小，表示模型越容易预测这些 token。训练侧优化交叉熵的等价表述与具体算式见 [第 2 章 §2.4 算力与算子](../chapter2/chapter2_pytorch与资源核算.md)。

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/|D|}
$$

其中 $p(D)$ 是模型给整个数据集 $D$ 中所有句子联合概率的乘积（通常取 log 求和再指数还原）；$|D|$ 是数据集中 token 的总数量。

### 11.3.2 为什么要用测试集？

在预训练阶段，我们通过最小化训练集上的困惑度来优化模型参数。但为了评估模型是否真的“学会了语言”，我们必须在 **未见过的数据（测试集）** 上测量困惑度：在未见过的测试集上测量困惑度，得到的是模型对未参与训练的数据的拟合能力，也称作泛化能力。

> [!WARNING]
> 不能只看训练集困惑度，否则容易过拟合。

### 11.3.3 经典基准数据集（Standard datasets）

这些是历史上用于评估语言模型的经典数据集：

| 数据集 | 特点 |
|--------|------|
| **Penn Treebank (PTB)** | 小规模，华尔街日报语料，常用于早期 RNN/LSTM 实验 |
| **WikiText-103** | 基于维基百科的大规模英文语料，更贴近真实语言分布 |
| **One Billion Word Benchmark (1BW)** | 来自机器翻译数据集（欧洲议会、联合国文件、新闻），词汇量大，挑战性强 |

> [!NOTE]
> 这些数据集曾是衡量语言模型性能的“黄金标准”，但今天更适合被理解为历史基线：它们帮助我们看清困惑度如何演化，并不代表所有现代 LLM 能力。

### 11.3.4 历史与演变

2016 年，Jozefowicz 等人在[《Exploring the Limits of Language Modeling》](https://arxiv.org/abs/1602.02410) 论文里，用纯 CNN + LSTM 架构，在 十亿（1B） Word Benchmark 上把困惑度从 51.3 降到 30.0 —— 当时是非常大的突破。

![图 11.3-1 CNN+LSTM 降低困惑度](images/11-3-1-cnn-lstm-perplexity.png)

*图 11.3-1 CNN+LSTM 降低困惑度*

OpenAI 在 [《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/) 论文中发布的 GPT-2，在 WebText（来自 Reddit 链接的网页文本，约 40GB）上训练，然后直接在标准数据集上做 zero-shot（零样本）评估。

这属于“分布外评估”（out-of-distribution），因为训练数据和测试数据来源不同。但它表现很好，说明大规模、多样化的训练数据能带来强大的泛化能力。

![图 11.3-2 GPT-2 zero-shot 评估](images/11-3-2-gpt2-zero-shot-evaluation.png)

*图 11.3-2 GPT-2 zero-shot 评估*

这是 GPT-2 论文中的关键表格，展示了不同规模模型在多个任务上的表现。可以读出三点：随着模型变大，困惑度持续下降；在 LAMBADA 等任务上，语言建模能力会迁移到下游准确率；但在 1BW 这类更大或更难的数据集上，增大模型带来的改进并不总是同样明显。

### 11.3.5 为什么困惑度仍重要？

自 GPT-2 和 GPT-3 以来，语言建模论文越来越重视下游任务准确率，但困惑度仍然是模型开发中的基础指标：

- 困惑度变化连续，适合拟合 scaling law；下游准确率往往更 noisy，也更容易接近天花板。
- 困惑度可以在任意文本上计算，不需要为每个任务重新构造标注集，因此适合作为预训练阶段的通用监控信号。
- 困惑度可以条件化：固定 prompt，只计算 response 的概率，用来更聚焦地衡量某类回答或候选答案。

因此，困惑度不仅是训练目标，也是分析模型内在能力的重要工具。

### 11.3.6 使用困惑度的陷阱

构建 perplexity leaderboard 时，评估方需要拿到每个 token 的概率分布，并确认这些概率形成有效归一化分布。相比直接收集模型回答再打分，perplexity 评估更依赖模型内部接口，因此存在额外信任问题：

- 如果模型偷偷作弊（比如输出虚假概率），你就无法检测。
- 早期模型遇到未知词（UNK）时处理不当，也会导致概率不准确。

> [!WARNING]
> 困惑度更适合研究者自己控制模型架构和训练过程的场景，不适合只返回文本的闭源服务评测。服务端如果不返回可信概率分布，就无法可靠计算 perplexity。

困惑度还有一个重要边界：它会惩罚所有 token 的预测错误，但下游任务通常只关心其中一部分信息。例如一句事实问答里，模型把无关修饰词预测错也会提高 perplexity；而真实用户可能只在意关键实体、数值或操作是否正确。因此困惑度适合作为预训练与 scaling 的连续指标，却不能单独代表问答、agent 或安全场景的可用性。

> [!TIP]
> 一个常见折中是条件困惑度：固定 prompt，只计算 response 的 $p(\text{response}\mid\text{prompt})^{1/|\text{response}|}$。很多选择题或答案打分 benchmark 也可以看作“困惑度的变体”：比较候选答案在给定上下文下谁更自然，而非直接让模型自由生成。

### 11.3.7 困惑度信念的理论版本

一个强版本的语言建模信念是：假设真实语言分布是 $t$，模型分布是 $p$，最优 perplexity 为 $\exp(H(t))$，并且只有 $p=t$ 时达到。如果模型足够接近真实分布，就可以通过条件生成回答问题、写代码或完成其他语言任务。

这个观点解释了为什么长期降低 perplexity 会成为 scaling 研究的重要驱动力。它的边界也很清楚：实际应用通常只关心语言分布中的一部分行为，例如关键事实、工具调用、数学推理、安全拒答和交互可靠性。为了这些目标，评估需要从平均 token 概率走向条件困惑度、任务准确率、agent 成功率和安全指标。

### 11.3.8 从语言建模走向下游任务

困惑度衡量的是模型对语言分布的拟合能力。更具体的下游任务会把这种分布建模能力压缩到某个关键判断上，例如填空、选择后续事件或回答考试题。

#### 完形填空类任务

完形填空类任务就是给定一段上下文（context），让模型预测其中一个被遮盖的词（target word）。这个任务要求模型不仅要理解局部语法，还要捕捉长距离依赖、语义连贯性、世界知识等。评估集中在关键位置的预测质量，不再把整个句子的平均概率作为唯一信号。

因此，完形填空类任务可以看作是“局部困惑度”。

来自 [LAMBADA](https://arxiv.org/abs/1606.06031) 数据集的三个具体例子

![图 11.3-3 LAMBADA 完形填空样例](images/11-3-3-lambada-cloze-examples.png)

*图 11.3-3 LAMBADA 完形填空样例*

案例 1：
> 原文上下文：“Yes, I thought I was going to lose the baby.” … “This baby wasn’t exactly planned for.”

- 目标句： “Do you honestly think that I would want you to have a _____ ?”
- 正确答案：miscarriage（流产）

> 中文翻译与理解：“是的，我当时以为我会流产。”……“这个孩子其实并不是计划中的。”

- 目标句：“你真的觉得我会希望你经历一次 _____ 吗？”
- 正确答案：流产

案例 2：
> 原文上下文：“Why?” … “He was a great craftsman,” said Heather. “That he was,” said Flannery.

- 目标句： “And Polish, to boot,” said ______.
- 正确答案：Gabriel

> 中文翻译与理解：“为什么？”……“他是个了不起的工匠，”海瑟说。“确实如此，”弗兰纳里说。

- 目标句：“而且，他还是波兰人呢，”______ 说道。
- 正确答案：加布里埃尔（Gabriel）

案例 3：

> 原文上下文：Preston had been the last person to wear those chains...

- 目标句： Sergei looked at me, surprised by my low, raspy please, but he put down the _____.
- 正确答案：chains（锁链）

> 中文翻译与理解：普雷斯顿是最后一个戴上那些锁链的人……

- 目标句：谢尔盖看着我，对我那低沉沙哑的恳求感到惊讶，但他还是放下了那条 _____。
- 正确答案：锁链

#### 常识推理任务

[HellaSwag](https://arxiv.org/pdf/1905.07830)是一个常识推理任务。通常是给定一个视频片段或一段文字描述（premise），然后给出四个选项（A, B, C, D），要求选出最符合常识、最自然、最可能发生的后续动作。它强调的是日常生活的常识推理（commonsense reasoning）和行为合理性判断。

![图 11.3-4 HellaSwag 常识推理样例](images/11-3-4-hellaswag-examples.png)

*图 11.3-4 HellaSwag 常识推理样例*

案例 1：给狗洗澡

> 一位女士正站在室外，手里拿着一只水桶，身边还有一只狗。那只狗四处奔跑，试图躲过洗澡。她……
>
> A. 用肥皂冲洗水桶，然后用吹风机吹干狗的头。
> B. 用水管防止它被弄湿。
> C. 把狗弄湿了，结果它又跑掉了。 ← 正确
> D. 和狗一起钻进浴缸里。

解析：现实中，狗通常讨厌洗澡，主人一泼水它就逃跑；所以“先弄湿 → 狗逃跑”是最常见、最自然的流程；A、B、D 要么违反常识（狗不可能乖乖吹头），要么操作不合理（用水管“防止”弄湿？）

案例 2：交通规则

> 在停车标志或红灯处必须完全停下……如果你在红灯前停下，请在灯变绿后继续行驶……
>
> A. 停车时间不超过两秒……
> B. 完全停下后，关闭你的转向灯……
> C. 远离对向车流……
> D. 如果你所在车道有白色停止线，请在这条线前停下。等所有车辆通过后再穿过路口。 ← 正确

解析：D 描述的是标准交通规则：停在白线前，确认安全再通行；A 错误（红灯必须停够时间）；B 无关（转向灯在此场景不适用）；C 模糊（“远离对向车流”不是红灯场景的核心操作）。

HellaSwag 可以看作是“情境下的困惑度”，模型不需要输出概率，但它的选择应该反映出它对“哪个结局最可能”的内在概率估计。

## 11.4 知识类基准

> [!NOTE]
> 本节覆盖四类知识型考试基准：MMLU / MMLU-Pro / GPQA / Humanity's Last Exam。读完后的能力：能区分「知识广度基准」与「高难度专家基准」的设计取舍，能读懂题目来源、难度档位和 contamination 防护。

这类基准旨在衡量模型所掌握的事实性知识。

### 11.4.1 MMLU (Massive Multitask Language Understanding)

[MMLU](https://arxiv.org/pdf/2009.03300.pdf) 包含 57 个学科（从数学、历史到法律、伦理）的多项选择题。问题源自网络，由学生收集。它更侧重于知识而非语言理解。

最初用 GPT-3 的少样本提示进行评估时，最大型号的 X-Large（175B）只能取得 43.9% 的平均准确率，而 Small（2.7B）、Medium（6.7B）、Large（13B）三档分别是 25.9%、24.9%、26.0%，基本停在 25% 的随机基线上（[Hendrycks et al., 2020, arXiv:2009.03300](https://arxiv.org/pdf/2009.03300.pdf) Table 1）。几年之后，MMLU 已经接近饱和，这恰恰说明单一知识基准很容易从“有区分度”变成“只剩刷榜价值”。

![图 11.4-1 GPT-3 在 MMLU 上的 few-shot 提示](images/11-4-1-gpt3-mmlu-few-shot.png)

*图 11.4-1 GPT-3 在 MMLU 上的 few-shot 提示*

[HELM 的 MMLU-Pro 视图](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)可以作为动态榜单入口；放在本章里，更重要的是看 MMLU-Pro 如何通过清理噪声题、增加选项和使用 CoT 评估来恢复区分度。

### 11.4.2 MMLU-Pro

[MMLU-Pro](https://arxiv.org/abs/2406.01574) 是对 MMLU 的改进版，移除了噪声大或过于简单的问题，并将选项从 4 个增加到 10 个。模型准确率显著下降，缓解了饱和问题。通常结合思维链（CoT）进行评估。MMLU-Pro 共 **12,032 题**，覆盖 14 个学科，来源组成如下表。

| 来源 | 题数 | 占比 |
|------|------|------|
| MMLU 过滤子集 | 6,810 | 56.60% |
| STEM 网站 | 4,083 | 33.93% |
| TheoremQA | 598 | 4.97% |
| SciBench | 541 | 4.50% |

![图 11.4-2 MMLU-Pro 与 MMLU 对比](images/11-4-2-mmlu-pro-comparison.png)

*图 11.4-2 MMLU-Pro 与 MMLU 对比*

把选项从 4 增到 10 并加入 CoT 评估后，模型在 MMLU-Pro 上的得分相对 MMLU 下降 16%–33%：领先模型 GPT-4o 在 MMLU-Pro 上只有 72.6%，而它在 MMLU 上是 88.7%（CoT）/ 87.2%（direct）。同时，GPT-4o、Claude-3-Opus、GPT-4-Turbo 这几个头部模型之间的分差从 MMLU 上的约 2% 拉开到 MMLU-Pro 上的约 9%，prompt 变体带来的分数波动也从 MMLU 的 4%–5% 收窄到 2%，区分度和稳定性同时改善。

### 11.4.3 GPQA (Graduate-Level Google-Proof Q&A)

[GPQA](https://arxiv.org/abs/2311.12022) Main 共 **448 题**（Extended 546 / Diamond 198 子集），由 61 名 PhD 领域专家通过 Upwork 平台设计（论文 §2.1 "The Collection Pipeline" 原文："We hire 61 contractors through Upwork to write and validate the dataset. We require that they have completed or are currently in a PhD program in their field of expertise"）。目标是创建"防谷歌"问题，即非专家即使花 30 分钟用谷歌搜索也难以解答。论文用三档**实测准确率**验证题目确实够难：

- 博士级专家在对应领域达到 **65%** accuracy，剔除专家事后自认的明显失误后为 74%（§1、§3.1）
- 非专家在可访问互联网的条件下达到 **34.1% ± 2.3%** accuracy，中位耗时 30 分钟（§3.2）
- GPT-4 few-shot CoT 在 Extended / Main / Diamond 三档分别是 38.7% / 39.7% / 38.8%，即摘要里概括的 **39%**（§4 Table 5）

论文 §2.3 "Dataset Splits" 定义的是子集划分规则：Main 剔除"两名专家都答错且三名非专家全部答对"的题，Diamond 进一步只保留"两名专家都答对且多数非专家答错"的题。划分规则与上面三档实测准确率是两件事，前者决定哪些题入选，后者只是入选后的难度验证。

![图 11.4-3 GPQA 高难问答基准](images/11-4-3-gpqa-benchmark.png)

*图 11.4-3 GPQA 高难问答基准*

[HELM 的 GPQA 视图](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)展示了不同模型在这一基准上的变化。读 GPQA 时，更重要的是它对高难科学问答仍有较强区分度，同时样本量和学科覆盖有限。

### 11.4.4 Humanity's Last Exam

[Humanity's Last Exam](https://arxiv.org/abs/2501.14249)（HLE）是一个由社区贡献的多模态、多学科基准，包含 2,500 道选择题与精确匹配题，覆盖数学、人文与自然科学等数十个学科。约 14% 的题目需要同时理解文本和图像，约 24% 是选择题，其余 76% 是精确匹配题。奖金池 50 万美元按贡献题目的难度分档：最难的 50 题各 5,000 美元，再往后 500 题各 500 美元。题目由社区贡献，先用前沿 LLM 筛选掉过于简单的题，再经过多阶段专家审查。HLE 的局限在于问题征集过程可能存在严重的选择偏差，且问题类型仍局限于有标准答案的”考试”形式。

![图 11.4-4 Humanity's Last Exam 收集筛选流程](images/11-4-4-hle-pipeline.png)

*图 11.4-4 Humanity's Last Exam 收集筛选流程*

[HLE 官方站点](https://agi.safe.ai/)提供持续更新的结果。HLE 仍然远未饱和，跟踪其时间序列可观察模型在多学科高难题上的增量进展。HLE 使用私有 held-out 测试集 + 公开开发集，避免训练流直接覆盖评测。

## 11.5 指令遵循基准

> [!NOTE]
> 本节覆盖聊天偏好、显式约束、长度归一化与真实用户分布四类指令遵循评估：Chatbot Arena / IFEval / AlpacaEval / WildBench，以及 LLM-as-judge 的四类偏差。读完后的能力：能在看到任意聊天基准分数时，定位它的评分机制（盲测 / 规则 / LLM judge），并指出对应的偏差来源与缓解方式。

这类基准评估模型是否“听话”，能否按照用户的要求进行输出。

### 11.5.1 Chatbot Arena

[Chatbot Arena](https://arxiv.org/abs/2403.04132)（现改名 LMArena）采用”盲测”和 ELO 评分系统。真实用户提交提示，同时收到两个匿名模型的回复，并选择更优者。优点是输入动态、能容纳新模型。问题在于评估者是网站访客，样本可能存在偏差；ELO 分数可能被策略性操纵。

![图 11.5-1 Chatbot Arena 分数排行榜](images/11-5-1-chatbot-arena-leaderboard.png)

*图 11.5-1 Chatbot Arena 分数排行榜*

[Arena 实时排行榜](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)适合观察人类偏好信号如何随模型和用户分布变化。Arena Score 试图用盲测人类偏好逼近真实使用体验，但它同时也更容易受到平台流量分布、提示词挖掘和策略性优化的影响——使用 Arena 数据作为评估依据时，应同时记录评估时间、流量来源与提示词模板。

### 11.5.2 IFEval (Instruction-Following Eval)

[IFEval](https://arxiv.org/abs/2311.07911) 使用可自动验证的约束（如“回答必须包含至少5句话”）来测试模型。优点是自动化程度高。局限是只评估约束遵守情况，不评估语义质量，且约束本身可能过于人工化。

![图 11.5-2 IFEval 指令约束样例](images/11-5-2-ifeval-instruction-details.png)

*图 11.5-2 IFEval 指令约束样例*

图 11.5-3 展示的是 HELM 上 IFEval 的排名视角。它衡量的是“模型是否遵守显式指令约束”，不等同于语义质量、创造力或完整助手能力。

![图 11.5-3 HELM IFEval 排行榜](images/11-5-3-helm-ifeval-leaderboard.png)

*图 11.5-3 HELM IFEval 排行榜*

### 11.5.3 AlpacaEval

[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) 包括各种来源的 805 条指令。使用一个强大的 LLM（如 GPT-4）作为裁判，判断候选模型的回答是否优于 GPT-4 自身的回答，并计算胜率。优点是自动化程度高，能处理开放域回答。问题在于存在裁判偏见，且早期版本易被回答长度等表面特征欺骗。

![图 11.5-4 AlpacaEval 排行榜](images/11-5-4-alpacaeval-leaderboard.png)

*图 11.5-4 AlpacaEval 排行榜*

AlpacaEval 2.0 的一个重要变化，是用回归方式修正长度偏置，避免模型通过生成更长、更啰嗦的回答来获得更高胜率。这说明开放式回答评估不能只依赖一个裁判分数，还要检查裁判是否被表面特征欺骗。

### 11.5.4 WildBench

[WildBench](https://arxiv.org/pdf/2406.04770) 从约 100 万条真实人机对话中先随机采样 1,500 条，再筛出 1,024 条构成评估集（论文 §2.1）。主评估以 GPT-4-Turbo 为裁判，输出 WB-Reward 与 WB-Score 两类指标（§3.2、§3.3）；检查清单由 GPT-4-Turbo 与 Claude-3-Opus 联合生成，用来降低单个 LLM 裁判自身的偏差（§3.1）。

论文 §4.3 ablation 还测试了 GPT-4、Claude 3 Opus 与 Mistral-Large 等替代裁判，结果显示它们给出的相对排名基本一致。WildBench 与 Chatbot Arena 高度相关，论文 §4.2 Table 3 报告的 Pearson 相关系数为：WB-Reward 对 top-ranking 模型 0.98、WB-Score 0.95，均高于 ArenaHard 的 0.91 和 AlpacaEval 2.0 length-controlled win rate 的 0.89。这些数值随评测设置而异，但总体说明 WildBench 已成为新基准有效性的“事实上的”检验标准之一。

![图 11.5-5 WildBench 构建流程](images/11-5-5-wildbench-pipeline.png)

*图 11.5-5 WildBench 构建流程*

[HELM 的 WildBench 视图](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)可以用来观察不同模型在真实对话样本上的表现。它的价值在于引入更接近真实用户提问分布的样本；某个固定时点的排行榜名次只是快照。

![图 11.5-6 HELM WildBench 排行榜](images/11-5-6-helm-wildbench-leaderboard.png)

*图 11.5-6 HELM WildBench 排行榜*

对于人工评估和 LLM-as-a-judge，rubric 或 checklist 的价值在于把“这个回答好不好”拆成可复核的判断项。它不能消除偏见，但能降低不同裁判之间的漂移，让错误更容易被定位。

### 11.5.5 LLM-as-judge 的偏差与缓解

LLM-as-judge 把评估成本压低到可大规模运行的级别，但也把 judge 模型自身的偏差带进了分数。主要偏差来源有四类：

- **长度偏差（length bias）**：judge 模型倾向给更长回答更高分，无论内容质量是否真的更高；这是 AlpacaEval、AlpacaEval 2.0 等基于 LLM-as-judge 的指标最被反复讨论的问题 ([Zheng et al., 2023, arXiv:2306.05685](https://arxiv.org/abs/2306.05685))。缓解办法包括按字符 / token / 段落长度归一化分数、报告 length-controlled win rate，或在 prompt 中显式要求 judge 忽略长度。
- **位置偏差（position bias）**：当 judge 同时看到两个回答时，倾向给第一个出现的回答更高分；多轮交换位置后取平均可以分离这一效应。
- **自我偏好（self-preference bias）**：judge 模型给同家族模型更高分；常见缓解是引入多 judge 集成或与人类标注的校准。
- **风格与格式偏差（style / format bias）**：judge 可能被 markdown 标题、bullet 列表、emoji 等表面格式影响；控制 rubric 中显式排除这些信号。

工程做法通常同时叠加：多 judge 投票（pairwise 偏好下用 majority vote）、length-controlled win rate、judge ensemble 与 human spot check。JudgeBench ([arXiv:2410.12784](https://arxiv.org/abs/2410.12784)) 等基准则直接评估 judge 模型本身的判别能力，而不是被评模型的能力。

LLM-as-judge 与 RLHF / RLVR 的连接在第 12 章偏好优化和第 13 章 verifier-as-reward 中再次出现：reward model 的偏差直接决定偏好优化的目标偏差，因此 judge 与 reward 的偏差清单需要一起维护。

## 11.6 智能体基准

> [!NOTE]
> 本节介绍四类 agent 基准：SWE-bench / Terminal-Bench / Cybench / MLE-bench，并明确一个关键边界：这些基准评估的是「模型 + agent scaffold」的系统能力，而非裸模型。读完后的能力：能在看到任一 agent benchmark 分数时，识别它对应的 scaffold 假设、工具权限与运行预算。

这类基准评估模型作为智能体（Agent） 的能力，即在复杂环境中通过工具调用和迭代规划完成任务。

需要特别区分：智能体基准通常评估的是**语言模型 + agent scaffold** 的系统能力，区别于纯语言模型本身。规划循环、工具调用策略、文件读写、上下文压缩、子任务分解和失败重试都会改变结果，因此同一个底座模型在不同 scaffold 下可能表现差异很大。同一个 benchmark 既是评估工具也是 RL 训练数据：SWE-bench 风格任务在 [第 13 章 RLVR](../chapter13/chapter13_可验证奖励的强化学习.md) 的 agentic RL 训练里被大量构造为可验证 rollout。

> [!NOTE]
> **Agent scaffold 的四个核心组件**：**explicit planning**（显式写出多步计划并勾选进度）、**hierarchical delegation**（任务分层委派，子任务可由更小的 agent loop 完成以保持上下文干净）、**persistent memory**（通过读写文件维护跨 turn 状态，区别于纯上下文窗口）、**extreme context engineering**（在 prompt 中显式给出大量过程性指令，区别于单纯的上下文压缩 / 重组 / 检索）。这四项与 2025-2026 主流 agent 框架（Claude Code、Cursor、Aider 等）的设计选择基本对齐；同一底座模型在不同 scaffold 组合下的 benchmark 分数可能差几倍。

> [!NOTE]
> **基准饱和与坐标移动**：早期 GPT-3 X-Large 在 MMLU 上只有 43.9%，但刷到接近饱和后被 MMLU-Pro 替代（具体口径见 §11.4.2）。GPQA 论文里 GPT-4 few-shot CoT 在 Diamond 子集上是 38.8%，而当前榜单上 GPQA Diamond 与 SWE-bench Verified 已被多个前沿模型刷到 90% 上下。基准饱和与坐标快速移动说明同一模型在不同时间窗的分数几乎不能直接横比，评估时需要同时记录版本、日期和 prompt 模板。

### 11.6.1 SWE-bench

[SWE-bench](https://arxiv.org/abs/2310.06770) 包含 12 个 Python 代码库中的 2294 个任务，根据 GitHub issue 描述，提交一个能通过单元测试的 Pull Request (PR)。评估直接运行单元测试来验证修复是否成功。

![图 11.6-1 SWE-bench 评测流程](images/11-6-1-swebench-workflow.png)

*图 11.6-1 SWE-bench 评测流程*

### 11.6.2 Terminal-Bench

[Terminal-Bench](https://www.tbench.ai/) 将任务放在通用终端环境中，要求模型通过 shell、文件系统和命令行工具完成开放式工作流。它比单步问答更接近真实工程任务，也更能暴露 agent scaffold 在规划、执行和恢复错误上的影响。

Terminal-Bench 由 93 位贡献者提交 229 个任务，经筛选后其中 89 个任务构成 Terminal-Bench 2.0 数据集（[arXiv:2601.11868](https://arxiv.org/abs/2601.11868) 与官方 https://www.tbench.ai/，后续版本可能扩展）。论文署名作者为 Mike A. Merrill 等 85 位。Terminal-Bench 不使用 first-solve time 作为难度指标：先解时间用来度量任务难度是 Cybench 的做法（见 §11.6.3），Terminal-Bench 通过环境多样性和人工解法时长分布来体现任务量级。

![图 11.6-2 Terminal-Bench 任务示例](images/11-6-2-terminal-bench-task-example.png)

*图 11.6-2 Terminal-Bench 任务示例*

![图 11.6-3 Terminal-Bench 人类耗时分布](images/11-6-3-terminal-bench-human-time.png)

*图 11.6-3 Terminal-Bench 人类耗时分布*

![图 11.6-4 Terminal-Bench 结果](images/11-6-4-terminal-bench-results.png)

*图 11.6-4 Terminal-Bench 结果*

这类结果衡量的是模型在给定 scaffold、工具权限和运行环境下完成真实终端任务的端到端能力。换一套规划循环、工具封装或错误恢复策略，同一个底座模型的分数也可能变化。

### 11.6.3 Cybench

[Cybench](https://arxiv.org/abs/2408.08926) 完成 40 个网络安全领域的"夺旗"（CTF）挑战。任务难度通过人类"首次解决时间"（First-Solve Time, FST）来度量；最难的题目由人类团队首解耗时约 24 小时 54 分钟，整体范围从约 2 分钟到 24 小时 54 分钟不等（论文 Figure 3 / Section 5）。

![图 11.6-5 Cybench 评测流程](images/11-6-5-cybench-workflow.png)

*图 11.6-5 Cybench 评测流程*

![图 11.6-6 Cybench 模型结果](images/11-6-6-cybench-results.png)

*图 11.6-6 Cybench 模型结果*

### 11.6.4 MLE-bench

[MLE-bench](https://arxiv.org/abs/2410.07095) 自动化参与 75 个 Kaggle 机器学习竞赛，包括数据处理、模型训练、超参调优和结果提交。在论文给定的设置下，当前最佳模型获得任何 Kaggle 奖牌（bronze / silver / gold）的比例为约 **16.9%**。

![图 11.6-7 MLE-bench 评测流程](images/11-6-7-mlebench-workflow.png)

*图 11.6-7 MLE-bench 评测流程*

![图 11.6-8 MLE-bench 模型结果](images/11-6-8-mlebench-results.png)

*图 11.6-8 MLE-bench 模型结果*

这些基准代表了 LLM 能力的前沿任务形态，但当前模型的成功率普遍较低，说明长程自主规划、环境交互和结果验证仍然是主要短板。

## 11.7 纯推理基准

> [!NOTE]
> 本节聚焦 ARC-AGI 系列：1 / 2 / 3 三代的任务形态与 o1/o3 test-time compute 范式对纯推理任务的改变。读完后的能力：能解释为什么 reasoning model 的出现把 ARC-AGI 这类「剥离语言的归纳推理」基准从「无法解」推到「接近饱和」，以及 ARC-AGI-3 引入交互环境后带来的新维度。

这类基准试图剥离知识，仅评估模型的抽象推理能力。

### 11.7.1 ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)

[ARC-AGI](https://arcprize.org/arc-agi) 提供一系列视觉化的输入-输出网格对，要求模型推断出变换规则并应用于新的输入。整个过程不涉及任何语言。

ARC-AGI-1:

![图 11.7-1 ARC-AGI-1 任务示例](images/11-7-1-arc-agi-1-task.png)

*图 11.7-1 ARC-AGI-1 任务示例*

![图 11.7-2 ARC-AGI-1 模型得分](images/11-7-2-arc-agi-1-results.png)

*图 11.7-2 ARC-AGI-1 模型得分*

更难的 ARC-AGI-2:

![图 11.7-3 ARC-AGI-2 任务示例](images/11-7-3-arc-agi-2-task.png)

*图 11.7-3 ARC-AGI-2 任务示例*

它捕捉了一种更纯粹的、类似人类的模式识别和泛化能力，是早期 AGI 研究的重要基准。传统 LLM 在此任务上表现极差，而新一代模型的进展说明交互、搜索和测试时适应正在改变这类任务的解法空间。

ARC-AGI-3 在 2026 年 3 月发布，把任务从一次性网格预测切换到交互环境：模型在环境中尝试操作、观察反馈并调整策略，覆盖了 ARC-AGI-1/2 未涉及的"规则未知环境下的探索与归纳"。这是 ARC-AGI 系列首次引入交互环境的新成员。

> [!NOTE]
> **ARC-AGI-1 自 o1/o3 后已基本解决**：ARC-AGI-1 早期传统 LLM 几乎无法通过，但在 OpenAI o1、o3 引入 test-time compute / search 后分数迅速上升至接近饱和，ARC-AGI-2 也正在被快速解决。这是 reasoning model 范式（CoT + search + test-time adaptation）改变纯推理任务解法空间的关键转折点，常作为 reasoning 模型价值的代表性证据。

![图 11.7-4 ARC-AGI-3 交互环境](images/11-7-4-arc-agi-3-environment.png)

*图 11.7-4 ARC-AGI-3 交互环境*

![图 11.7-5 ARC-AGI-3 结果](images/11-7-5-arc-agi-3-results.png)

*图 11.7-5 ARC-AGI-3 结果*

## 11.8 安全基准

> [!NOTE]
> 本节覆盖四类安全评估视角：HarmBench（行为拒答率）/ AIR-Bench（监管与政策对齐）/ Jailbreaking（绕过对齐的攻击）/ 部署前测试（监管机构预审）。读完后的能力：能区分「能力」（capability）与「倾向」（propensity）两类风险评估口径，以及 dual-use 场景下安全边界对评估设计的约束。

对于 AI 来说，安全意味着什么？

![图 11.8-1 AI safety 的类比问题](images/11-8-1-ai-safety-analogy.png)

*图 11.8-1 AI safety 的类比问题*

安全评估旨在衡量模型执行有害行为的倾向。在 HELM 的 [Leaderboard Safety Scenarios](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard) 展示了当前 LLM 在安全领域的排行榜。

![图 11.8-2 HELM safety 排行榜](images/11-8-2-helm-safety-leaderboard.png)

*图 11.8-2 HELM safety 排行榜*

### 11.8.1 HarmBench

[HarmBench](https://arxiv.org/abs/2402.04249) 定义了 510 种违反法律或社会规范的有害行为，通过提示词测试模型是否会执行，并评估其拒绝率。

在 HELM 的 [HarmBench leaderboard](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench) 可以看到不同模型在该基准上的排行。

![图 11.8-3 HarmBench 排行榜](images/11-8-3-harmbench-leaderboard.png)

*图 11.8-3 HarmBench 排行榜*

### 11.8.2 AIR-Bench

[AIR-Bench](https://arxiv.org/abs/2407.17436) 基于全球监管框架和公司政策，构建了一个包含 314 个风险类别和 5694 个提示的系统性评测集。

![图 11.8-4 AIR-Bench 风险类别概览](images/11-8-4-air-bench-overview.png)

*图 11.8-4 AIR-Bench 风险类别概览*

在 HELM 的 [Leaderboard AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard) 展示了当前 LLM 在 AIR-Bench 基准上的排行。

![图 11.8-5 AIR-Bench 排行榜](images/11-8-5-air-bench-leaderboard.png)

*图 11.8-5 AIR-Bench 排行榜*

### 11.8.3 越狱（Jailbreaking）

即使模型被训练为拒绝有害请求，[攻击者也能通过自动优化提示（如 Greedy Coordinate Gradient, GCG）来绕过安全防护](https://arxiv.org/pdf/2307.15043)。这类攻击甚至能从开源模型迁移到闭源模型。

![图 11.8-6 模型越狱案例](images/11-8-6-jailbreak-example.png)

*图 11.8-6 模型越狱案例*

### 11.8.4 部署前测试

美国安全研究所和英国人工智能安全研究所携手合作，公司在发布前向安全机构提供模型访问权限（目前为自愿性质），安全机构进行评估并向公司提交[报告](https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf)。

### 11.8.5 但安全究竟是什么？

安全不仅关乎“拒绝”，也关乎“能力”。在医疗等高风险场景，减少幻觉本身就是提升安全性和能力。对于闭源 API 模型，倾向性（propensity） 是关键；而对于开源基础模型，能力（capability） 本身就是风险。

安全还具有强烈的上下文依赖：政治、法律、社会规范、行业政策和国家地区都会改变“可接受回答”的边界。网络安全智能体尤其具有 dual-use 特征，同一套能力既可以用于渗透测试，也可以用于攻击系统，因此评估时必须同时记录任务意图、权限边界和部署语境。

## 11.9 真实性

> [!NOTE]
> 本节回答一个核心问题：现有考试型 benchmark 与真实使用场景的距离有多大？通过 GDPval / Clio / MedHELM 三个案例，展示「真实任务 + 真实分布」评估的工程成本与隐私边界。读完后的能力：能在「考试型」与「职业型」评估之间做出选择权衡。

语言模型在实践中被广泛应用：

![图 11.9-1 OpenAI 模型使用量案例](images/11-9-1-openai-usage.png)

*图 11.9-1 OpenAI 模型使用量案例*

![图 11.9-2 Cursor 编程工具使用案例](images/11-9-2-cursor-usage.png)

*图 11.9-2 Cursor 编程工具使用案例*

然而，大多数现有的基准测试（如 MMLU）是标准化的“考试”，与现实世界的使用场景相去甚远。真实用户的提示更多是“求助型”（asking）而非“测验型”（quizzing）。

### 11.9.1 GDPval

[GDPval](https://arxiv.org/pdf/2510.04374) 试图把评估拉回真实职业场景：任务来自美国 GDP 贡献较高行业中的职业工作，目标是衡量模型在专业工作流中的实际可用性。这类高真实度评估往往需要领域专家、私有任务和更复杂的评分标准，成本也会显著提高。

GDPval 覆盖美国 GDP 前 9 个行业中的 44 个职业。这个细节很重要：它试图用经济活动权重来定义真实使用分布，同时也天然面临隐私、版权和专家评分成本的约束。

![图 11.9-3 GDPval 任务示例](images/11-9-3-gdpval-task-example.png)

*图 11.9-3 GDPval 任务示例*

### 11.9.2 Clio (Anthropic)

在 [Clio: Privacy-Preserving Insights into Real-World AI Use](https://arxiv.org/abs/2412.13678) 论文里，提出使用语言模型分析真实用户数据，分享人们提问的一般模式：

![图 11.9-4 Clio 真实用户数据分析](images/11-9-4-clio-use-patterns.png)

*图 11.9-4 Clio 真实用户数据分析*

### 11.9.3 MedHELM

为解决此问题，出现了如 [MedHELM](https://arxiv.org/abs/2505.23802) 这样的基准，它由 29 名临床医生贡献了 121 个真实的临床任务，更贴近实际医疗应用场景。然而，真实性与隐私之间存在天然矛盾，许多真实数据（如患者病历）无法公开。

![图 11.9-5 MedHELM 构建流程](images/11-9-5-medhelm-pipeline.png)

*图 11.9-5 MedHELM 构建流程*

## 11.10 有效性

> [!NOTE]
> 本节覆盖两类「让分数失真」的源头：训练-测试重叠（contamination，§11.10.1 的四条路线）与数据集质量（题目噪声、scaffold 漏洞、§11.10.2）。读完后的能力：能在面对任一 benchmark 分数时，系统检查「题是否干净」「模型是否可能见过题」「scoring 是否真的奖励了目标行为」三步。

评估的有效性面临两大核心挑战。数据污染的源头在训练侧——具体去重与过滤方案在 [第 10 章 §10.2.2](../chapter10/chapter10_数据工程.md) 详细展开，本节只关注"如何在评估时检测并控制 contamination"。

### 11.10.1 训练-测试集重叠（Train-Test Overlap）

在预训练数据即整个互联网的时代，确保测试集未被模型“见过”变得极其困难。这会导致评估结果虚高。解决方法包括开发工具推断重叠情况和推动行业规范，要求模型提供商主动报告重叠检测结果。

经典机器学习评估要求测试集不参与训练。ImageNet、SQuAD 这类基础模型时代的数据集通常有明确 train/test 划分；语言模型训练则会混合大规模、多来源语料，大部分机构也不会公开完整数据清单。

#### 路线 1：尝试从模型中推断训练集和测试集的重叠部分

[PROVING TEST SET CONTAMINATION IN BLACK BOX LANGUAGE MODELS](https://arxiv.org/pdf/2310.17623)利用数据点的可交换性尝试从模型中推断训练集和测试集的重叠部分：

![图 11.10-1 训练-测试重叠推断思路](images/11-10-1-contamination-exchangeability.png)

*图 11.10-1 训练-测试重叠推断思路*

#### 路线 2：鼓励报告规范

[Language model developers should report train-test overlap](https://arxiv.org/abs/2410.08385) 提倡模型提供者应报告训练集与测试集的重叠情况。

#### 路线 3：使用新鲜评估集

LiveCodeBench、UncheatableEval 这类评估会持续抓取新网页或新任务，降低模型在预训练阶段见过测试题的概率。但“时间戳新”并不自动等于“无污染”，因为网页内容可能复制旧题，或者评估题在发布后很快进入训练数据流。

#### 路线 4：使用私有评估集

企业内部代码库、未公开文档、个人写作和私有业务数据更不容易被预训练语料覆盖。私有评估更贴近真实业务，也更适合做条件困惑度或端到端任务评估；代价是外部可复现性较差，难以形成公开排行榜。

### 11.10.2 数据集质量

[许多基准存在标注错误或噪声](https://arxiv.org/abs/2502.03461)。例如，[修正后的 SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) 版本显示，原始分数可能因错误而被低估。

![图 11.10-2 基准标注错误和噪声](images/11-10-2-benchmark-label-noise.png)

*图 11.10-2 基准标注错误和噪声*

因此，现代评估不只需要更难的题，也需要更干净的题。Platinum benchmark 的思路是对已有基准做高质量人工复核，减少错误答案、歧义题和噪声标签（["Do Large Language Model Benchmarks Test Reliability?", arXiv:2502.03461](https://arxiv.org/abs/2502.03461)）。

对智能体基准，题目之外还要检查测试用例是否充分、任务是否能被简单脚本绕过、agent trace 是否暴露不合理捷径。一个具体例子是 τ-bench 把空响应记成成功：在设计上本就无法完成的任务（例如改签不可退票）里，一个只返回空响应的 trivial agent 在 airline 子集上拿到 38% 成功率，反而超过基于 GPT-4o 的 agent（[Zhu 等, Establishing Best Practices for Building Rigorous Agentic Benchmarks, arXiv:2507.02825](https://arxiv.org/abs/2507.02825)）。[Docent](https://transluce.org/introducing-docent) 一类方法尝试用 LLM 辅助审查 agent 执行轨迹，帮助发现评估集自身的问题。

有效性检查可以总结成三步：先检查题目是否干净，再检查模型是否可能见过题，最后检查 scoring 是否真的奖励了目标行为。fresh evals、private evals、人工复核、Platinum benchmark 和 agent trace 审查，分别是在这三步上降低不同类型的失真。

## 11.11 我们到底在评估什么？

> [!NOTE]
> 本节回到 evaluation 最底层的边界问题：评估对象是「方法」（method）还是「模型/系统」（model/system）？两种范式对应不同的科学价值与工程价值。读完后的能力：能在引用任何榜单或论文实验结果前，先定位它属于哪一种范式，再判断结论适用范围。

评估对象必须先说清楚：当前比较的是方法还是模型/系统。这个边界就是评估的 rules of the game。

- 过去：在 ImageNet 时代，我们评估的是方法（method），即在固定数据集和训练协议下，新算法的优劣。
- 现在：我们更多评估的是模型/系统（model/system），即”端到端”的最终产品，开发者可以使用任何数据、任何技巧。

这两种范式各有价值：评估方法能促进算法创新；评估系统对下游用户更有意义。但必须清晰界定评估的“游戏规则”，否则比较将失去意义。

也存在介于两者之间的评估，例如 nanoGPT speedrun：固定数据和目标验证损失，比较在有限时间内谁能把训练系统跑得更快。它在明确规则下鼓励算法、系统和工程优化。

![图 11.11-1 nanoGPT speedrun](images/11-11-1-nanogpt-speedrun.png)

*图 11.11-1 nanoGPT speedrun*

## 本章总结与下章衔接

评估设计本身的工程判断可以收成三条规则。第一，**单一分数不足以支撑结论**——perplexity、exam、chat、agent、推理、安全、真实使用这七类评估各回答一个独立问题，任何一份模型评估材料都应先定位它属于哪一类，再判断结论的适用边界。第二，**评估的规则优先于分数本身**——相同的「78%」分数在不同 prompt 范围、采样参数、工具权限、agent scaffold 下含义不同，比较前必须先确认 rules of the game 一致（§11.2 与 §11.11）。第三，**评估与训练相互定义**——LM-as-judge 的偏差清单（§11.5.5）直接决定偏好优化目标（[第 12 章](../chapter12/chapter12_大模型基本训练流程.md) §12.5）和 RLVR 验证信号（[第 13 章](../chapter13/chapter13_可验证奖励的强化学习.md) §13.3）的偏差结构，judge 与 reward 的偏差需要一起维护。

下章进入 [第 12 章 大模型基本训练流程](../chapter12/chapter12_大模型基本训练流程.md)：评估方法定下来后，训练流水线按 pre-training → mid-training → SFT → RLHF/PPO/DPO 组织；其中 RLHF 与 DPO 的偏好数据来源与 judge 偏差控制直接对应本章 §11.5.5 的四类偏差。

## 思考

下列问题分别对应本章学习目标的 5 条；每条先写一个具体场景，再追问「如何用 benchmark 给出可复核的判断」。

- **四类「好模型」定义（对应学习目标 1）**：当前正在采购一个对话产品用于客服。四个候选模型在 Artificial Analysis 能力榜、OpenRouter 使用量榜、LMArena 偏好榜与真实 NPS 上分别排名 A/B/C/D。哪一个排名才是该场景的合适信号？哪些排名在这个场景下根本不该用？
- **perplexity 的边界（对应学习目标 2）**：手上有一份待评估模型的测试集 perplexity 报告，它在 PTB / WikiText-103 / 1BW 上分别给出 18 / 12 / 22。如果下游任务是开放域问答与多轮对话，这两个数字能告诉你什么、不能告诉你什么？
- **exam 基准与 contamination（对应学习目标 3）**：当前训练语料与 MMLU / GPQA / HLE 三档考试存在部分来源重叠。哪一档最容易通过「收集更多训练数据」绕过？哪一档最依赖 §11.10.1 的四条 contamination 控制路线？
- **chat 与 agent 基准差异（对应学习目标 4）**：一个 5 分的 win-rate 差距在 Chatbot Arena 的统计噪声内吗？confidence interval 是多少？同样的差距换到 SWE-bench Verified 上意味着什么？
- **rules of the game（对应学习目标 5）**：safety benchmark（HarmBench / AIR-Bench）与 safety 部署表现的相关性如何？是否需要额外的 red-team？把同一份安全评估放在闭源 API 模型与开源基础模型上，规则有什么不同（参看 §11.8.5 的「倾向性 vs 能力」边界）？

## 参考文献

- [HellaSwag, arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
- [MMLU / Hendrycks 2020, arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- [MMLU-Pro, arXiv:2406.01574](https://arxiv.org/abs/2406.01574)
- [GPQA, arXiv:2311.12022](https://arxiv.org/abs/2311.12022)
- [Humanity's Last Exam, arXiv:2501.14249](https://arxiv.org/abs/2501.14249)
- [LAMBADA, arXiv:1606.06031](https://arxiv.org/abs/1606.06031)
- [Jozefowicz 等, Exploring the Limits of Language Modeling, arXiv:1602.02410](https://arxiv.org/abs/1602.02410)
- [Oren 等, Proving Test Set Contamination in Black Box Language Models, arXiv:2310.17623](https://arxiv.org/abs/2310.17623) — 用 permutation test 给出 black-box LLM 数据污染的可证明下界
- [Schaeffer 等, Language model developers should report train-test overlap, arXiv:2410.08385](https://arxiv.org/abs/2410.08385)
- [Chatbot Arena, arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- [IFEval, arXiv:2311.07911](https://arxiv.org/abs/2311.07911)
- [AlpacaEval 2.0 (length-controlled debiasing), arXiv:2404.04475](https://arxiv.org/abs/2404.04475)
- [WildBench, arXiv:2406.04770](https://arxiv.org/abs/2406.04770)
- [LLM-as-judge (Zheng 2023), arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- [JudgeBench, arXiv:2410.12784](https://arxiv.org/abs/2410.12784)
- [SWE-bench, arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- [Terminal-Bench, arXiv:2601.11868](https://arxiv.org/abs/2601.11868)
- [Zhang 等, Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models, arXiv:2408.08926](https://arxiv.org/abs/2408.08926)
- [MLE-Bench, arXiv:2410.07095](https://arxiv.org/abs/2410.07095)
- [AIR-Bench, arXiv:2407.17436](https://arxiv.org/abs/2407.17436)
- [HarmBench, arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
- [Zou 等, Universal and Transferable Adversarial Attacks (GCG), arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- [Clio, arXiv:2412.13678](https://arxiv.org/abs/2412.13678)
- [MedHELM, arXiv:2505.23802](https://arxiv.org/abs/2505.23802)
- [GDPval, arXiv:2510.04374](https://arxiv.org/abs/2510.04374)
- ["Do Large Language Model Benchmarks Test Reliability?" (Platinum benchmarks), arXiv:2502.03461](https://arxiv.org/abs/2502.03461)
- [Zhu 等, Establishing Best Practices for Building Rigorous Agentic Benchmarks, arXiv:2507.02825](https://arxiv.org/abs/2507.02825)
- [Docent（Transluce）：用 LLM 审查 agent 执行轨迹](https://transluce.org/introducing-docent)
- [nanoGPT speedrun (Modded NanoGPT)](https://github.com/KellerJordan/ModdedNanoGPT)
- [OpenAI, Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
- [HELM Leaderboard](https://crfm.stanford.edu/helm/latest/)
- [LMArena Leaderboard](https://lmarena.ai/)
- [Artificial Analysis](https://artificialanalysis.ai/)
- [OpenRouter Rankings](https://openrouter.ai/rankings)

## 来源与更新记录

- 来源：本章以公开评估资料、论文与模型技术报告为主；现代榜单、模型分数和 benchmark 声明需要按官方榜单、论文或模型卡逐条复核。
- 课程映射：Lecture 12 提供 difficulty、realism、validity、rules of the game、contamination、agent benchmark 和安全评估主线。
- 来源说明：现代榜单、模型分数和 benchmark 声明随官方来源更新。
- 查阅日期：2026-05-28（首次成体系核证）；2026-09-05（Section 11.4.1 MMLU Table 1、Section 11.4.2 MMLU-Pro、Section 11.4.3 GPQA 分档准确率、Section 11.5.4 WildBench 小节编号与相关系数、Section 11.10.2 agentic benchmark 复核，参考文献补全）。
