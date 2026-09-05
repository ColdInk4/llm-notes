# LLM Notes

这是一份面向长期维护的大语言模型学习笔记，内容整理自 `datawhalechina/diy-llm` 的 `docs/` 章节，并持续结合 Stanford CS336、公开论文、模型技术报告和工程实践更新。仓库结构、阅读顺序、章节写作约定见 [前言](docs/00_前言.md)；写作与来源日期约定见 [STYLE.md](STYLE.md)；课程来源对应关系见 [CS336 2026 来源映射](sources/cs336-2026.md)。

## 阅读入口

- [前言](docs/00_前言.md)
- [第 1 章 分词器](docs/chapter1/chapter1_分词器.md)
- [第 2 章 PyTorch 与资源核算](docs/chapter2/chapter2_pytorch与资源核算.md)
- [第 3 章 语言模型架构与训练细节](docs/chapter3/chapter3_语言模型架构和训练技术细节.md)
- [第 4 章 混合专家模型](docs/chapter4/chapter4_混合专家模型.md)
- [第 5 章 GPU 与相关优化](docs/chapter5/chapter5_GPU和GPU相关优化.md)
- [第 6 章 GPU 高性能编程](docs/chapter6/chapter6_GPU高性能编程.md)
- [第 7 章 分布式训练](docs/chapter7/chapter7_分布式训练.md)
- [第 8 章 Scaling Laws](docs/chapter8/chapter8_Scaling_Laws.md)
- [第 9 章 推理系统](docs/chapter9/chapter9_推理系统.md)
- [第 10 章 数据工程](docs/chapter10/chapter10_数据工程.md)
- [第 11 章 评估与基准测试](docs/chapter11/chapter11_评估与基准测试.md)
- [第 12 章 大模型的基本训练流程](docs/chapter12/chapter12_大模型基本训练流程.md)
- [第 13 章 可验证奖励的强化学习](docs/chapter13/chapter13_可验证奖励的强化学习.md)
- [第 14 章 多模态模型](docs/chapter14/chapter14_多模态模型.md)
- [专题：推理行为与能力](docs/topics/reasoning_behavior.md)
- [CS336 2026 来源映射](sources/cs336-2026.md)
- [写作与维护风格](STYLE.md)

## 仓库结构

- `docs/`：稳定章节笔记，命名约定 `chapterN_标题.md`，前言为 `00_前言.md`。
- `docs/topics/`：不占主章节编号的专题补充。
- `sources/`：公开来源映射和课程材料对应关系。
- `STYLE.md`：Markdown、图片、来源日期和文件命名约定。

本仓库不保留站点构建产物，例如 `index.html`、`.nojekyll` 和未引用的旧资源文件。

## 来源与许可证

本笔记基于 `datawhalechina/diy-llm` 的文档内容改编，并继续沿用原作品的 Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 许可证。详见 [LICENSE.md](LICENSE.md) 和 [NOTICE.md](NOTICE.md)。仓库用于个人学习与公开非商业分享。
