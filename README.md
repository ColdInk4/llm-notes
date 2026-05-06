# LLM Notes

这是一份面向长期维护的大语言模型学习笔记，内容从
`datawhalechina/diy-llm` 的 `docs/` 章节整理而来，并会继续结合
Stanford CS336、公开论文、模型技术报告和工程实践进行更新。

本仓库只保留笔记内容，不包含原项目中的课程实践组织方式。后续新增的前沿模型、
训练设置、推理系统、硬件和评测信息，会优先进入 `frontier/`，等内容稳定后再整理
进具体章节。

## 阅读入口

- [前言](docs/前言.md)
- [第1章 工具使用](docs/chapter1/wandb使用介绍.md)
- [第2章 分词器](docs/chapter2/chapter2_分词器.md)
- [第3章 PyTorch 与资源核算](docs/chapter3/chapter3_pytorch与资源核算.md)
- [第4章 语言模型架构与训练细节](docs/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md)
- [第5章 混合专家模型](docs/chapter5/chapter5_混合专家模型.md)
- [第6章 GPU 与相关优化](docs/chapter6/chapter6_第六章GPU和GPU相关的优化.md)
- [第7章 GPU 高性能编程](docs/chapter7/chapter7_第七章GPU高性能编程.md)
- [第8章 分布式训练](docs/chapter8/chapter8_第八章分布式训练.md)
- [第9章 Scaling Laws](docs/chapter9/chapter9_Scaling_Laws.md)
- [第10章 推理](docs/chapter10/推理.md)
- [第11章 数据工程](docs/chapter11/chapter11_数据工程.md)
- [第12章 评估与基准测试](docs/chapter12/chapter12_评估与基准测试.md)
- [第13章 大模型的基本训练流程](docs/chapter13/chapter13_第十三章大模型的基本训练流程.md)
- [第14章 可验证奖励的强化学习](docs/chapter14/chapter14_可验证奖励的强化学习.md)
- [第15章 扩展内容：什么是 LLM 推理](docs/chapter15/什么是LLM推理.md)

## 维护说明

- 稳定章节优先写经过反复验证的概念、公式、代码解释和工程经验。
- 快速变化的信息，例如新模型参数、硬件指标、榜单表现和厂商声明，需要标注来源与
  查阅日期。
- Stanford CS336 2026 课件会作为重要更新来源，但本仓库不是课程镜像，也不局限于
  课程内容。
- 外部图片链接暂时保留；如果本仓库已有对应本地图片，后续会逐步改为相对路径。

## 来源与许可证

本笔记基于 `datawhalechina/diy-llm` 的文档内容改编，并继续沿用原作品的
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
许可证。详见 [LICENSE.md](LICENSE.md) 和 [NOTICE.md](NOTICE.md)。

本仓库用于个人学习与公开非商业分享。
