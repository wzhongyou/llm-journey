# llm-journey

从零跑通大模型 **数据工程 → LoRA 微调 → 效果评测 → 推理部署 → 迭代调试** 全链路。

以 Qwen2.5-1.5B 为基座，CMExam 中文医学问答为场景，**每步只学一个核心知识点**。

---

## 学习路径

可以先通读 [技术设计文档](docs/llm-journey-design.md) 了解全貌，再按顺序走每个环节。

```
                        ┌─────────────────────┐
                        │   技术设计文档        │
                        │   docs/llm-journey-  │
                        │   design.md          │
                        │   ↑ 先读这个了解全局   │
                        └─────────┬───────────┘
                                  │
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  01-data         │    │  02-train        │    │  03-eval         │
│  数据工程         │───▶│  LoRA 微调       │───▶│  效果评测         │
│                  │    │                  │    │                  │
│  学什么：         │    │  学什么：         │    │  学什么：         │
│  • HuggingFace   │    │  • LoRA 原理     │    │  • Ollama API    │
│  • ShareGPT 格式 │    │  • YAML 配置     │    │  • 自动评分      │
│  • 数据集划分     │    │  • wandb 监控    │    │  • Bad case 分析 │
│                  │    │  • bf16 / 4090   │    │                  │
│  📄 prepare_     │    │  📄 train.sh     │    │  📄 evaluate.py  │
│     data.py      │    │  📄 configs/*.yaml│   │                  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                      │                        │
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  04-serve        │    │  05-notes        │    │  设计文档         │
│  推理部署         │───▶│  迭代记录         │───▶│  docs/           │
│                  │    │                  │    │                  │
│  学什么：         │    │  学什么：         │    │  反复查阅         │
│  • LoRA 合并     │    │  • 实验对比      │    │                  │
│  • GGUF 量化     │    │  • Bad case      │    │                  │
│  • Ollama 部署   │    │    驱动迭代       │    │                  │
│                  │    │                  │    │                  │
│  📄 deploy.sh    │    │  📄 exp-001.md   │    │                  │
│  📄 Modelfile    │    │                  │    │                  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

> **💡 阅读顺序建议**：每个环节只读对应的一两个文件，代码量都在 200 行以内，一次消化一个概念。

---

## 环境

| 环境 | 硬件 | 职责 |
|------|------|------|
| 本地 💻 | Apple M2 16GB | 推理验证、数据处理、Ollama 部署、评测 |
| 云端 ☁️ | AutoDL RTX 4090 | LoRA 微调训练 |

## 快速开始

```bash
# 安装依赖
uv sync

# 1. 数据工程：下载 CMExam → 清洗 → ShareGPT 格式 → 划分
cd 01-data && python prepare_data.py

# 2. 训练（在 AutoDL 上跑）
cd 02-train && bash train.sh

# 3. 评测（先基座建立基线，训练后再对比）
cd 03-eval && python evaluate.py --base-only
python evaluate.py

# 4. 部署
cd 04-serve && bash deploy.sh
```

## 项目结构

```
llm-journey/
├── 01-data/
│   └── prepare_data.py          下载 → 清洗 → 格式转换 → 划分
├── 02-train/
│   ├── configs/
│   │   └── qwen1.5b_lora_sft.yaml  LoRA 训练配置（参数详解见注释）
│   └── train.sh                   一键训练脚本
├── 03-eval/
│   └── evaluate.py               基座 vs 微调对比评测
├── 04-serve/
│   ├── Modelfile                 Ollama 模型定义
│   └── deploy.sh                 合并 → 量化 → 部署
├── 05-notes/
│   └── exp-001.md                实验记录模板
├── dataset_info.json             LLaMA-Factory 数据集注册
└── docs/
    └── llm-journey-design.md     技术设计文档（先读）
```

## 进度

- [x] 环境搭建 + 项目初始化
- [ ] 数据准备（运行 prepare_data.py）
- [ ] LoRA 微调训练（AutoDL 上运行 train.sh）
- [ ] 效果评测（运行 evaluate.py）
- [ ] 合并权重 + Ollama 部署（运行 deploy.sh）
- [ ] bad case 分析 + 数据迭代
