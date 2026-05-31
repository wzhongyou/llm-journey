# llm-journey

从零跑通大模型 **数据工程 → LoRA 微调 → 效果评测 → 推理部署 → 迭代调试** 全链路。

以 Qwen2.5-1.5B 为基座，CMExam 中文医学问答为场景，每个环节一个脚本，直奔知识点。

## 环境

| 环境 | 硬件 | 职责 |
|------|------|------|
| 本地 | Apple M2 16GB | 推理验证、数据处理、Ollama 部署、评测 |
| 云端 | AutoDL RTX 4090 | LoRA 微调训练 |

## 快速开始

```bash
# 安装依赖
uv sync

# 1. 数据工程：下载 CMExam → 清洗 → ShareGPT 格式 → 划分
cd 01-data && python prepare_data.py

# 2. 训练（在 AutoDL 上）
cd 02-train && bash train.sh

# 3. 评测（基座模型先建立基线）
cd 03-eval && python evaluate.py --base-only
# 微调后对比
python evaluate.py

# 4. 部署
cd 04-serve && bash deploy.sh
```

## 项目结构

```
llm-journey/
├── 01-data/
│   └── prepare_data.py     # 下载 → 清洗 → 格式转换 → 划分
├── 02-train/
│   ├── configs/             # 训练 YAML 配置
│   │   └── qwen1.5b_lora_sft.yaml
│   └── train.sh             # 一键训练脚本
├── 03-eval/
│   └── evaluate.py          # 基座 vs 微调对比评测
├── 04-serve/
│   ├── Modelfile            # Ollama 模型定义
│   └── deploy.sh            # 合并 → 量化 → 部署
├── 05-notes/
│   └── exp-001.md           # 实验记录模板
├── dataset_info.json        # LLaMA-Factory 数据集注册
└── docs/
    └── llm-journey-design.md
```

## 进度

- [x] 环境搭建 + 项目初始化
- [ ] 数据准备（运行 prepare_data.py）
- [ ] LoRA 微调训练（AutoDL 上运行 train.sh）
- [ ] 效果评测（运行 evaluate.py）
- [ ] 合并权重 + Ollama 部署（运行 deploy.sh）
- [ ] bad case 分析 + 数据迭代
