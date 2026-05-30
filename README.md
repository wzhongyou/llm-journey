# llm-journey

从零跑通大模型 **数据工程 → LoRA 微调 → 效果评测 → 推理部署 → 迭代调试** 全链路。

## 环境

| 环境 | 硬件 | 职责 |
|------|------|------|
| 本地 | Apple M2 16GB | 推理验证、数据处理、Ollama 部署、评测 |
| 云端 | AutoDL RTX 4090 | LoRA 微调训练 |

## 技术栈

- 环境管理：uv
- 基座模型：Qwen2.5-1.5B
- 数据处理：pandas + HuggingFace datasets
- 微调训练：LLaMA-Factory（LoRA）
- 训练监控：wandb
- 推理部署：Ollama

## 项目结构

```
llm-journey/
├── 01-data/           # 数据工程：清洗、转格式、数据集注册
│   ├── raw/           # 原始数据
│   └── cleaned/       # 清洗后数据（ShareGPT 格式）
├── 02-train/          # LoRA 微调：配置、训练输出
│   ├── configs/       # 训练 YAML 配置
│   └── output/        # Adapter 权重 + 训练日志
├── 03-eval/           # 效果评测：基座 vs 微调对比
├── 04-serve/          # 推理部署：合并权重、GGUF 量化、Ollama
├── 05-notes/          # 实验记录 & bad case 分析
└── docs/              # 技术设计文档
```

## 进度

- [ ] 环境搭建 + 基座推理跑通
- [ ] 数据准备（清洗 + 转格式）
- [ ] LoRA 微调训练
- [ ] 效果评测（基座 vs 微调对比）
- [ ] 合并权重 + Ollama 部署
- [ ] bad case 分析 + 数据迭代

## 详细设计

见 [docs/llm-journey-design.md](docs/llm-journey-design.md)
