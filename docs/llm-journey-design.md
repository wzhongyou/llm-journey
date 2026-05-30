# LLM-Journey 技术设计

## 1. 项目定位

以 Qwen2.5-1.5B 为基座，亲手跑通 **数据处理 → LoRA 微调 → 效果评测 → 推理部署 → 迭代调试** 全链路，形成可复用的学习与实践范式。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    llm-journey 全链路                        │
│                                                             │
│  01-data ──▶ 02-train ──▶ 03-eval ──▶ 04-serve             │
│  数据工程     LoRA微调     效果评测    推理部署              │
│      ▲                                    │                 │
│      └──────── 05-notes ◀────────────────┘                 │
│            迭代记录 & bad case 驱动数据回流                  │
└─────────────────────────────────────────────────────────────┘
```

### 双环境分工

| 环境 | 硬件 | 职责 |
|------|------|------|
| 本地 | Apple M2 16GB | 推理验证、数据处理、Ollama 部署、评测 |
| 云端 | AutoDL RTX 4090 | LoRA 微调训练（GPU 密集） |

---

## 3. 各阶段技术设计

### 3.1 数据工程（01-data/）

**目标**：为微调准备高质量、格式合规的训练集。

```
原始数据 ──▶ 清洗 ──▶ 去重/质检 ──▶ 格式转换 ──▶ 训练集/验证集
                                              (sharegpt/alpaca)
```

#### 关键决策

| 项目 | 方案 | 理由 |
|------|------|------|
| 数据格式 | ShareGPT（多轮对话） | LLaMA-Factory 原生支持，兼容多轮 |
| 数据源 | 领域公开数据集 + 自构造 | 根据微调场景选择 |
| 清洗工具 | pandas | 轻量，本地可跑 |
| 存储 | HuggingFace datasets | 与训练链路无缝衔接 |

#### 数据格式规范

```json
{
  "conversations": [
    {"from": "human", "value": "问题内容"},
    {"from": "gpt", "value": "回答内容"}
  ]
}
```

#### 产出物

| 文件 | 说明 |
|------|------|
| `raw/` | 原始数据备份 |
| `cleaned/` | 清洗后数据 |
| `dataset_info.json` | LLaMA-Factory 数据集注册 |

---

### 3.2 LoRA 微调训练（02-train/）

**目标**：在 4090 上以最低成本获得领域适配模型。

```
Qwen2.5-1.5B ──▶ LoRA Adapter 训练 ──▶ Adapter 权重
                      │
                 wandb 监控
```

#### 关键决策

| 项目 | 方案 | 理由 |
|------|------|------|
| 训练框架 | LLaMA-Factory | 统一 CLI/YAML 配置，开箱即用 |
| 微调方式 | LoRA (r=8/16) | 1.5B 模型 LoRA 即可，显存友好 |
| 训练精度 | bf16 | 4090 原生支持，稳定高效 |
| 监控 | wandb | 实时 loss 曲线，跨实验对比 |

#### 超参基线

```yaml
# 初版超参，根据 loss 曲线调整
model_name_or_path: Qwen/Qwen2.5-1.5B
stage: sft
finetuning_type: lora
lora_rank: 8
lora_target: all           # 全线性层挂 LoRA
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
bf16: true
logging_steps: 10
save_steps: 500
```

#### 产出物

| 文件 | 说明 |
|------|------|
| `configs/` | YAML 训练配置 |
| `output/` | LoRA adapter 权重 + 训练日志 |
| `wandb/` | 实验记录（gitignore） |

---

### 3.3 效果评测（03-eval/）

**目标**：量化微调收益，建立基线 → 微调的对比体系。

```
评测问题集 ──▶ 基座模型推理 ──▶ 微调模型推理 ──▶ 对比评分
     │                                          │
     └──────── bad case 标注 ◀──────────────────┘
```

#### 评测方法

| 层级 | 方法 | 工具 |
|------|------|------|
| 主观评测 | 人工对比（基座 vs 微调） | 结构化问题集 + 评分表 |
| 客观评测 | 自动指标（如有标准答案） | 脚本计算 |
| 回归评测 | 原有能力是否退化 | 通用能力问题集 |

#### 评测流程

1. 构造 20-50 条领域问题 + 通用问题
2. 分别用基座和微调模型推理，记录回答
3. 按 **准确性 / 完整性 / 格式合规** 打分
4. bad case 打标，回流到 01-data 驱动下一轮迭代

#### 产出物

| 文件 | 说明 |
|------|------|
| `questions.json` | 评测问题集 |
| `results_base.json` | 基座模型回答 |
| `results_finetuned.json` | 微调模型回答 |
| `comparison.md` | 对比分析报告 |

---

### 3.4 推理部署（04-serve/）

**目标**：将微调后的模型通过 Ollama 本地部署，实现低延迟推理。

```
LoRA Adapter ──▶ 合并到基座 ──▶ GGUF 量化 ──▶ Ollama 加载 ──▶ API 服务
```

#### 关键决策

| 项目 | 方案 | 理由 |
|------|------|------|
| 合并方式 | LLaMA-Factory export | 一键合并 LoRA 到基座 |
| 量化格式 | GGUF Q4_K_M | 精度/体积平衡，M2 16GB 可跑 |
| 部署工具 | Ollama | 本地一键加载，REST API 开箱即用 |
| API 格式 | OpenAI-compatible | 通用接口，方便对接前端 |

#### 部署流程

```bash
# 1. 合并权重
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-1.5B \
  --adapter_name_or_path ./output --export_dir ./merged

# 2. 转 GGUF
python -m llama_cpp.convert_hf_to_gguf ./merged --outtype f16

# 3. 量化
./llama-quantize ./merged/ggml-model-f16.gguf Q4_K_M

# 4. Ollama 注册
ollama create my-qwen -f Modelfile
ollama run my-qwen
```

#### 产出物

| 文件 | 说明 |
|------|------|
| `Modelfile` | Ollama 模型定义 |
| `serve.py` | API 服务封装（可选） |

---

### 3.5 迭代调试（05-notes/）

**目标**：记录每轮实验，bad case 驱动数据迭代。

```
评测 bad case ──▶ 分析原因 ──▶ 补充数据 ──▶ 重新训练 ──▶ 再评测
```

#### 笔记模板

每轮记录：

- **实验 ID**：exp-001
- **配置变更**：超参 / 数据 / prompt 变了什么
- **Loss 曲线**：wandb 截图 / 关键指标
- **评测对比**：基座 vs 本轮 vs 上轮
- **bad case 清单**：哪些问题没答好，归因
- **下一步**：数据补充 or 超参调整

---

## 4. 数据流全景

```
                    ┌──────────────┐
                    │  原始数据集   │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  清洗 + 转格式 │  01-data (本地 M2)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 上传到云端     │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  LoRA 微调训练          │  02-train (AutoDL 4090)
              │  Qwen2.5-1.5B + LoRA   │
              │  wandb 监控             │
              └────────────┬───────────┘
                           ▼
                    ┌──────────────┐
                    │ 下载 Adapter  │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  效果评测               │  03-eval (本地 M2)
              │  基座 vs 微调对比       │
              │  bad case 标注          │
              └────────────┬───────────┘
                           ▼
                    ┌──────────────┐
                    │ 合并 + 量化   │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  Ollama 部署            │  04-serve (本地 M2)
              │  GGUF Q4_K_M           │
              │  REST API              │
              └────────────┬───────────┘
                           │
                    ┌──────▼───────┐
                    │ bad case 分析 │  05-notes
                    │ 数据回流      │────▶ 回到 01-data
                    └──────────────┘
```

---

## 5. 学习路径规划

按周推进，每周有明确产出和验收标准：

### Week 1 — 环境搭建 + 基线记录

- [ ] 本地安装 uv、Ollama，拉取 Qwen2.5-1.5B
- [ ] 基座模型推理 5-10 个问题，记录回答
- [ ] AutoDL 注册，验证 GPU 可用
- [ ] 确定微调场景（领域/任务）
- **验收**：基座能推理，有基线记录

### Week 2 — 数据准备

- [ ] 收集/构造领域数据
- [ ] 清洗 → ShareGPT 格式转换
- [ ] 注册到 LLaMA-Factory dataset_info.json
- [ ] 本地跑通 LLaMA-Factory 数据加载
- **验收**：数据集可用，训练框架能读取

### Week 3 — LoRA 微调训练

- [ ] 编写训练 YAML 配置
- [ ] AutoDL 上启动训练，wandb 监控
- [ ] 观察 loss 曲线，调整超参
- [ ] 下载 adapter 权重到本地
- **验收**：训练完成，loss 收敛

### Week 4 — 效果评测

- [ ] 构造评测问题集
- [ ] 基座 vs 微调对比评测
- [ ] bad case 标注与分析
- **验收**：有对比报告，识别改进方向

### Week 5 — 推理部署

- [ ] 合并 adapter → 完整权重
- [ ] GGUF 量化 + Ollama 部署
- [ ] REST API 调通
- **验收**：本地可调 API 推理

### Week 6 — 迭代优化

- [ ] bad case 驱动数据补充
- [ ] 第二轮微调
- [ ] 再评测，对比 v1 vs v2
- [ ] 整理笔记，输出总结
- **验收**：微调 v2 有提升，有完整笔记

---

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 4090 显存不足 | OOM 导致训练中断 | 降低 batch_size / 启用 gradient_checkpointing |
| LoRA 效果不明显 | 微调后无提升 | 增大 lora_rank / 补充高质量数据 |
| 量化后精度下降 | GGUF 部署效果变差 | 尝试 Q5_K_M 或 Q8_0 量化档位 |
| M2 推理慢 | 评测效率低 | 减少评测问题数 / 使用小 batch 推理 |
| 数据质量差 | 模型学偏 | 人工审核数据子集，建立质检流程 |
