# LLM-Journey 技术设计

## 1. 项目定位

以 Qwen2.5-1.5B 为基座，亲手跑通 **数据处理 → LoRA 微调 → 效果评测 → 推理部署 → 迭代调试** 全链路，形成可复用的学习与实践范式。

---

### 1.1 微调场景选型

在启动链路前，需要确定微调的目标场景。场景决定了数据策略、评测设计和成功判定。

| 候选场景 | 数据可获取性 | 评测可量化性 | 学习价值 | 推荐度 |
|----------|-------------|-------------|---------|--------|
| 医学问答（CMExam / MedQA） | 高（公开数据集） | 高（有标准答案） | 高（领域适配典型） | ★★★ |
| 代码生成（CodeAlpaca / MBPP） | 高 | 高（可执行验证） | 高（推理能力增强） | ★★★ |
| 中文文言文翻译（WMT / 自构造） | 中 | 中（需人工判定流畅度） | 中 | ★★ |
| 客服对话（自构造） | 低（需业务数据） | 低（主观评判） | 中 | ★ |

**选择标准**：
1. 数据可公开获取，无需业务脱敏
2. 有客观评测基准，避免纯主观打分
3. 与基座能力有明确差距，微调收益可感知

**推荐**：医学问答或代码生成，二者均有高质量公开数据且评测可量化。

---

### 1.2 环境搭建

#### 本地环境（Apple M2 16GB）

| 项目 | 版本/规格 | 说明 |
|------|----------|------|
| macOS | Ventura+ | 原生 ARM 生态 |
| Python | 3.12 | uv 管理，推荐 3.11+，实际已验证 3.12 兼容 |
| uv | 最新版 | Python 环境与依赖管理 |
| Ollama | 最新版 | 本地推理引擎 |

```bash
# 初始化项目
uv init --python 3.12
uv add pandas datasets transformers accelerate

# 安装 LLaMA-Factory（开发模式，方便查看和修改源码）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

# 安装 Ollama
brew install ollama
ollama pull qwen2.5:1.5b

# wandb
pip install wandb
wandb login
```

#### 云端环境（AutoDL RTX 4090）

| 项目 | 版本/规格 | 说明 |
|------|----------|------|
| 镜像 | PyTorch 2.5 + Python 3.12 | AutoDL 社区镜像 |
| CUDA | 12.1+ | 4090 原生支持 bf16 |
| 显存 | 24GB | LoRA 微调 1.5B 绰绰有余 |
| 磁盘 | 50GB+ | 模型权重 + 数据 |

```bash
# AutoDL 实例内
pip install -e ".[torch,metrics]"
pip install wandb
wandb login
```

#### 版本锁定

关键依赖版本记录于 `pyproject.toml`，首次跑通后执行 `uv pip freeze > requirements-lock.txt` 固定版本，确保实验可复现。

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

#### 数据集划分

| 子集 | 比例 | 用途 | 备注 |
|------|------|------|------|
| 训练集 | 80% | LoRA 微调 | 主要训练数据 |
| 验证集 | 10% | 训练过程中 loss 监控 | 防过拟合，决定 early stopping |
| 测试集 | 10% | 最终评测，不参与训练 | 严格 held-out，仅评测时使用 |

划分方式：按对话随机 shuffle 后按比例切分，确保各子集领域分布一致。测试集在训练全程不可见。

#### 产出物

| 文件 | 说明 |
|------|------|
| `raw/` | 原始数据备份 |
| `cleaned/` | 清洗后数据 |
| `dataset_info.json` | LLaMA-Factory 数据集注册 |

---

#### 云端-本地数据同步

| 方向 | 数据 | 方式 | 预估大小 |
|------|------|------|---------|
| 本地 → 云端 | 清洗后训练数据 | `scp -r 01-data/cleaned/ root@autodl:/root/llm-journey/01-data/cleaned/` | < 100MB |
| 云端 → 本地 | LoRA adapter 权重 | AutoDL 文件管理器下载 / `scp` | 50-200MB |
| 云端 → 本地 | 训练日志 | AutoDL 文件管理器下载 | < 10MB |

**同步检查清单**：
- 上传前：本地 `sha256sum` 校验文件完整性
- 下载后：对比 adapter 文件数与云端一致
- 数据一致性：训练前在云端打印数据集条数，与本地核对

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

#### 量化成功标准

| 指标 | 基座基线 | 微调目标 | 说明 |
|------|---------|---------|------|
| 领域准确率 | 记录基座值 | ≥ 基座 +15% | 核心指标，微调首要目标 |
| 格式合规率 | 记录基座值 | ≥ 90% | 回答是否遵循指令格式 |
| 通用能力保持率 | 100%（基座自身） | ≥ 85% | 通用问题得分不低于基座的 85% |
| 完整性 | 记录基座值 | ≥ 基座 +10% | 回答信息是否完整 |

**评分规则**（每题 1-5 分）：

| 分数 | 标准 |
|------|------|
| 5 | 完全正确，格式规范，信息完整 |
| 4 | 基本正确，小瑕疵 |
| 3 | 方向正确但不完整或有明显缺漏 |
| 2 | 部分正确但偏离核心 |
| 1 | 完全错误或拒绝回答 |

每道题由至少 2 人独立评分取均值，如分差 ≥ 2 则仲裁。

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
# 1. 合并权重（LLaMA-Factory 一键合并 LoRA 到基座）
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-1.5B \
  --adapter_name_or_path ./output \
  --export_dir ./merged \
  --export_size 2 \
  --export_legacy_format false

# 2. 转 GGUF（依赖 llama.cpp，需确认版本兼容性）
# 安装 llama.cpp（推荐从源码编译以匹配当前环境）
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j && cd ..

# 转换为 f16 GGUF
python llama.cpp/convert_hf_to_gguf.py ./merged --outtype f16 --outfile ./merged/model-f16.gguf

# 3. 量化为 Q4_K_M
./llama.cpp/llama-quantize ./merged/model-f16.gguf ./merged/model-Q4_K_M.gguf Q4_K_M

# 4. Ollama 注册
ollama create my-qwen -f Modelfile
ollama run my-qwen
```

> **注意**：llama.cpp 接口变更频繁，以上命令基于 2025-05 版本。如遇 `convert_hf_to_gguf.py` 路径变动，请检查 `llama.cpp/` 目录下的实际脚本位置。

#### Modelfile 模板

```dockerfile
FROM ./merged/model-Q4_K_M.gguf

# 系统提示词（根据微调场景调整）
SYSTEM """你是一个专业的领域助手。请基于你的知识准确回答用户问题。如果不确定，请诚实说明。"""

# 推理参数
PARAM temperature 0.3
PARAM top_p 0.9
PARAM max_tokens 512
PARAM repeat_penalty 1.1

# 对话模板
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
```

#### API 接口规范

Ollama 启动后默认提供 OpenAI 兼容接口：

```bash
# 启动服务
ollama serve

# 调用示例
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-qwen",
    "messages": [
      {"role": "system", "content": "你是一个专业的领域助手。"},
      {"role": "user", "content": "请解释什么是LoRA微调？"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容对话接口 |
| `/v1/completions` | POST | 文本续写接口 |
| `/api/generate` | POST | Ollama 原生生成接口 |
| `/api/tags` | GET | 列出已加载模型 |

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

#### 笔记规范

**文件命名**：`exp-NNN.md`（如 `exp-001.md`、`exp-002.md`）

**存放路径**：`05-notes/exp-NNN.md`

**与 wandb 关联**：每篇笔记开头记录 wandb run URL，便于回溯训练曲线。

#### 笔记模板

```markdown
# exp-001

- **日期**：2025-06-XX
- **wandb**：https://wandb.ai/xxx/runs/xxxxx
- **配置变更**：超参 / 数据 / prompt 变了什么
- **Loss 曲线**：关键指标（final train loss、eval loss、是否过拟合）
- **评测对比**：基座 vs 本轮 vs 上轮（分数表格）
- **bad case 清单**：
  | # | 问题 | 基座回答 | 微调回答 | 评分 | 归因 |
  |---|------|---------|---------|------|------|
  | 1 | ... | ... | ... | 2/5 | 训练数据缺少该知识点 |
- **下一步**：数据补充 or 超参调整
```

#### bad case → 数据补充工作流

```
1. 从评测 bad case 中提取共性模式（如"X 类问题全部答错"）
2. 针对性构造/收集补充数据（至少覆盖 bad case 的 2x 数量）
3. 补充数据走完整清洗 → 格式转换流程（01-data/）
4. 与原有训练集合并，注意去重
5. 启动新一轮训练，exp-NNN+1
```

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

## 5. 成本估算

### AutoDL GPU 训练成本

| 项目 | 估算 | 说明 |
|------|------|------|
| 模型参数 | 1.5B | Qwen2.5-1.5B |
| LoRA 可训练参数 | ~10M | rank=8, target=all |
| 训练数据量 | 5K-20K 条 | 取决于场景 |
| 单 epoch 时间 | 5-15 分钟 | 4090 + bf16 + batch=4 |
| 3 epoch 总时间 | 15-45 分钟 | 含 eval |
| AutoDL 4090 价格 | ~¥2.5/h | 社区镜像 |
| **单次训练成本** | **~¥1-2** | 3 epoch |
| **含调参（5-8 次）** | **~¥10-20** | 预算控制在 ¥50 以内 |

> 首次训练建议先跑 1 epoch 验证链路，确认无误后再跑完整 3 epoch。

---

## 6. 学习路径规划

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

## 7. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 4090 显存不足 | OOM 导致训练中断 | 降低 batch_size / 启用 gradient_checkpointing |
| LoRA 效果不明显 | 微调后无提升 | 增大 lora_rank / 补充高质量数据 |
| 量化后精度下降 | GGUF 部署效果变差 | 尝试 Q5_K_M 或 Q8_0 量化档位 |
| M2 推理慢 | 评测效率低 | 减少评测问题数 / 使用小 batch 推理 |
| 数据质量差 | 模型学偏 | 人工审核数据子集，建立质检流程 |
