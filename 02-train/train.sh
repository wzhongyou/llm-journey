#!/bin/bash
# LoRA 微调启动脚本
# 在 AutoDL 4090 实例上运行
#
# 知识点：
#   - llamafactory-cli 是 LLaMA-Factory 的统一命令行入口
#   - 训练前确保数据已上传到云端，dataset_info.json 已放到 LLaMA-Factory 目录
#   - wandb 需要先 wandb login

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 将项目根目录的 dataset_info.json 复制到 LLaMA-Factory
# (假设 LLaMA-Factory 在项目根目录的相邻位置)
LLAMA_FACTORY_DIR="${PROJECT_DIR}/LLaMA-Factory"
if [ -d "$LLAMA_FACTORY_DIR" ]; then
    cp "${PROJECT_DIR}/dataset_info.json" "${LLAMA_FACTORY_DIR}/data/dataset_info.json"
    echo "已复制 dataset_info.json 到 LLaMA-Factory"
fi

echo "启动 LoRA 微调训练..."
echo "配置: ${SCRIPT_DIR}/configs/qwen1.5b_lora_sft.yaml"
echo ""

llamafactory-cli train "${SCRIPT_DIR}/configs/qwen1.5b_lora_sft.yaml"

echo ""
echo "训练完成！Adapter 权重在: ${SCRIPT_DIR}/output/"
echo "请下载 output/ 目录到本地 02-train/output/"
