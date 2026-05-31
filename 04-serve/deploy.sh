#!/bin/bash
# 推理部署：合并 LoRA → 量化 GGUF → 注册 Ollama
#
# 你会在这一步学到：
#   - LoRA adapter 是差异权重（~10M 参数），不能独立使用，需合并回基座
#   - GGUF 是 llama.cpp 专有格式，Q4_K_M 精度/体积平衡点
#   - Ollama 加载 GGUF 模型后提供 OpenAI 兼容 REST API
#
# 前提：本地已装 LLaMA-Factory（含 export），02-train/output/ 下有 adapter 权重

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MERGED_DIR="${SCRIPT_DIR}/merged"
ADAPTER_PATH="${PROJECT_DIR}/02-train/output"

echo "=== 1/3 合并 LoRA Adapter 到基座 ==="
echo "Adapter: ${ADAPTER_PATH}"
echo "输出: ${MERGED_DIR}"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --adapter_name_or_path "${ADAPTER_PATH}" \
    --export_dir "${MERGED_DIR}" \
    --export_size 2 \
    --export_legacy_format false

echo ""
echo "=== 2/3 转 GGUF 并量化 ==="
# 检查 llama.cpp 是否存在
LLAMA_CPP="${PROJECT_DIR}/llama.cpp"
if [ ! -d "${LLAMA_CPP}" ]; then
    echo "llama.cpp 不存在，正在克隆..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP}"
    cd "${LLAMA_CPP}" && make -j && cd "${SCRIPT_DIR}"
fi

# 转 f16 GGUF
python "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}" \
    --outtype f16 \
    --outfile "${MERGED_DIR}/model-f16.gguf"

# 量化为 Q4_K_M
"${LLAMA_CPP}/llama-quantize" \
    "${MERGED_DIR}/model-f16.gguf" \
    "${MERGED_DIR}/model-Q4_K_M.gguf" \
    Q4_K_M

echo ""
echo "=== 3/3 注册到 Ollama ==="
ollama create my-qwen -f "${SCRIPT_DIR}/Modelfile"
echo ""
echo "部署完成！运行以下命令测试："
echo "  ollama run my-qwen"
echo "  curl http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"my-qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
