"""
效果评测：基座模型 vs 微调模型对比

你会在这一步学到：
1. 通过 Ollama REST API 调用模型推理（本地部署，无需 GPU）
2. 结构化评测：医学领域问题 + 通用能力问题
3. 自动评分：用正则从模型回答中提取答案字母，比对标准答案
4. Bad case 自动提取：答错的题目归档，驱动数据迭代

用法：
    python evaluate.py                                    # 完整评测（基座+微调对比）
    python evaluate.py --base-only                        # 仅评测基座模型（建立基线）
    python evaluate.py --model my-qwen-finetuned          # 指定微调模型名

前提：
    ollama serve 已启动，基座/微调模型已加载
"""

import argparse
import json
import re
from pathlib import Path

import requests

# ── 配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

OLLAMA_BASE_URL = "http://localhost:11434"
BASE_MODEL = "qwen2.5:1.5b"
DEFAULT_FINETUNED_MODEL = "my-qwen"

# 评测问题集路径（使用 01-data 的测试集）
TEST_DATA_PATH = PROJECT_DIR / "01-data" / "cleaned" / "cmexam_test.json"

# 通用能力问题（检测微调后通用能力是否退化）
GENERAL_QUESTIONS = [
    {"question": "请用一句话解释什么是机器学习。", "category": "general"},
    {"question": "中国的首都是哪座城市？", "category": "general"},
    {"question": "请写一首关于春天的五言绝句。", "category": "general"},
    {"question": "1+2+3+...+100等于多少？请给出计算过程。", "category": "general"},
    {"question": "请解释「亡羊补牢」这个成语的意思。", "category": "general"},
]


def call_ollama(model: str, question: str) -> str:
    """调用 Ollama API 推理"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": question,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_answer(text: str) -> str | None:
    """
    从模型回答中提取答案字母（支持"答案：A""选A""A."等多种格式）
    """
    patterns = [
        r"答案[是为：:]\s*([A-E])",  # 答案：A / 答案是A
        r"选\s*([A-E])",             # 选A / 选择 A
        r"([A-E])[。.]",             # A。 / A.
        r"\b([A-E])\b",              # 独立的 A
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    return None


def load_test_questions() -> list[dict]:
    """加载测试集问题"""
    if not TEST_DATA_PATH.exists():
        print(f"警告: 测试数据不存在 ({TEST_DATA_PATH})，仅使用通用问题")
        return []

    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        conv = item["conversations"]
        human_msg = conv[0]["value"]
        gpt_msg = conv[1]["value"]
        # 从标准答案中提取字母
        correct = extract_answer(gpt_msg)
        questions.append({
            "question": human_msg,
            "correct_answer": correct,
            "category": "medical",
        })
    return questions


def evaluate_model(model: str, questions: list[dict]) -> list[dict]:
    """对模型进行评测，返回结果列表"""
    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] 评测中...", end="\r")
        response = call_ollama(model, q["question"])
        predicted = extract_answer(response)

        results.append({
            "question": q["question"],
            "category": q["category"],
            "correct_answer": q.get("correct_answer"),
            "predicted_answer": predicted,
            "response": response,
            "correct": predicted == q.get("correct_answer") if q.get("correct_answer") and predicted else None,
        })

    print(f"  [{len(questions)}/{len(questions)}] 评测完成   ")
    return results


def compute_metrics(results: list[dict]) -> dict:
    """计算评测指标"""
    # 选择题自动评分（仅对有标准答案的题目）
    auto_gradable = [r for r in results if r["correct_answer"] is not None]
    correct_count = sum(1 for r in auto_gradable if r["correct"] is True)

    metrics = {
        "total": len(results),
        "auto_gradable": len(auto_gradable),
        "auto_correct": correct_count,
        "accuracy": correct_count / len(auto_gradable) if auto_gradable else 0,
    }

    # 按类别统计
    categories = set(r["category"] for r in results)
    metrics["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in auto_gradable if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"] is True)
        metrics["by_category"][cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": cat_correct / len(cat_results) if cat_results else 0,
        }

    return metrics


def find_bad_cases(results: list[dict]) -> list[dict]:
    """提取 bad case（答错或无法提取答案的题目）"""
    bad = []
    for r in results:
        if r["correct_answer"] is None:
            continue
        if r["correct"] is False or r["predicted_answer"] is None:
            bad.append(r)
    return bad


def save_results(results: list[dict], metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  → 结果保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="效果评测：基座 vs 微调")
    parser.add_argument("--base-only", action="store_true", help="仅评测基座模型")
    parser.add_argument("--model", default=DEFAULT_FINETUNED_MODEL, help="微调模型名称")
    args = parser.parse_args()

    # 加载问题
    print("[1/3] 加载评测问题")
    medical_questions = load_test_questions()
    all_questions = medical_questions + GENERAL_QUESTIONS
    print(f"  → 领域问题 {len(medical_questions)} 条 + 通用问题 {len(GENERAL_QUESTIONS)} 条")

    # 评测基座模型
    print(f"\n[2/3] 评测基座模型: {BASE_MODEL}")
    base_results = evaluate_model(BASE_MODEL, all_questions)
    base_metrics = compute_metrics(base_results)
    save_results(base_results, base_metrics, BASE_DIR / "results_base.json")
    print(f"  准确率: {base_metrics['accuracy']:.1%}")

    if args.base_only:
        return

    # 评测微调模型
    print(f"\n[3/3] 评测微调模型: {args.model}")
    ft_results = evaluate_model(args.model, all_questions)
    ft_metrics = compute_metrics(ft_results)
    save_results(ft_results, ft_metrics, BASE_DIR / "results_finetuned.json")
    print(f"  准确率: {ft_metrics['accuracy']:.1%}")

    # 对比
    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    print(f"{'指标':<20} {'基座':>10} {'微调':>10} {'变化':>10}")
    print("-" * 50)
    print(f"{'总准确率':<20} {base_metrics['accuracy']:>9.1%} {ft_metrics['accuracy']:>9.1%} {ft_metrics['accuracy'] - base_metrics['accuracy']:>+9.1%}")
    for cat in base_metrics["by_category"]:
        b = base_metrics["by_category"][cat]["accuracy"]
        f = ft_metrics["by_category"].get(cat, {}).get("accuracy", 0)
        print(f"  {cat:<18} {b:>9.1%} {f:>9.1%} {f - b:>+9.1%}")

    # bad case
    bad_cases = find_bad_cases(ft_results)
    if bad_cases:
        bad_path = BASE_DIR / "bad_cases.json"
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        print(f"\nbad case: {len(bad_cases)} 条 → {bad_path}")
        print("请分析 bad case，补充数据后进入下一轮训练")


if __name__ == "__main__":
    main()
