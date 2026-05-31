"""
数据工程全流程：下载 → 清洗 → 格式转换 → 划分

你会在这一步学到：
1. HuggingFace datasets 加载公开数据集（需网络，失败自动 fallback）
2. pandas 数据清洗：去重 → 去空 → 过滤短文本
3. 转 ShareGPT 格式：LLaMA-Factory 原生支持的对话格式，conversations 列表支持多轮
4. 训练/验证/测试集 80/10/10 划分，固定 seed 保证可复现

用法：
    python prepare_data.py                    # 下载 CMExam（需网络）
    python prepare_data.py --demo             # 用内置 demo 数据（无需网络，跑通链路）
    python prepare_data.py --skip-download    # 跳过下载，使用 raw/ 下的已有数据
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

# 国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
CLEANED_DIR = BASE_DIR / "cleaned"

DATASET_NAME = "FudanDISC/CMExam"

# ── 内置 demo 数据（30 条医学问答，网络不通时 fallback） ──
DEMO_DATA = [
    {"question": "以下哪种细胞负责免疫反应中的抗原呈递？\nA. 红细胞\nB. B细胞\nC. 树突状细胞\nD. 血小板", "answer": "C", "explanation": "树突状细胞是体内最有效的抗原呈递细胞，能将抗原信息传递给T细胞，启动免疫应答。"},
    {"question": "人体最大的器官是？\nA. 肝脏\nB. 皮肤\nC. 大脑\nD. 肺", "answer": "B", "explanation": "皮肤是人体最大的器官，成人皮肤面积约1.5-2平方米。"},
    {"question": "以下哪种维生素缺乏会导致夜盲症？\nA. 维生素A\nB. 维生素B\nC. 维生素C\nD. 维生素D", "answer": "A", "explanation": "维生素A是视紫红质的组成成分，缺乏时暗适应能力下降，导致夜盲症。"},
    {"question": "正常成人的心率范围是？\nA. 40-60次/分\nB. 60-100次/分\nC. 100-120次/分\nD. 120-160次/分", "answer": "B", "explanation": "正常成人静息心率为60-100次/分，低于60为心动过缓，高于100为心动过速。"},
    {"question": "以下哪种激素由胰岛β细胞分泌？\nA. 胰高血糖素\nB. 胰岛素\nC. 皮质醇\nD. 肾上腺素", "answer": "B", "explanation": "胰岛素由胰岛β细胞分泌，促进葡萄糖的摄取和利用，降低血糖。"},
    {"question": "肺部气体交换的主要场所是？\nA. 气管\nB. 支气管\nC. 肺泡\nD. 胸膜腔", "answer": "C", "explanation": "肺泡壁薄且血管丰富，是O₂和CO₂交换的主要场所。"},
    {"question": "以下哪种药物属于β-内酰胺类抗生素？\nA. 红霉素\nB. 青霉素\nC. 庆大霉素\nD. 四环素", "answer": "B", "explanation": "青霉素含有β-内酰胺环，是典型的β-内酰胺类抗生素。"},
    {"question": "人体安静时的主要产热器官是？\nA. 肌肉\nB. 肝脏\nC. 大脑\nD. 皮肤", "answer": "B", "explanation": "安静时机体主要依靠内脏产热，其中肝脏代谢最旺盛，产热量最大。"},
    {"question": "急性阑尾炎最常见的早期症状是？\nA. 右下腹痛\nB. 脐周或上腹痛\nC. 恶心呕吐\nD. 发热", "answer": "B", "explanation": "急性阑尾炎典型表现为转移性右下腹痛，早期为脐周或上腹不适，后转移至右下腹。"},
    {"question": "以下哪种血型被称为「万能受血者」？\nA. A型\nB. B型\nC. AB型\nD. O型", "answer": "C", "explanation": "AB型血红细胞上既有A抗原又有B抗原，血清中无抗A和抗B抗体，可接受任意血型的红细胞。"},
    {"question": "以下哪项不是细胞器的特征？\nA. 线粒体\nB. 内质网\nC. 核糖体\nD. 细胞膜", "answer": "D", "explanation": "细胞膜是细胞的结构，不属于细胞器。线粒体、内质网、核糖体都是细胞器。"},
    {"question": "高血压的诊断标准是收缩压和/或舒张压分别大于等于？\nA. 120/80 mmHg\nB. 130/85 mmHg\nC. 140/90 mmHg\nD. 160/100 mmHg", "answer": "C", "explanation": "根据中国高血压防治指南，非同日三次测量收缩压≥140mmHg和/或舒张压≥90mmHg即可诊断。"},
    {"question": "消化液中不含消化酶的是？\nA. 唾液\nB. 胃液\nC. 胆汁\nD. 胰液", "answer": "C", "explanation": "胆汁由肝细胞分泌，主要成分为胆盐，无消化酶，其作用是乳化脂肪。"},
    {"question": "以下哪种病原体引起肺结核？\nA. 葡萄球菌\nB. 结核分枝杆菌\nC. 肺炎链球菌\nD. 流感病毒", "answer": "B", "explanation": "肺结核由结核分枝杆菌引起，属于抗酸杆菌，抗酸染色呈红色。"},
    {"question": "人体内含量最多的矿物质是？\nA. 铁\nB. 锌\nC. 钙\nD. 磷", "answer": "C", "explanation": "钙是人体内含量最多的矿物质，主要存在于骨骼和牙齿中。"},
    {"question": "心动周期中，心室容积最大的时期是？\nA. 等容收缩期\nB. 快速射血期\nC. 减慢射血期\nD. 心室充盈期末", "answer": "D", "explanation": "心室充盈期末即心房收缩期末，心室容积达到最大值。"},
    {"question": "阿司匹林的主要作用机制是？\nA. 抑制COX酶\nB. 阻断钙通道\nC. 阻断β受体\nD. 抑制ACE酶", "answer": "A", "explanation": "阿司匹林通过不可逆抑制环氧化酶(COX)，减少血栓素A2的合成，发挥抗血小板聚集作用。"},
    {"question": "以下哪种疾病属于自身免疫性疾病？\nA. 感冒\nB. 系统性红斑狼疮\nC. 肺炎\nD. 骨折", "answer": "B", "explanation": "系统性红斑狼疮是典型的自身免疫性疾病，机体产生针对自身组织的抗体，导致多系统损害。"},
    {"question": "肾小球滤过率(GFR)的正常值约为？\nA. 50 ml/min\nB. 80 ml/min\nC. 125 ml/min\nD. 200 ml/min", "answer": "C", "explanation": "正常成人GFR约为125 ml/min，每日生成原尿约180L。"},
    {"question": "DNA复制时，引导链的合成方向是？\nA. 3'→5'\nB. 5'→3'\nC. 两个方向都有\nD. 随机方向", "answer": "B", "explanation": "DNA聚合酶只能沿5'→3'方向合成新链，引导链沿此方向连续合成。"},
    {"question": "以下哪种检查方法对骨折诊断最有价值？\nA. B超\nB. X线\nC. 心电图\nD. 血常规", "answer": "B", "explanation": "X线检查是骨折最基本、最常用的诊断方法，可显示骨折的部位、类型和移位情况。"},
    {"question": "甲状腺功能亢进时，基础代谢率会？\nA. 升高\nB. 降低\nC. 不变\nD. 先升高后降低", "answer": "A", "explanation": "甲状腺激素是调节基础代谢率最重要的激素，甲亢时甲状腺激素分泌增多，基础代谢率升高。"},
    {"question": "以下哪种体液属于细胞内液？\nA. 血浆\nB. 组织液\nC. 淋巴液\nD. 细胞质", "answer": "D", "explanation": "体液分为细胞内液和细胞外液，细胞质属于细胞内液，血浆和组织液属于细胞外液。"},
    {"question": "青霉素过敏休克的首选抢救药物是？\nA. 地塞米松\nB. 肾上腺素\nC. 异丙嗪\nD. 葡萄糖酸钙", "answer": "B", "explanation": "肾上腺素是过敏性休克的首选药物，可迅速收缩血管、增加心输出量、缓解支气管痉挛。"},
    {"question": "以下哪种细胞没有细胞核？\nA. 白细胞\nB. 红细胞\nC. 血小板\nD. 神经元", "answer": "B", "explanation": "成熟红细胞没有细胞核和细胞器，为双凹圆盘形，有利于携氧和通过毛细血管。"},
    {"question": "正常尿液中最主要的含氮废物是？\nA. 氨基酸\nB. 尿素\nC. 尿酸\nD. 肌酐", "answer": "B", "explanation": "尿素是蛋白质代谢的主要终产物，占尿液中含氮废物的80%以上。"},
    {"question": "影响药物作用的因素不包括？\nA. 剂量\nB. 给药途径\nC. 患者个体差异\nD. 药品价格", "answer": "D", "explanation": "药品价格是经济因素，不影响药物在体内的药理作用。"},
    {"question": "以下哪个解剖结构连接咽和中耳？\nA. 咽鼓管\nB. 乳突窦\nC. 外耳道\nD. 半规管", "answer": "A", "explanation": "咽鼓管连接鼻咽部和中耳鼓室，维持鼓膜内外压力平衡。"},
    {"question": "人体体温调节中枢位于？\nA. 大脑皮层\nB. 下丘脑\nC. 小脑\nD. 脊髓", "answer": "B", "explanation": "下丘脑是体温调节的基本中枢，通过调节产热和散热过程维持体温恒定。"},
    {"question": "2型糖尿病最常见的首发症状是？\nA. 多饮多尿\nB. 体重明显下降\nC. 视力模糊\nD. 皮肤感染", "answer": "A", "explanation": "2型糖尿病最典型的首发症状为「三多一少」：多饮、多尿、多食和体重下降，其中多饮多尿最为常见。"},
]


def download() -> pd.DataFrame:
    """从 HuggingFace 下载 CMExam 数据集，保存原始备份"""
    from datasets import load_dataset

    print(f"[1/4] 下载数据集: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    df = ds.to_pandas()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "cmexam_raw.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"  → 保存原始数据: {raw_path} ({len(df)} 条)")
    return df


def load_demo() -> pd.DataFrame:
    """加载内置 demo 数据（无需网络，跑通链路）"""
    print("[1/4] 使用内置 demo 数据（30 条医学问答）")
    df = pd.DataFrame(DEMO_DATA)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "cmexam_demo.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"  → 保存 demo 数据: {raw_path} ({len(df)} 条)")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：去重 → 去空 → 过滤短文本
    """
    print("[2/4] 数据清洗")
    before = len(df)

    df = df.drop_duplicates(subset=["question"])
    df = df.dropna(subset=["question", "answer"])
    df = df[df["question"].str.len() >= 5]
    df = df[df["answer"].str.len() >= 1]

    after = len(df)
    print(f"  → 清洗前 {before} 条 → 清洗后 {after} 条 (去除 {before - after} 条)")
    return df


def convert_to_sharegpt(df: pd.DataFrame) -> list[dict]:
    """
    转换为 ShareGPT 格式（LLaMA-Factory 原生支持）
    """
    print("[3/4] 格式转换 → ShareGPT")
    records = []
    for _, row in df.iterrows():
        gpt_value = f"答案：{row['answer']}"
        if "explanation" in df.columns and pd.notna(row.get("explanation")) and row["explanation"]:
            gpt_value += f"\n\n解析：{row['explanation']}"

        records.append({
            "conversations": [
                {"from": "human", "value": row["question"]},
                {"from": "gpt", "value": gpt_value},
            ]
        })

    print(f"  → 转换完成: {len(records)} 条 ShareGPT 对话")
    return records


def split_and_save(records: list[dict], seed: int = 42) -> None:
    """
    划分训练/验证/测试集（80/10/10），固定 seed 保证可复现
    """
    print("[4/4] 划分数据集")
    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = CLEANED_DIR / f"cmexam_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  → {name}: {len(data)} 条 → {path}")


def main():
    parser = argparse.ArgumentParser(description="数据工程：下载 → 清洗 → 转换 → 划分")
    parser.add_argument("--demo", action="store_true", help="使用内置 demo 数据（无需网络）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，使用 raw/ 下的已有数据")
    args = parser.parse_args()

    if args.demo:
        df = load_demo()
    elif args.skip_download:
        raw_path = RAW_DIR / "cmexam_raw.parquet"
        if not raw_path.exists():
            raw_path = RAW_DIR / "cmexam_demo.parquet"
        print(f"[1/4] 跳过下载，读取已有数据: {raw_path}")
        df = pd.read_parquet(raw_path)
        print(f"  → 读取 {len(df)} 条")
    else:
        try:
            df = download()
        except Exception as e:
            print(f"\n  ⚠ 下载失败: {e}")
            print("  → 自动切换到 demo 数据\n")
            df = load_demo()

    df = clean(df)
    records = convert_to_sharegpt(df)
    split_and_save(records)
    print("\n完成！数据已保存到 01-data/cleaned/")


if __name__ == "__main__":
    main()
