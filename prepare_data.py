"""
数据准备脚本
自动下载和准备各阶段训练数据
"""
import os
import json
import argparse
from pathlib import Path


# 数据集信息
DATASETS = {
    "pretrain": {
        "name": "LLaVA-Pretrain (LCS-558K)",
        "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain",
        "description": "558K 图文对，用于 Stage 1 特征对齐预训练",
        "size": "~15GB (含图像)",
        "command": "git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain data/pretrain",
    },
    "sft": {
        "name": "LLaVA-v1.5-mix665k",
        "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K",
        "description": "665K 指令微调数据，用于 Stage 2 SFT",
        "size": "~50GB (含图像: COCO, GQA, OCR-VQA, TextVQA, VG)",
        "images": {
            "coco": "http://images.cocodataset.org/zips/train2017.zip",
            "gqa": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "textvqa": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        },
        "command": (
            "# 1. 下载标注文件\n"
            "wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json -P data/sft/\n"
            "# 2. 下载图像 (按需选择)\n"
            "mkdir -p data/sft/images\n"
            "# COCO train2017\n"
            "wget http://images.cocodataset.org/zips/train2017.zip -P data/sft/images/\n"
            "cd data/sft/images && unzip train2017.zip && cd -\n"
        ),
    },
    "dpo": {
        "name": "RLHF-V Dataset",
        "url": "https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset",
        "description": "偏好数据，用于 Stage 3 DPO 偏好对齐，减少幻觉",
        "size": "~5GB",
        "command": "git clone https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset data/dpo",
    },
}


def generate_dummy_data(stage: str, num_samples: int = 100):
    """
    生成模拟数据用于测试训练流程
    """
    output_dir = f"data/{stage}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # 生成模拟图像 (纯色小图)
    try:
        from PIL import Image
        import random
        for i in range(min(num_samples, 50)):
            img = Image.new("RGB", (384, 384),
                            color=(random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255)))
            img.save(os.path.join(output_dir, "images", f"dummy_{i:05d}.jpg"))
    except ImportError:
        print("PIL 未安装，跳过图像生成")

    if stage == "pretrain":
        data = []
        for i in range(num_samples):
            data.append({
                "id": f"dummy_{i:05d}",
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "conversations": [
                    {"from": "human", "value": f"<image>\nDescribe this image."},
                    {"from": "gpt", "value": f"This is a colorful image with index {i}."},
                ]
            })
        json_path = os.path.join(output_dir, "llava_pretrain_lcs558k.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Pretrain] 生成 {num_samples} 条模拟数据 -> {json_path}")

    elif stage == "sft":
        data = []
        for i in range(num_samples):
            data.append({
                "id": f"sft_{i:05d}",
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "conversations": [
                    {"from": "human", "value": f"<image>\nWhat do you see in this image?"},
                    {"from": "gpt", "value": f"I can see a colorful pattern in this image. The image appears to be a solid color block."},
                    {"from": "human", "value": "Can you describe the colors?"},
                    {"from": "gpt", "value": "The image contains a single dominant color that fills the entire frame."},
                ]
            })
        json_path = os.path.join(output_dir, "llava_v1_5_mix665k.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SFT] 生成 {num_samples} 条模拟数据 -> {json_path}")

    elif stage == "dpo":
        data = []
        for i in range(num_samples):
            data.append({
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "question": "Describe what you see in this image.",
                "chosen": "The image shows a solid color background with uniform coloring throughout.",
                "rejected": "The image shows a beautiful sunset over the ocean with dolphins jumping. There are also mountains in the background.",
            })
        json_path = os.path.join(output_dir, "rlhf_v_preference.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DPO] 生成 {num_samples} 条模拟数据 -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="数据准备工具")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["pretrain", "sft", "dpo", "all", "dummy"],
                        help="准备哪个阶段的数据")
    parser.add_argument("--dummy", action="store_true",
                        help="生成模拟数据用于测试")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="模拟数据样本数")
    args = parser.parse_args()

    if args.dummy or args.stage == "dummy":
        print("生成模拟数据用于流程测试...\n")
        for stage in ["pretrain", "sft", "dpo"]:
            generate_dummy_data(stage, args.num_samples)
        print("\n模拟数据生成完成! 可以运行:")
        print("  python train.py --stage pretrain")
        return

    stages = ["pretrain", "sft", "dpo"] if args.stage == "all" else [args.stage]

    for stage in stages:
        info = DATASETS[stage]
        print(f"\n{'='*60}")
        print(f"数据集: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"大小: {info['size']}")
        print(f"来源: {info['url']}")
        print(f"\n下载命令:")
        print(info['command'])
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
