"""
数据集模块
支持三个阶段的数据加载:
- Stage 1 预训练: 图文对 (LCS-558K 格式)
- Stage 2 SFT: 多轮对话指令数据 (LLaVA-mix-665K 格式)
- Stage 3 DPO: 偏好对数据 (chosen/rejected 格式)

数据集下载指南:
- LCS-558K: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
- LLaVA-mix-665K: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- ShareGPT4V: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V
- RLHF-V: https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset
"""
import os
import json
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any
from PIL import Image
from transformers import SiglipImageProcessor

from models.litevl import IMAGE_TOKEN, IMAGE_TOKEN_INDEX


class PretrainDataset(Dataset):
    """
    Stage 1: 预训练数据集 (图文对)
    数据格式 (LCS-558K):
    {
        "id": "000000033471",
        "image": "images/000000033471.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nProvide a brief description..."},
            {"from": "gpt", "value": "A close-up of a ..."}
        ]
    }
    """

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[PretrainDataset] 加载 {len(self.data)} 条预训练数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])

        # 加载并处理图像
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception as e:
            # 数据容错: 使用空白图像
            print(f"加载图像失败 {image_path}: {e}")
            pixel_values = torch.zeros(3, 384, 384)

        # 处理对话 -> 提取 caption
        conversations = item["conversations"]
        # 构建输入: <image>\n{question}
        # 构建目标: {answer}
        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"]

        # Tokenize
        prompt = human_msg.replace(IMAGE_TOKEN, "").strip()
        full_text = f"{prompt}\n{gpt_msg}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # Labels: 只对 answer 部分计算 loss
        labels = input_ids.clone()
        # 将 padding 部分的 labels 设为 -100
        labels[attention_mask == 0] = -100
        prompt_tokens = self.tokenizer(
            f"{prompt}\n",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[:len(prompt_tokens)] = -100

        # 在 input_ids 开头插入 IMAGE_TOKEN_INDEX
        # 标记图像 token 位置
        input_ids_with_image = torch.cat([
            torch.tensor([IMAGE_TOKEN_INDEX]),
            input_ids[:-1]  # 去掉最后一个 token 保持长度
        ])

        return {
            "input_ids": input_ids_with_image,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataset(Dataset):
    """
    Stage 2: 指令微调数据集 (多轮对话)
    数据格式 (LLaVA-mix-665K):
    {
        "id": "000000033471",
        "image": "coco/train2017/000000033471.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat are the colors..."},
            {"from": "gpt", "value": "The colors of the bus are..."},
            {"from": "human", "value": "What feature can be seen..."},
            {"from": "gpt", "value": "In the image, ..."}
        ]
    }
    """

    SYSTEM_PROMPT = "You are a helpful assistant."
    CONVERSATION_TEMPLATE = {
        "qwen2": "<|im_start|>system\n{system}<|im_end|>\n{conversations}",
        "default": "{system}\n{conversations}",
    }

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048, llm_type: str = "qwen2"):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.llm_type = llm_type

        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[SFTDataset] 加载 {len(self.data)} 条SFT数据")

    def __len__(self):
        return len(self.data)

    def _format_conversation(self, conversations: List[Dict]) -> str:
        """将多轮对话格式化为 Qwen2 的 ChatML 格式"""
        formatted = ""
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"].replace(IMAGE_TOKEN, "").strip()
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # 处理图像
        has_image = "image" in item and item["image"]
        if has_image:
            image_path = os.path.join(self.image_dir, item["image"])
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
            except Exception:
                pixel_values = torch.zeros(3, 384, 384)
                has_image = False
        else:
            pixel_values = torch.zeros(3, 384, 384)

        conversations = item["conversations"]
        # 构建 ChatML 格式文本
        formatted_text = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        formatted_text += self._format_conversation(conversations)

        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # Labels: 只对 assistant 回复计算 loss
        labels = input_ids.clone()
        # 将 padding 部分的 labels 设为 -100
        labels[attention_mask == 0] = -100
        # 将非 assistant 部分设为 -100
        labels = self._mask_non_assistant_tokens(input_ids, labels)

        # 插入图像 token
        if has_image:
            input_ids = torch.cat([
                torch.tensor([IMAGE_TOKEN_INDEX]),
                input_ids[:-1]
            ])

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values if has_image else None,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_non_assistant_tokens(self, input_ids, labels):
        """将非 assistant 回复部分的 labels 设为 -100"""
        # 简化实现: 找 <|im_start|>assistant 和 <|im_end|> 之间的 token
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        in_assistant = False
        for i in range(len(input_ids)):
            if input_ids[i] == im_start_id:
                in_assistant = False
                # 检查后续是否是 "assistant"
                remaining = self.tokenizer.decode(input_ids[i:i+3])
                if "assistant" in remaining:
                    in_assistant = True
                labels[i] = -100
            elif input_ids[i] == im_end_id:
                in_assistant = False
                labels[i] = -100
            elif not in_assistant:
                labels[i] = -100

        return labels


class DPODataset(Dataset):
    """
    Stage 3: DPO 偏好数据集
    数据格式:
    {
        "image": "images/xxx.jpg",
        "question": "Describe what you see...",
        "chosen": "A detailed and accurate description...",
        "rejected": "A hallucinated or inaccurate description..."
    }
    """

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[DPODataset] 加载 {len(self.data)} 条偏好数据")

    def __len__(self):
        return len(self.data)

    def _tokenize(self, question: str, answer: str):
        """构建完整的对话并 tokenize"""
        text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>")
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # 处理图像
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 384, 384)

        question = item["question"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_ids, chosen_mask = self._tokenize(question, chosen)
        rejected_ids, rejected_mask = self._tokenize(question, rejected)

        # 插入图像 token
        chosen_ids = torch.cat([torch.tensor([IMAGE_TOKEN_INDEX]), chosen_ids[:-1]])
        rejected_ids = torch.cat([torch.tensor([IMAGE_TOKEN_INDEX]), rejected_ids[:-1]])

        return {
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


def collate_fn_pretrain(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """预训练数据的 collate 函数"""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def collate_fn_sft(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """SFT 数据的 collate 函数，处理有/无图像混合的 batch"""
    has_images = [b["pixel_values"] is not None for b in batch]

    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

    if any(has_images):
        pixel_values = []
        for b in batch:
            if b["pixel_values"] is not None:
                pixel_values.append(b["pixel_values"])
            else:
                pixel_values.append(torch.zeros(3, 384, 384))
        result["pixel_values"] = torch.stack(pixel_values)
    else:
        result["pixel_values"] = None

    return result


def collate_fn_dpo(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """DPO 数据的 collate 函数"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
        "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
    }


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                     num_workers: int = 4, collate_fn=None) -> DataLoader:
    """构建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
