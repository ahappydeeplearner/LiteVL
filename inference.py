"""
模型推理脚本
支持单图推理和批量推理
"""
import os
import argparse
import torch
from PIL import Image

from configs.base_config import ModelConfig
from models.litevl import LiteVL, IMAGE_TOKEN_INDEX


def load_model(model_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    import json

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        config = ModelConfig(**{k: v for k, v in config_dict.items()
                                if hasattr(ModelConfig, k)})
    else:
        config = ModelConfig()

    model = LiteVL(config)

    # 加载 projector
    projector_path = os.path.join(model_path, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(
            torch.load(projector_path, map_location="cpu"))

    # 加载 LLM (可能含 LoRA)
    llm_path = os.path.join(model_path, "llm")
    if os.path.exists(llm_path):
        from peft import PeftModel
        try:
            model.llm = PeftModel.from_pretrained(model.llm, llm_path)
        except Exception:
            pass  # 非 LoRA 模型

    model.to(device)
    model.eval()
    return model


def chat(model: LiteVL, image_path: str, question: str,
         max_new_tokens: int = 512, temperature: float = 0.7):
    """单轮对话推理"""
    device = next(model.parameters()).device

    # 处理图像
    image = Image.open(image_path).convert("RGB")
    image_processor = model.vision_encoder.get_image_processor()
    pixel_values = image_processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)

    # 构建对话
    prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")

    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # 在开头插入图像 token
    image_token = torch.tensor([[IMAGE_TOKEN_INDEX]], device=device)
    input_ids = torch.cat([image_token, input_ids], dim=1)

    attention_mask = torch.ones_like(input_ids)

    # 生成
    output_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="LiteVL Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--image", type=str, required=True,
                        help="图像路径")
    parser.add_argument("--question", type=str, default="Describe this image.",
                        help="问题")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("正在加载模型...")
    model = load_model(args.model_path)
    print("模型加载完成!")

    response = chat(model, args.image, args.question,
                    args.max_tokens, args.temperature)
    print(f"\nQ: {args.question}")
    print(f"A: {response}")


if __name__ == "__main__":
    main()
