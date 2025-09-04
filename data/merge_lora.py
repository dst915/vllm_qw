import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. 定义路径 ---
base_model_path = "/home/doust/vllm_qw/Qwen2-1.5B-Instruct"
adapter_path = "/home/doust/vllm_qw/data/qwen2-1.5b-sft-lora-adapters-bf16"
# 定义合并后模型的保存路径 (新的、独立的文件夹)
merged_model_path = "/home/doust/vllm_qw/data/Qwen2-1.5B-Instruct-SFT-eCommerce"
# --- 2. 加载基础模型和分词器 ---
print(f"加载基础模型: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# --- 3. 加载LoRA适配器并与基础模型合并 ---
print(f"加载LoRA适配器: {adapter_path}")
# PeftModel会自动识别adapter_config.json并加载
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
# 执行合并
merged_model = merged_model.merge_and_unload()
print("权重合并完成。")

# --- 4. 保存合并后的完整模型和分词器 ---
print(f"正在保存合并后的模型至: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("模型保存成功！")