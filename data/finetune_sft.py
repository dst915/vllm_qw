import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- 1. Define Paths ---
base_model_path = "/home/doust/vllm_qw/Qwen2-1.5B-Instruct"
# MODIFIED: 路径现在指向我们新生成的、经过转换的数据集文件。
dataset_path = "./sft_data.jsonl" 
new_adapter_path = "./qwen2-1.5b-sft-lora-adapters-bf16" # 建议改个名以区分

# --- 2. Load Model and Tokenizer (bfloat16 for efficiency) ---
print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# REASON: 确保pad_token被设置，这对于SFTTrainer的批处理是必需的。
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"

# --- 3. Define Formatting Function ---
# REASON: SFTTrainer会遍历数据集的每一行，并将该行（一个包含'messages'键的字典）
# 传递给这个函数。这个函数的作用就是从这一行中抽取出 'messages' 列表，
# 然后使用tokenizer的聊天模板将其转换成一个完整的、可供模型训练的字符串。
def formatting_prompts_func(example):
    """
    Processes a batch of examples from the dataset.
    The data is already in the `messages` format, so we just apply the template.
    SFTTrainer expects a list of strings, so we process each item in the batch.
    """
    output_texts = []
    for messages in example['messages']:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False # 因为'assistant'角色已存在，所以不需要添加生成提示
        )
        output_texts.append(text)
        
    return output_texts

# --- 4. Configure LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# --- 5. Configure Training Arguments ---
training_arguments = TrainingArguments(
    output_dir=os.path.join(new_adapter_path, "training_checkpoints"),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=10,
    # MODIFIED: 学习率大幅降低！
    # REASON: 2e-4对于微调来说太高了，很容易破坏模型的预训练知识，导致“灾难性遗忘”。
    # 2e-5是一个更安全、更稳定的起点，有助于模型在保留原有能力的基础上学习新格式。
    learning_rate=2e-5,
    bf16=True,
    num_train_epochs=1,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    ddp_find_unused_parameters=False,
)

# --- 6. Load Dataset ---
# REASON: 加载我们新创建的、格式正确的数据集。
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 7. Initialize SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # 告诉SFTTrainer使用我们的函数来格式化数据
    max_seq_length=1024, # REASON: 增加了长度，因为系统指令和JSON输出可能很长
    tokenizer=tokenizer,
    args=training_arguments,
)

# --- 8. Start Training & Save ---
print("Starting SFT fine-tuning with JSON as target format...")
trainer.train()

print(f"Training complete. Adapter weights saved to: '{new_adapter_path}'")
trainer.save_model(new_adapter_path)