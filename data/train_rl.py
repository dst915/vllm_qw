import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
from peft import LoraConfig
import os
import re
import json
from tqdm import tqdm
import time
from accelerate import Accelerator
import random

# 环境设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 彩色日志工具 (无修改)
class COLORS:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# =====================================================================================
# 1. 强化学习环境
# =====================================================================================
class EcommerceEnv:
    def __init__(self, scenarios, tokenizer):
        self.scenarios = scenarios
        self.tokenizer = tokenizer
        self.system_prompt = self._build_system_prompt()
        self.current_trace = None
        self.history = []

    def _build_system_prompt(self):
        # 保持和SFT阶段完全一致的系统指令
        return """You are an intelligent e-commerce assistant... (省略，使用你SFT时的完整prompt)"""

    def reset(self):
        # 随机选择一个多轮对话场景
        self.current_trace = random.choice(self.scenarios)
        # 初始化历史记录，包含系统指令和第一轮的用户输入
        self.history = self.current_trace['messages'][:2] 
        return self.history

    def step(self, generated_json_str: str):
        # 如果历史记录长度已经超过了标准答案的长度，说明模型生成了多余的对话，给予惩罚
        if len(self.history) >= len(self.current_trace['messages']):
            return self.history, -2.0, True

        # 获取当前轮次的标准答案
        golden_assistant_message = self.current_trace['messages'][len(self.history)]
        golden_json_str = golden_assistant_message['content']
        
        # 计算奖励
        reward = self.get_reward(generated_json_str, golden_json_str)
        
        # 将模型的生成结果加入历史
        self.history.append({"role": "assistant", "content": generated_json_str})
        
        # 判断对话是否结束
        done = len(self.history) >= len(self.current_trace['messages'])
        
        if not done:
            # 如果没结束，自动加入下一轮用户的发言
            self.history.append(self.current_trace['messages'][len(self.history)])
            
        return self.history, reward, done

    # CRITICAL FIX 1: 重写奖励函数以解析和评估JSON
# 在 EcommerceEnv 类中，替换这个函数

    def get_reward(self, generated_str: str, golden_str: str):
        """
        MODIFIED: Added a robust try-except block to catch and report errors
        in the golden dataset, which helps in debugging the data itself.
        """
        # --- 核心修改点：为golden_str的解析添加保护 ---
        try:
            golden_actions = json.loads(golden_str)
            if not isinstance(golden_actions, list):
                golden_actions = [golden_actions]
        except json.JSONDecodeError:
            print(f"\n{COLORS.FAIL}CRITICAL ERROR: Failed to parse golden_str as JSON.{COLORS.ENDC}")
            # 尝试从 self.current_trace 中获取 session_id 来帮助定位
            # 注意：'messages' 键存在于转换后的数据中，但原始的 session_id 可能需要从原始数据结构中寻找
            # 为简化，我们直接打印出导致问题的字符串
            print(f"{COLORS.FAIL}Problematic golden_str: -->{golden_str}<--{COLORS.ENDC}")
            print(f"{COLORS.FAIL}This indicates a problem in your sft_data_transformed.jsonl file.{COLORS.ENDC}")
            # 抛出异常，中断程序，因为这是一个需要修复的数据问题
            raise
        # --- 修改结束 ---

        try:
            gen_actions = json.loads(generated_str)
            if not isinstance(gen_actions, list):
                gen_actions = [gen_actions]
        except json.JSONDecodeError:
            return -2.0 # 模型生成了无效JSON，给予惩罚

        # 奖励逻辑保持不变
        if gen_actions == golden_actions:
            return 2.0

        gen_tool_names = {action.get('tool_name') for action in gen_actions}
        golden_tool_names = {action.get('tool_name') for action in golden_actions}

        if gen_tool_names != golden_tool_names:
            return -2.0
        
        return -1.0

# =====================================================================================
# 2. 主训练流程 
# =====================================================================================
if __name__ == "__main__":
    accelerator = Accelerator()

    sft_model_path = "./qwen2-1.5b-sft-lora-adapters-bf16" # SFT模型路径
    rl_adapter_path = "./qwen2-1.5b-rl-adapters" # 新的RL模型保存路径
    dataset_path = "./sft_data.jsonl" # SFT时用的数据集

    config = PPOConfig(
        model_name=sft_model_path,
        learning_rate=1.41e-5,
        batch_size=16,
        mini_batch_size=2,
        # SUGGESTION 3: 简化配置，在PPO中通常不使用梯度累积
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_kwargs={"logging_dir": "./rl_logs_v2"},
        kl_penalty="kl",
        adap_kl_ctrl=True,
        init_kl_coef=0.1,
        ppo_epochs=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 对于生成任务，padding在左边是最佳实践
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
        peft_config=lora_config,
        torch_dtype=torch.bfloat16,
    )
    
    
    ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)
    
    scenarios = list(load_dataset("json", data_files=dataset_path, split="train"))
    env = EcommerceEnv(scenarios, tokenizer)

    generation_kwargs = {
        "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 256,
    }

    total_ppo_epochs = 4
    
    if accelerator.is_main_process:
        print(f"\n{COLORS.HEADER}===== 🚀 Starting PPO Training ====={COLORS.ENDC}")
        
    for epoch in range(total_ppo_epochs):
        if accelerator.is_main_process:
            print(f"\n{COLORS.BOLD}Epoch {epoch+1}/{total_ppo_epochs}{COLORS.ENDC}")
            scenario_iterator = tqdm(scenarios, desc="Scenario")
        else:
            scenario_iterator = scenarios

        batch_queries, batch_responses, batch_rewards = [], [], []
        
        for scenario in scenario_iterator:
            history = env.reset()
            done = False
            
            while not done:
                query_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                # SUGGESTION 3: PPOTrainer内部会处理设备，无需手动.to(device)
                query_tensor = tokenizer.encode(query_text, return_tensors="pt").squeeze(0).to(accelerator.device)

                # 生成、解码、计算奖励
                response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
                response_only_tensor = response_tensor.squeeze()[len(query_tensor):]
                # 从完整输出中提取模型新生成的部分
                action_text = tokenizer.decode(response_only_tensor, skip_special_tokens=True)
                
                history, reward_float, done = env.step(action_text.strip())
                
                # 保存数据，准备PPO步骤
                batch_queries.append(query_tensor)
                batch_responses.append(response_tensor.squeeze())
                batch_rewards.append(torch.tensor(reward_float, device=accelerator.device))

                # 当收集到足够数据时，执行PPO优化步骤
                if len(batch_queries) >= config.batch_size:
                    stats = ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
                    ppo_trainer.log_stats(stats, {}, [r.cpu().item() for r in batch_rewards])
                    batch_queries, batch_responses, batch_rewards = [], [], []

    if accelerator.is_main_process:
        print(f"\n{COLORS.HEADER}===== ✅ RL Training Complete. Saving Adapters... ====={COLORS.ENDC}")
        ppo_trainer.save_pretrained(rl_adapter_path)
        print(f"{COLORS.OKGREEN}✅ Adapters saved to: {rl_adapter_path}{COLORS.ENDC}")