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
    """
    MODIFIED: This version is now robust to conversations with consecutive
    user or assistant turns by using an index to track progress.
    """
    def __init__(self, scenarios, tokenizer):
        self.scenarios = scenarios
        self.tokenizer = tokenizer
        # 这些变量在 reset() 中被初始化
        self.current_trace = None
        self.history = None
        self.turn_index = 0

    def reset(self):
        """
        随机选择一个场景，并找到第一个assistant回合的位置来初始化环境。
        """
        self.current_trace = random.choice(self.scenarios)
        self.history = []
        
        # 寻找第一个 assistant 回合的索引
        first_assistant_idx = -1
        for i, msg in enumerate(self.current_trace['messages']):
            if msg['role'] == 'assistant':
                first_assistant_idx = i
                break
        
        # 如果找不到assistant回合（数据格式问题），则重新随机选择
        if first_assistant_idx == -1:
            return self.reset()

        # 初始化历史记录，包含第一个assistant回合之前的所有内容
        self.history = self.current_trace['messages'][:first_assistant_idx]
        self.turn_index = first_assistant_idx
        
        return self.history

    def step(self, generated_json_str: str):
        """
        根据模型生成的内容推进对话，并计算奖励。
        """
        # 获取当前指针位置的“黄金标准答案”
        golden_assistant_message = self.current_trace['messages'][self.turn_index]
        golden_json_str = golden_assistant_message['content']
        
        # 计算奖励
        reward = self.get_reward(generated_json_str, golden_json_str)
        
        # 将模型的生成结果（实际行为）加入历史
        self.history.append({"role": "assistant", "content": generated_json_str})
        
        # --- 核心修改逻辑：智能地寻找下一个用户回合 ---
        # 从当前标准答案之后开始，寻找下一个assistant回合
        next_assistant_idx = -1
        for i in range(self.turn_index + 1, len(self.current_trace['messages'])):
            message = self.current_trace['messages'][i]
            # 把所有遇到的user回合都加入历史
            if message['role'] == 'user':
                self.history.append(message)
            # 找到第一个assistant回合就停止
            elif message['role'] == 'assistant':
                next_assistant_idx = i
                break
        
        # 如果找到了下一个assistant回合
        if next_assistant_idx != -1:
            self.turn_index = next_assistant_idx
            done = False
        else:
            # 如果没找到，说明对话结束
            done = True
            
        return self.history, reward, done

    def get_reward(self, generated_str: str, golden_str: str):
        """
        奖励函数，增加了对黄金标准数据格式的健壮性检查。
        """
        try:
            golden_actions = json.loads(golden_str)
            if not isinstance(golden_actions, list):
                golden_actions = [golden_actions]
        except (json.JSONDecodeError, TypeError):
             # 这是一个保护措施，如果黄金数据本身有问题，就跳过这个样本
            print(f"Warning: Skipping a scenario due to malformed golden_str: {golden_str}")
            return 0.0, True # 返回0奖励并结束

        try:
            gen_actions = json.loads(generated_str)
            if not isinstance(gen_actions, list):
                gen_actions = [gen_actions]
        except json.JSONDecodeError:
            return -2.0

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