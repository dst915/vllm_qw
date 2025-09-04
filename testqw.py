import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import cprint

# --- 使用你当前配置的评委模型路径 ---
JUDGE_MODEL_PATH = "/disk/doust/models/Qwen2-7B-Instruct" 

print(f"--- 正在加载模型: {JUDGE_MODEL_PATH} ---")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_PATH, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    cprint("✅ 模型加载成功！", 'green')

    # --- 准备一个最简单的对话 ---
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好,你叫什么名字"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # --- 生成回答 ---
    cprint("--- 正在生成回答... ---", 'yellow')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # --- 打印结果 ---
    cprint("\n--- 模型的回答是: ---", 'cyan')
    if response.strip():
        cprint(response, 'light_green')
    else:
        cprint("!!! 模型返回了空字符串 !!!", 'red')

except Exception as e:
    cprint(f"❌ 在测试过程中发生严重错误: {e}", 'red')
    import traceback
    traceback.print_exc()