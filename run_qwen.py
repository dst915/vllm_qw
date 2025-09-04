from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- 1. 定义【本地模型路径】（核心修改点）---
# 替换为 hfd 工具下载的本地文件夹路径，即 "/home/doust/vllm_qw/Qwen2-1.5B-Instruct"
model_local_path = "/home/doust/vllm_qw/Qwen2-1.5B-Instruct"

# --- 2. 初始化 vLLM 引擎（从本地加载模型）---
# 关键：将 model 参数改为本地路径，而非 Hugging Face 模型名
print("正在加载模型...")
llm = LLM(
    model=model_local_path,  # 核心修改：指向本地模型文件夹
    trust_remote_code=True,  # Qwen 模型需自定义代码，保持不变
    tensor_parallel_size=1,  # 单 GPU 运行，保持不变
    # 可选优化：若显存紧张，可添加以下参数（1.5B 模型通常无需）
    # dtype="float16",  # 用 float16 精度（比 bfloat16 更省显存）
    # gpu_memory_utilization=0.8  # 限制 GPU 显存占用比例
)
print("模型加载完毕。")

# --- 3. 加载本地分词器（核心修改点）---
# 同样将 pretrained_model_name_or_path 改为本地路径，避免线上下载
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_local_path,  # 核心修改：指向本地模型文件夹
    trust_remote_code=True  # 匹配 Qwen 分词器的自定义逻辑，必须添加
)

# --- 4. 定义采样参数（保持不变，控制生成行为）---
sampling_params = SamplingParams(
    temperature=0.7,  # 随机性：0 接近确定性，1 随机性高
    top_p=0.95,       # 核心词汇范围：过滤概率累计低于 0.95 的词
    max_tokens=256,   # 单次生成最大长度，避免输出过长
    stop_token_ids=[tokenizer.eos_token_id]  # 可选：明确终止符（匹配 Qwen 分词器）
)

# --- 5. 准备输入（保持不变，遵循 Qwen 对话模板）---
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请用一句话介绍一下自己。"}
]

# 用本地分词器构建 prompt（格式正确，无需修改）
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,        # 不直接转为 token，返回字符串格式
    add_generation_prompt=True  # 自动添加“助手回复”的前缀（Qwen 必需）
)

print("\n--- 输入 Prompt ---")
print(prompt)

# --- 6. 执行推理（保持不变，vLLM 批量生成接口）---
print("\n--- 模型生成中... ---")
outputs = llm.generate([prompt], sampling_params)  # 传入本地构建的 prompt
print("生成完毕。")

# --- 7. 打印结果（保持不变，提取生成文本）---
for output in outputs:
    generated_text = output.outputs[0].text
    print("\n--- 模型输出 ---")
    # 可选：清理输出（去除可能的冗余终止符，Qwen 通常无需）
    # generated_text = generated_text.strip().rstrip(tokenizer.decode(tokenizer.eos_token_id))
    print(f"生成的文本: {generated_text}")