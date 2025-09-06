import json

# 理由: 这是我们为模型设计的“系统指令”或“元提示”。
# 它在每个训练样本的最开始告诉模型它的角色、可用的工具(API)以及必须遵循的输出格式。
# 这是让模型学会生成稳定、可解析的JSON的关键。
SYSTEM_PROMPT = """You are an intelligent e-commerce assistant. You have access to the following tools to help users.

# Available Tools:
[
    {"name": "get_price_by_name", "description": "Get the price of a product by its name.", "parameters": {"product_name": "string"}},
    {"name": "get_user_coupons", "description": "Get all available coupons for a given user.", "parameters": {"user_id": "string"}},
    {"name": "get_latest_order_id", "description": "Get the latest order ID for a user.", "parameters": {"user_id": "string"}},
    {"name": "get_product_details", "description": "Get the detailed information of a product.", "parameters": {"product_id": "string"}},
    {"name": "query_products", "description": "Query products based on category, tags, and price.", "parameters": {"category": "string", "tags": "list[string]", "max_price": "integer"}},
    {"name": "get_stock_by_sku", "description": "Get the stock quantity of a product by its SKU.", "parameters": {"sku": "string"}},
    {"name": "find_alternatives", "description": "Find alternative products for a given SKU.", "parameters": {"sku": "string"}},
    {"name": "get_shipping_status", "description": "Get the shipping status of an order.", "parameters": {"order_id": "string"}}
]

# Instructions:
Based on the user's latest message, generate a list of JSON objects representing the next actions. Each object must contain 'thought', 'action_type', 'tool_name', and 'parameters'. For parallel tasks, generate a list of multiple JSON objects. If you need to clarify, set 'action_type' to 'CLARIFY'.
"""

def transform_data(original_data_path: str, output_path: str):
    """
    Transforms the original multi-turn conversation data into the 'messages' format
    required by SFTTrainer.
    """
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for session in original_data:
            messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
            
            for turn in session["turns"]:
                if turn["speaker"] == "user":
                    messages.append({"role": "user", "content": turn["utterance"]})
                
                elif turn["speaker"] == "assistant":
                    # 理由: 这是整个转换的核心。我们告诉模型，它的回答(completion)
                    # 不再是那个自定义的'[CALL]...'字符串，而是一个标准的、可解析的JSON字符串。
                    # 这让模型的学习目标变得清晰、简单且更符合其预训练任务。
                    assistant_completion = json.dumps(
                        turn["assistant_actions"], 
                        ensure_ascii=False, 
                        indent=4
                    )
                    
                    # 复制当前的对话历史，并附上助理的目标回答
                    training_sample = list(messages)
                    training_sample.append({"role": "assistant", "content": assistant_completion})
                    
                    # 将这个完整的对话（包含目标回答）写入新文件
                    f_out.write(json.dumps({"messages": training_sample}, ensure_ascii=False) + "\n")
                    
                    # 为了构建下一轮对话的正确历史，将助理的“实际”行为（而不是我们想让他说的）加入历史
                    # 这里简化处理，直接将工具调用结果加入，实际应用中可以设计的更复杂
                    messages.append({"role": "assistant", "content": assistant_completion})

if __name__ == "__main__":
    # 假设你把原始数据保存为 `original_data.json`
    original_file = "golden_traces.jsonl"
    transformed_file = "sft_data.jsonl"
    
    print(f"Transforming '{original_file}' to '{transformed_file}'...")
    transform_data(original_file, transformed_file)
    print("Transformation complete.")