import json

# 系统指令保持不变
SYSTEM_PROMPT = """You are an intelligent e-commerce assistant... (省略，使用你SFT时的完整prompt)"""

def merge_and_transform_data(original_data_path: str, output_path: str):
    """
    MODIFIED: This script now correctly handles consecutive assistant turns
    by merging them into a single turn before creating SFT samples.
    """
    with open(original_data_path, 'r', encoding='utf-8') as f:
        # 假设原始文件是JSONL格式，逐行读取
        original_sessions = [json.loads(line) for line in f if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for session in original_sessions:
            if not session.get("turns"):
                continue

            # --- 核心修改点：合并逻辑 ---
            merged_turns = []
            for turn in session["turns"]:
                # 如果merged_turns不为空，且当前turn和最后一个turn都是assistant，则合并
                if (merged_turns and 
                    turn["speaker"] == "assistant" and 
                    merged_turns[-1]["speaker"] == "assistant"):
                    
                    # 合并 assistant_actions
                    if "assistant_actions" in turn and "assistant_actions" in merged_turns[-1]:
                        merged_turns[-1]["assistant_actions"].extend(turn["assistant_actions"])
                else:
                    merged_turns.append(turn)
            # --- 合并逻辑结束 ---

            # 使用合并后的turns来生成训练样本
            history = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
            for turn in merged_turns:
                if turn["speaker"] == "user":
                    history.append({"role": "user", "content": turn["utterance"]})
                
                elif turn["speaker"] == "assistant":
                    if "assistant_actions" not in turn or not turn["assistant_actions"]:
                        continue # 跳过没有actions的助理回合

                    assistant_completion = json.dumps(
                        turn["assistant_actions"], 
                        ensure_ascii=False, 
                        indent=4
                    )
                    
                    training_sample_messages = list(history)
                    training_sample_messages.append({"role": "assistant", "content": assistant_completion})
                    
                    f_out.write(json.dumps({"messages": training_sample_messages}, ensure_ascii=False) + "\n")
                    
                    # 更新历史以进行下一轮
                    history.append({"role": "assistant", "content": assistant_completion})


if __name__ == "__main__":
    # 假设你的原始数据是 golden_traces.jsonl
    original_file = "golden_traces.jsonl"
    transformed_file = "sft_data.jsonl"
    
    print(f"Transforming and merging '{original_file}' to '{transformed_file}'...")
    merge_and_transform_data(original_file, transformed_file)
    print("Transformation complete.")