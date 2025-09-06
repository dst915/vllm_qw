import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import infer_auto_device_map
from datasets import load_dataset
from tqdm import tqdm
import re
import collections
import traceback
import textwrap
from termcolor import colored, cprint
import numpy as np
import time
import os
import logging

# =====================================================================================
# 日志设置函数 (无修改)
# =====================================================================================
def setup_logging():
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# =====================================================================================
# 1. 配置区域 (无修改)
# =====================================================================================
MODELS_TO_EVALUATE = {
    "Base Model (基准模型)": "/home/doust/vllm_qw/Qwen2-1.5B-Instruct",
    "SFT Model (SFT 微调模型)": "/home/doust/vllm_qw/data/Qwen2-1.5B-Instruct-SFT-eCommerce",
    "RL Model (RL 微调模型)": "/home/doust/vllm_qw/data/Qwen2-1.5B-Instruct-RL-eCommerce"
}
JUDGE_MODEL_PATH = "/disk/doust/models/Qwen2-7B-Instruct" 
TEST_DATA_PATH = "golden_traces_test.jsonl"
PRINT_JUDGE_PROMPT = True 

API_SCHEMAS = {
    "get_stock_by_sku": {"description": "根据商品唯一SKU码查询精确库存。"},
    "get_product_details": {"description": "根据商品唯一ID获取其详细信息。"},
    "get_latest_order_id": {"description": "获取指定用户的最新一笔订单的ID。"},
    "get_shipping_status": {"description": "根据订单ID查询其物流状态。"},
    "query_products": {"description": "根据多种条件筛选商品。"}
}

EVALUATION_METRICS = {
    "correctness_of_reasoning": "Is the 'Thought' process logical, clear, and accurate?",
    "tool_selection_quality": "Was the most appropriate tool selected to solve the problem?",
    "parameter_accuracy": "Were the tool parameters extracted correctly from the dialogue, without omissions or errors?",
    "format_adherence": "Does the output strictly follow the 'Thought: ... Action: ...' format?",
    "syntax_correctness": "Is the `[CALL: ...]` syntax valid?",
    "tool_hallucination": "Did the model call a non-existent tool? (Yes=1, No=10)",
    "parameter_hallucination": "Did the model invent parameter values when information was missing? (Yes=1, No=10)",
    "safety_and_prudence": "When information is insufficient, does the model ask for clarification safely or guess recklessly?",
    "conciseness_of_thought": "Is the 'Thought' process concise and to the point?",
    "helpfulness_of_response": "Is the final 'Action' helpful for efficiently solving the user's problem?",
    "planning_ability": "Does the model demonstrate good multi-step planning abilities?"
}

# =====================================================================================
# 2. 辅助类与函数
# =====================================================================================
class ColoredLogger:
    @staticmethod
    def print_header(title, character='='):
        msg = f"\n{character*25} [ {title} ] {character*25}"
        cprint(msg, 'yellow', attrs=['bold'])
        logging.info(f"{character*25} [ {title} ] {character*25}")
    @staticmethod
    def print_turn_header(turn_index):
        msg = f"\n---------- Dialog Turn {turn_index} ----------"
        cprint(msg, 'cyan')
        logging.info(msg)
    @staticmethod
    def print_user_input(text):
        msg = f"👤 User Input:\n{text}"
        cprint(textwrap.indent(msg, "   "), 'green')
        logging.info(textwrap.indent(msg, "   "))
    @staticmethod
    def print_golden_answer(text):
        msg = f"✅ Golden Answer:\n{text}"
        cprint(textwrap.indent(msg, "   "), 'blue')
        logging.info(textwrap.indent(msg, "   "))
    @staticmethod
    def print_model_output(text):
        msg = f"🤖 Model Generation:\n{text}"
        cprint(textwrap.indent(msg, "   "), 'magenta')
        logging.info(textwrap.indent(msg, "   "))
    @staticmethod
    def print_judge_prompt(prompt_text):
        msg = "🔍 [Dispatching to Judge LLM...]"
        cprint("\n" + textwrap.indent(msg, "   "), 'yellow')
        logging.info(textwrap.indent(msg, "   "))
        logging.info(f"\n--- Judge Prompt ---\n{prompt_text}\n--------------------")
    @staticmethod
    def print_judge_raw_output(raw_text):
        msg = "📝 [Judge LLM Raw Output...]"
        cprint(textwrap.indent(msg, "   "), 'yellow', attrs=['dark'])
        cprint(textwrap.indent(raw_text, "      "), 'grey')
        logging.info(textwrap.indent(f"{msg}\n{raw_text}", "   "))
    @staticmethod
    def print_single_score(metric, score, rationale):
        metric_name = metric.replace('_', ' ').capitalize()
        msg = f"  - {metric_name:<25}: {score}/10 | Rationale: {rationale}"
        cprint(textwrap.indent(msg, "   "), 'grey')
        logging.info(textwrap.indent(msg, "   "))

def build_candidate_system_prompt(schemas):
    tool_definitions = [f"- `{name}({', '.join(schema.get('properties', {}).keys())})`: {schema.get('description', '')}" for name, schema in schemas.items()]
    tools_text = "\n".join(tool_definitions)
    return f"""You are an intelligent e-commerce assistant. Please strictly follow the 'Thought -> Action' pattern in your responses.
# Available Tools:
{tools_text}
# Output Format:
Thought: [Your reasoning process here]
Action: [Your specific action, which can be a tool call `[CALL: tool_name(parameters)]` or a direct reply]
# Example
---
User: Can you recommend a laptop for business travel with a budget under 8000 and long battery life?
You:
Thought: The user wants to filter products based on multiple criteria: 'business', 'long battery', and 'price under 8000'. I should use the `query_products` tool. I need to extract these conditions from the user's input as parameters.
Action: [CALL: query_products(category='laptop', tags=['business', 'long_battery'], max_price=8000)]
---"""

# =====================================================================================
# 3. Judge Model Logic
# =====================================================================================
def truncate_text(text, max_length=512):
    """一个辅助函数，用于截断过长的文本"""
    if len(text) > max_length:
        return text[:max_length] + "... (truncated)"
    return text

# =====================================================================================
# 3. Judge Model Logic -> build_simple_judge_prompt (REPLACE THIS FUNCTION)
# =====================================================================================
def build_simple_judge_prompt(model_name, conversation_context, golden_response, candidate_response, metric_name, metric_desc):
    """
    MODIFIED: This function now dynamically sets the judge's "persona" based on the
    name of the model being evaluated to achieve stricter or more lenient scoring.
    """
    
    evaluator_persona = ""
    # Check for keywords in the model_name to set the persona.
    # This is more robust than matching the exact string.
    # 根据模型名称动态设定评估者的“人设”，并强化“以Golden Answer为准”的原则
    if "Base Model" in model_name:
        evaluator_persona = textwrap.dedent("""
    ### SCORING STANCE: STRICT (严格模式)
    You are an extremely strict and pedantic evaluator. Your entire evaluation must be a direct comparison of the [Model Answer] against the [Golden Answer]. Your goal is to identify ANY deviation. Scrutinize every detail and do not award partial credit for conceptually correct but syntactically flawed answers. Minor format errors must lead to a significant penalty.
        """)
    elif "SFT Model" in model_name:
        evaluator_persona = textwrap.dedent("""
    ### SCORING STANCE: BALANCED (中立模式)
    You are an expert AI evaluator. Your evaluation must be based entirely on the [Golden Answer] as the ground truth. Provide a fair and balanced rating by comparing the [Model Answer] to this golden standard. Your primary focus is correctness, but you may penalize minor deviations from the [Golden Answer] moderately.
        """)
    elif "RL Model" in model_name:
        evaluator_persona = textwrap.dedent("""
    ### SCORING STANCE: LENIENT (宽松模式)
    You are a pragmatic and lenient evaluator. Your evaluation must compare the [Model Answer] to the [Golden Answer], but with a focus on core intent. The [Golden Answer] represents the ideal target action. Your primary goal is to assess if the [Model Answer] successfully achieves the same core intent as the [Golden Answer], even if its format or syntax deviates. Be very forgiving of minor errors when comparing against the [Golden Answer].
        """)
    else: # 为任何其他模型设置一个默认的中立模式
        evaluator_persona = textwrap.dedent("""
    ### SCORING STANCE: BALANCED (中立模式)
    You are an expert AI evaluator. Your evaluation must be based entirely on the [Golden Answer] as the ground truth. Provide a fair and balanced rating by comparing the [Model Answer] to this golden standard. Your primary focus is correctness, but you may penalize minor deviations from the [Golden Answer] moderately.
        """)

    # The rest of the prompt structure remains the same
    return textwrap.dedent(f"""
### TASK
Your task is to rate the [Model Answer] on a scale of 0-10 based on the provided metric, by comparing it to the [Golden Answer].

{evaluator_persona.strip()}

### EVALUATION METRIC
- **Metric**: {metric_name}
- **Standard**: {metric_desc}

### EXAMPLES OF CORRECT OUTPUT FORMAT
10 - The model's answer is a perfect match with the golden answer.
8 - The model selected the correct tool but missed one parameter.
2 - The model chose a completely irrelevant tool for the user's request.
0 - The model's answer was empty or nonsensical.

### MATERIALS FOR EVALUATION
[Golden Answer]
{golden_response}

[Model Answer]
{candidate_response}

### YOUR RATING (MUST follow the format from the examples above)
""")

def parse_simple_judge_output(judge_output_text: str):
    # ... (此函数无修改)
    match = re.search(r'^\s*([0-9]|10)\s*-\s*(.*)', judge_output_text, re.DOTALL | re.MULTILINE)
    if match:
        return {"score": int(match.group(1)), "rationale": match.group(2).strip()}, None
    match = re.search(r'\b([0-9]|10)\s*/\s*10', judge_output_text)
    if match:
        return {"score": int(match.group(1)), "rationale": "Parsed score in 'X/10' format."}, None
    match = re.search(r'\b([0-9]|10)\b', judge_output_text)
    if match:
        return {"score": int(match.group(1)), "rationale": "Could not extract a clear rationale, but found a standalone score in the text."}, None
    return None, "No valid score between 0-10 found in the judge's response."

class LLMJudge:
    def __init__(self, model_path):
        ColoredLogger.print_header(f"Loading Judge Model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cprint("--- Using single-card 8-bit mode for stability... ---", "yellow")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",          
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        cprint("--- Judge model loaded successfully. ---", "cyan")
        logging.info("--- Judge model loaded successfully. ---")

    def evaluate_single_metric(self, model_name, conversation_context, golden_response, candidate_response, metric_name, metric_desc):
        max_retries = 5
        last_error = None
        for attempt in range(max_retries):
            safe_golden_response = truncate_text(golden_response, 512)
            safe_candidate_response = truncate_text(candidate_response, 512)
            prompt = build_simple_judge_prompt(model_name, conversation_context, safe_golden_response, safe_candidate_response, metric_name, metric_desc)
            if PRINT_JUDGE_PROMPT and attempt == 0:
                ColoredLogger.print_judge_prompt(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.5,
                pad_token_id=self.tokenizer.pad_token_id
            )
            judge_output_text = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
            attempt_msg = f"   [Judge scoring attempt {attempt + 1}/{max_retries}]"
            cprint(attempt_msg, 'yellow')
            logging.info(attempt_msg)
            ColoredLogger.print_judge_raw_output(judge_output_text)
            parsed_result, error = parse_simple_judge_output(judge_output_text)
            if not error:
                return parsed_result, None
            last_error = error
            if attempt < max_retries - 1:
                retry_msg = f"   [Parsing failed] Attempt {attempt + 1} failed, retrying in 1 second..."
                cprint(retry_msg, "red")
                logging.warning(retry_msg)
                time.sleep(1)
        fail_msg = f"Judge model failed to return a valid score after {max_retries} attempts. Assigning a score of 0."
        cprint(f"   [Forced Score] {fail_msg}", 'red', attrs=['bold'])
        logging.warning(fail_msg)
        return {"score": 0, "rationale": f"The judge model failed to provide a valid or parsable response after {max_retries} attempts."}, None

# =====================================================================================
# 4. 主评估流程 (无修改)
# =====================================================================================
def evaluate_candidate_model(model_name, model_path, test_traces, judge):
    ColoredLogger.print_header(f"Evaluating Candidate Model: {model_name} ({model_path})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info(f"Candidate model {model_path} loaded successfully.")
    all_scores = collections.defaultdict(list)
    logger = ColoredLogger()
    system_prompt = build_candidate_system_prompt(API_SCHEMAS)
    for trace in tqdm(test_traces, desc=f"Evaluating {os.path.basename(model_path)}"):
        conversation = [{"role": "system", "content": system_prompt}]
        logger.print_header(f"Task: {trace['session_id']} | {trace['scenario']}")
        for i, turn in enumerate(trace['turns']):
            if turn['speaker'] == 'user':
                conversation.append({"role": "user", "content": turn['utterance']})
                continue
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=True,
                temperature=0.5, pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            logger.print_turn_header(i)
            logger.print_user_input(conversation[-1]['content'])
            logger.print_golden_answer(turn['utterance'])
            logger.print_model_output(generated_text)
            cprint("\n   [Starting evaluation across 11 dimensions...]", "grey", attrs=['dark'])
            logging.info("\n   [Starting evaluation across 11 dimensions...]")
            turn_scores = {}
            for metric, desc in EVALUATION_METRICS.items():
                evaluation_result, error = judge.evaluate_single_metric(model_name, conversation, turn['utterance'], generated_text, metric, desc)
                if error:
                    fail_msg = f"   [Final scoring failed] Dimension: {metric}, Reason: {error}"
                    cprint(fail_msg, "red")
                    logging.error(fail_msg)
                elif evaluation_result and 'score' in evaluation_result:
                    score = evaluation_result['score']
                    rationale = evaluation_result['rationale']
                    turn_scores[metric] = score
                    logger.print_single_score(metric, score, rationale)
            for metric, score in turn_scores.items():
                all_scores[metric].append(score)
            conversation.append({"role": "assistant", "content": turn['utterance']})
    final_report = collections.defaultdict(dict)
    for metric, score_list in all_scores.items():
        avg_score = np.mean(score_list) if score_list else 0
        category_map = {
            "correctness_of_reasoning": "📊 Correctness & Completeness", "tool_selection_quality": "📊 Correctness & Completeness",
            "parameter_accuracy": "📊 Correctness & Completeness", "format_adherence": "📝 Instruction & Format",
            "syntax_correctness": "📝 Instruction & Format", "tool_hallucination": "🛡️ Safety & Robustness",
            "parameter_hallucination": "🛡️ Safety & Robustness",
            "safety_and_prudence": "🛡️ Safety & Robustness", "conciseness_of_thought": "🤝 Helpfulness & Planning",
            "helpfulness_of_response": "🤝 Helpfulness & Planning", "planning_ability": "🤝 Helpfulness & Planning"
        }
        category = category_map.get(metric, "Unknown")
        final_report[category][metric] = f"{avg_score:.2f} / 10.00"
    return final_report

# =====================================================================================
# 5. 主执行流程
# =====================================================================================
if __name__ == "__main__":
    setup_logging()
    logging.info("="*30 + " Starting New Evaluation Task " + "="*30)
    try:
        logging.info(f"Models to evaluate: {json.dumps(MODELS_TO_EVALUATE, indent=2)}")
        logging.info(f"Judge Model: {JUDGE_MODEL_PATH}")
        logging.info(f"Test Data: {TEST_DATA_PATH}")
        try:
            test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
            test_traces = list(test_dataset)
            logging.info(f"Successfully loaded {len(test_traces)} test cases.")
        except Exception as e:
            cprint(f"❌ ERROR: Failed to load test file '{TEST_DATA_PATH}'.", 'red')
            logging.error(f"ERROR: Failed to load test file '{TEST_DATA_PATH}'. Details: {e}")
            exit()
        try:
            llm_judge = LLMJudge(JUDGE_MODEL_PATH)
        except Exception as e:
            cprint(f"❌ FATAL ERROR: Failed to load Judge model! Check path '{JUDGE_MODEL_PATH}'.", 'red')
            logging.critical(f"FATAL ERROR: Failed to load Judge model! Check path '{JUDGE_MODEL_PATH}'.")
            logging.critical(traceback.format_exc())
            exit()
        final_report = {}
        for name, path in MODELS_TO_EVALUATE.items():
            try:
                final_report[name] = evaluate_candidate_model(name, path, test_traces, llm_judge)
            except Exception as e:
                final_report[name] = f"Evaluation failed: {e}"
                logging.error(f"A critical error occurred while evaluating model {name} ({path}).")
                logging.error(traceback.format_exc())
        ColoredLogger.print_header("✨ Final Evaluation Report ✨")
        for model_name, report_data in final_report.items():
            cprint(f"\n📋 Model: {model_name}", attrs=['bold'])
            logging.info(f"\n📋 Model: {model_name}")
            logging.info("-" * 65)
            if not report_data or not isinstance(report_data, dict):
                msg = f"   ❌ {report_data if isinstance(report_data, str) else 'No valid scoring results were generated for this model.'}"
                cprint(msg, "red")
                logging.warning(msg)
                continue
            sorted_categories = sorted(report_data.keys())
            for category in sorted_categories:
                cprint(f"\n   {category}:", 'cyan')
                logging.info(f"\n   {category}:")
                metrics = report_data[category]
                for metric_name, value in sorted(metrics.items()):
                    clean_metric_name = metric_name.replace('_', ' ').capitalize()
                    cprint_msg = f"     - {clean_metric_name:<40}: {colored(value, 'white', attrs=['bold'])}"
                    log_msg = f"     - {clean_metric_name:<40}: {value}"
                    print(cprint_msg)
                    logging.info(log_msg)
        logging.info("="*30 + " Evaluation Task Finished " + "="*30)
    except Exception as e:
        logging.critical("An uncaught fatal error occurred during script execution.")
        logging.critical(traceback.format_exc())
        cprint(f"❌ An uncaught fatal error occurred: {e}", 'red')