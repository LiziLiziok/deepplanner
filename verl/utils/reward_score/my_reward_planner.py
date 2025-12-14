"""
这是最终答案的评分函数
"""

import random
import re
import string
import requests
import json
import ast
import logging
import os
from  . import my_reward_final_answer
prompt_path = "verl/experimental/agent_loop/prompt/planner_reward.md"
logging.basicConfig(level=logging.INFO)

def save_results_to_file(res):
    """Save the results to a file with a counter."""
    # Add a counter to the generation part; here just read the counter
    try:
        # Create output directory
        os.makedirs(os.path.dirname("./outputs/eval_train_plan/"), exist_ok=True)
        
        count_file_path = "./outputs/eval_train_plan/count.txt"
        current_count = 0
        
        # # Handle the counter file, read if it exists (usually it does)
        if os.path.exists(count_file_path):
            with open(count_file_path, 'r', encoding='utf-8') as f:
                count_str = f.read().strip()
                if count_str.isdigit():
                    current_count = int(count_str)
        else:
             # Create a new counter file
            with open(count_file_path, 'w', encoding='utf-8') as f:
                f.write('0')
                logging.info("Created new counter file: count.txt")

        # Save data to the new file
        save_json = res  

        data_source=res["data_source"]

        # Generate the file name with the counter
        json_file_path_test = f"./outputs/eval_train_plan/planner_optim_test_{current_count}.jsonl"
        json_file_path_train = f"./outputs/eval_train_plan/planner_optim_train_{current_count}.jsonl"

        json_line = json.dumps(save_json, ensure_ascii=False)
        
        if data_source.endswith("_val"):
            json_file_path = json_file_path_test
        else:
            json_file_path = json_file_path_train
        # Write to the JSONL file
        with open(json_file_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
            logging.info(f"File saved: {json_file_path}")
        
            
    except (IOError, OSError) as e:
        logging.error(f"File write failed: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")


def call_sglang_server(
    prompt: str,
    host: str = "0.0.0.0",
    port: int = 30000,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list = None,
):
    """调用 SGLang 推理服务获取模型输出
    
    Args:
        prompt: 输入的提示文本
        host: 服务器地址
        port: 服务端口
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus sampling 参数
        stop: 停止词列表
        
    Returns:
        生成的文本内容，如果失败返回 None
    """
    url = f"http://{host}:{port}/v1/completions"
    
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if stop:
        payload["stop"] = stop
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling SGLang server: {e}")
        return None


def call_sglang_chat(
    messages: list,
    host: str = "0.0.0.0",
    port: int = 30000,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list = None,
):
    """调用 SGLang 推理服务的 Chat 接口
    
    Args:
        messages: 对话消息列表，格式为 [{"role": "user", "content": "..."}]
        host: 服务器地址
        port: 服务端口
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus sampling 参数
        stop: 停止词列表
        
    Returns:
        生成的文本内容，如果失败返回 None
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if stop:
        payload["stop"] = stop
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling SGLang chat server: {e}")
        return None


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        pass
    solution_str = solution_str.split('\n')[-1]
    return solution_str.strip()

def _parse_result(res):
    """Parse the result from the solution string."""
    text = res.strip().split('\n')
    text = re.sub(r"```json|```", "", '\n'.join(text)).strip()      

    try:
        data = json.loads(text)
        return data
    except Exception:
        pass     
    try:
        data = ast.literal_eval(text)
        return data
    except Exception:
        return {
  "reasoning": res,
  "score": 0.0
}

def compute_score(solution_str, ground_truth,extra_info,data_source,format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    question = extra_info["question"]
    final_output = extra_info['final_output_AgentLoopOutput']
    all_successful_sub_answers = extra_info['all_successful_sub_answers']
    all_failed_sub_questions = extra_info['all_failed_sub_questions']
    all_attempts = extra_info['all_attempts']
    
    if answer is None:
        res = {
            "question": question,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "reasoning": "Extracted answer is None!",
            "score": 0.0,
            "data_source":data_source,
            "em": 0.0,
            "f1": 0.0,
            "all_successful_sub_answers": all_successful_sub_answers,
            "all_failed_sub_questions": all_failed_sub_questions,
            "all_attempts": all_attempts,

        }
        save_results_to_file(res)
        return 0
    
    
    final_answer = final_output.extra_fields["response"]

    if final_answer is None:
        res = {
            "question": question,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "reasoning": "final_answer is None!",
            "score": 0.0,
            "data_source":data_source,
            "em": 0.0,
            "f1": 0.0,
            "all_successful_sub_answers": all_successful_sub_answers,
            "all_failed_sub_questions": all_failed_sub_questions,
            "all_attempts": all_attempts,
        }
        save_results_to_file(res)
        return 0
    
    em = my_reward_final_answer.em_check(final_answer, ground_truth["target"])
    f1 = my_reward_final_answer.f1_check(final_answer, ground_truth["target"])

    if em==1:
        score = 1.0
        res = {
            "question": question,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "reasoning": "EM check passed!",
            "score": score,
            "data_source":data_source,
            "em": em,
            "f1": f1,
            "all_successful_sub_answers": all_successful_sub_answers,
            "all_failed_sub_questions": all_failed_sub_questions,
            "all_attempts": all_attempts,
        }
    else:
        with open(prompt_path, "r") as f:
            prompt = f.read()
        prompt = prompt.replace("${Golden_Answer}", ground_truth['gt_plan'])
        prompt = prompt.replace("${Response}", answer)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        result = call_sglang_chat(messages)  
        result = _parse_result(result)
        score = result["score"]
        res ={  
            "question": question,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "reasoning": result["reasoning"],
            "score": score,
            "data_source":data_source,
            "em": em,
            "f1": f1,
            "all_successful_sub_answers": all_successful_sub_answers,
            "all_failed_sub_questions": all_failed_sub_questions,
            "all_attempts": all_attempts,
            } 

    save_results_to_file(res)
    
    return score
