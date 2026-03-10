"""
这是最终答案的评分函数
"""
import os
import logging
import json
import random
import re
import string
logging.basicConfig(level=logging.INFO)
_counter_incremented = False
_current_run_count = None
def save_results_to_file(res):
    """Save the results to a file with a counter."""
    # Add a counter to the generation part; here just read the counter
    try:
        # Create output directory
        os.makedirs(os.path.dirname("./outputs/eval/"), exist_ok=True)
        
        count_file_path = "./outputs/eval/count.txt"
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
        # Generate the file name with the counter
        json_file_path_test = f"./outputs/eval/final_optim_test_{current_count}.jsonl"
        json_file_path_train = f"./outputs/eval/final_optim_train_{current_count}.jsonl"

        json_line = json.dumps(save_json, ensure_ascii=False)
        data_source = res["data_source"]
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


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def f1_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    def compute_f1_score(prediction_tokens, golden_tokens):
        common = set(prediction_tokens) & set(golden_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(golden_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    normalized_prediction = normalize_answer(prediction).split()
    normalized_prediction = normalized_prediction
    max_f1 = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer).split()
        golden_answer = golden_answer
        f1 = compute_f1_score(normalized_prediction, golden_answer)
        # print(f1)
        if f1 > max_f1:
            max_f1 = f1
    
    return max_f1

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



def compute_score(solution_str, ground_truth, extra_info,data_source,format_score=0.0, score=1.0):
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

    if answer is None:
        return {"f1": 0.0, "em": 0}
    
    em = em_check(answer, ground_truth["target"])
    f1 = f1_check(answer, ground_truth["target"])
    output_planner = extra_info.get("output_planner_AgentLoopOutput")
    stage = output_planner.extra_fields["stage"]  # "planner"
    sub_questions = output_planner.extra_fields["sub_questions"]
    raw_planner_response = output_planner.extra_fields["raw_planner_response"]
    replan_attempt_num = output_planner.extra_fields["replan_attempt_num"]
    question = extra_info.get("question", "")
    all_successful_sub_answers = extra_info.get("all_successful_sub_answers", [])
    all_failed_sub_questions = extra_info.get("all_failed_sub_questions", [])
    all_attempts = extra_info.get("all_attempts", [])

    res = {
            "question": question,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "f1": f1,
            "em": em,
            "all_successful_sub_answers": all_successful_sub_answers,
            "final_failed_sub_questions": all_failed_sub_questions,
            "data_source":data_source,
            "replan_attempt_num": replan_attempt_num,
            "all_attempts": all_attempts,
        } 

    save_results_to_file(res)

    return em

# def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
#     """The scoring function for substring exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1

#     if do_print:
#         print("--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")

#     if answer is None:
#         return 0
#     else:
#         if subem_check(answer, ground_truth["target"]):
#             return score
#         else:
#             return format_score
