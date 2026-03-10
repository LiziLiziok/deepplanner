# \verl\utils\entropy
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import random 
import os
import torch
from verl import DataProto

def split_tag_into_subwords(tokenizer, tag):
    """
    将标签（如 '<answer>'）手动拆分为子词序列。
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): 分词器
        tag (str): 要拆分的标签，如 '<answer>'
        
    Returns:
        List[int]: 子词对应的 token ID 列表
    """
    # 使用 tokenizer.encode 拆分标签为 token IDs
    subword_ids = tokenizer.encode(tag, add_special_tokens=False)
    return subword_ids

def find_subword_sequence(token_ids, subword_ids):
    """
    在 token_ids 中查找子词序列的起始位置。
    
    Args:
        token_ids (torch.Tensor): 形状为 [seq_len] 或 [batch_size, seq_len]
        subword_ids (List[int]): 子词 ID 列表
    
    Returns:
        List[List[int]]: 每个样本中子词序列的起始索引列表
    """
    batch_size, seq_len = token_ids.shape
    subword_len = len(subword_ids)
    start_indices = []

    for i in range(batch_size):
        indices = []
        for j in range(seq_len - subword_len + 1):
            if all(token_ids[i, j + k].item() == subword_ids[k] for k in range(subword_len)):
                indices.append(j)
        start_indices.append(indices)
    
    return start_indices

def find_tag_positions(token_ids, tokenizer, tag="answer"):
    """
    在 token_ids 中查找 <tag> 和 </tag> 的起始位置。
    返回每个样本中 <tag> 的起始索引和 </tag> 的结束索引。
    """
    # 拆分标签为子词
    # 0_get_token_id.py 得到对应的 token
    # start_tag = f"<{tag}>"
    # end_tag = f"</{tag}>"
    # start_subwords = split_tag_into_subwords(tokenizer, start_tag)
    # end_subwords = split_tag_into_subwords(tokenizer, end_tag)
    start_subwords = [27, 9217, 29] # <answer>
    end_subwords = [522, 9217, 29] # </answer>


    # 查找子词序列
    start_indices_list = find_subword_sequence(token_ids, start_subwords)
    end_indices_list = find_subword_sequence(token_ids, end_subwords)

    batch_size = token_ids.shape[0]
    start_indices = []
    end_indices = []

    for i in range(batch_size):
        # 取第一个匹配的起始位置
        start_idx = start_indices_list[i][0] if start_indices_list[i] else None
        end_idx = end_indices_list[i][0] if end_indices_list[i] and start_idx is not None else None

        # 单独处理 start_idx 和 end_idx 的缺失情况
        if start_idx is None:
            start_idx = 0  # 未找到 <answer>
        if end_idx is None:
            end_idx = token_ids.shape[1]  # 未找到 </answer>

        start_indices.append(start_idx)
        end_indices.append(end_idx)
    
    return torch.tensor(start_indices), torch.tensor(end_indices)


def compute_log_to_confidence(padded_answer_log_probs):
    # 防御性检查
    if not isinstance(padded_answer_log_probs, torch.Tensor):
        raise ValueError(f"padded_answer_log_probs 必须是 torch.Tensor，但实际类型是 {type(padded_answer_log_probs)}")

    # 假设填充值为 0，构建 mask
    mask = (padded_answer_log_probs != 0)
    valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # 避免除以 0

    # 计算均值 log_prob
    mean_log_probs = (padded_answer_log_probs * mask).sum(dim=1, keepdim=True) / valid_lengths

    # 转换为置信度（0-1 之间）
    retrival_confidence = torch.exp(mean_log_probs)

    return retrival_confidence

## 加速版 ver1:
def extract_answer_log_probs(log_probs, start_indices, end_indices, offset=3):
    """
    批量提取 log_probs 中 [start + offset : end] 的 token 概率
    :param log_probs: Tensor of shape [batch_size, seq_len]
    :param start_indices: LongTensor of shape [batch_size], 已跳过 ans 标签
    :param end_indices: LongTensor of shape [batch_size]
    :return: List[Tensor], 每个元素为 [answer_token_num, ]
    """
    batch_size, seq_len = log_probs.shape

    # Step 1: 构建一个 batch x seq_len 的 mask，标记哪些位置要保留
    arange_tensor = torch.arange(seq_len, device=log_probs.device).expand(batch_size, -1)

    # Step 2: start + offset
    starts = start_indices + offset
    ends = end_indices

    # Step 3: 创建 mask
    mask = (arange_tensor >= starts.unsqueeze(1)) & (arange_tensor < ends.unsqueeze(1))

    # Step 4: 使用 mask 提取答案部分
    masked_log_probs = log_probs.masked_select(mask).split((ends - starts).tolist())

    return masked_log_probs

### ver2: 加速版
def find_answer_start_backward(token_ids, answer_subwords=[27, 9217, 29], window_size=64):
    """
    从后往前查找 <answer> 子词序列的起始位置。
    
    Args:
        token_ids: [B, L] 输入 token IDs
        answer_subwords: <answer> 的 token ID 序列
        window_size: 只检查最后 window_size 个 token（默认 64）

    Returns:
        start_indices: [B]，表示 <answer> 的起始位置；未找到则为 -1
    """
    B, L = token_ids.shape
    K = len(answer_subwords)
    device = token_ids.device

    # 只检查最后 window_size 个 token
    # start_pos = torch.clamp(L - window_size, min=0)
    start_pos = max(L - window_size, 0)
    search_window = token_ids[:, start_pos:]  # [B, W], W=min(L, window_size)
    W = search_window.size(1)

    # 构造滑动窗口视图：[B, W-K+1, K]
    windows = search_window.unfold(dimension=1, size=K, step=1)

    # 构建 answer_subwords 的广播张量
    answer_tensor = torch.tensor(answer_subwords, device=device).unsqueeze(0).unsqueeze(0).expand(B, W - K + 1, K)

    # 比较每个窗口是否匹配
    matches = (windows == answer_tensor).all(dim=-1)  # shape [B, W-K+1]

    # 找出最后一个匹配的位置（从后往前）
    first_match_in_window = (matches.flip(dims=(1,)).cumsum(dim=1) > 0).int().argmax(dim=1)
    found_mask = matches.flip(dims=(1,)).any(dim=1)

    # 转换为原始 token_ids 中的位置
    raw_positions = (W - K) - first_match_in_window
    global_positions = start_pos + raw_positions
    global_positions[~found_mask] = -1  # 未找到的设为 -1

    return global_positions  # [B]

def extract_first_n_after_answer(token_ids, log_probs, n=5, offset=3, window_size=64):
    start_indices = find_answer_start_backward(token_ids, window_size=window_size)

    result = []
    for i in range(token_ids.shape[0]):
        start = start_indices[i].item()
        if start == -1:
            result.append(torch.tensor([], device=log_probs.device))
        else:
            start += offset  # 跳过 <answer> 自身
            end = min(start + n, token_ids.shape[1])
            result.append(log_probs[i, start:end])

    return result

## 加速版 ver3
def find_answer_positions_backward(token_ids, answer_subwords=[27, 9217, 29], end_subwords=[522, 9217, 29], window_size=128):
    """
    从后往前查找 <answer> 和 </answer> 的起始位置。
    
    Args:
        token_ids: [B, L] 输入 token IDs
        answer_subwords: <answer> 的 token ID 序列
        end_subwords: </answer> 的 token ID 序列
        window_size: 只检查最后 window_size 个 token
    
    Returns:
        start_indices: <answer> 的起始位置
        end_indices: </answer> 的起始位置
    """
    B, L = token_ids.shape
    K_ans = len(answer_subwords)
    K_end = len(end_subwords)
    device = token_ids.device

    # Step 1: 截取最后 window_size 个 token
    start_pos = max(L - window_size, 0)
    search_window = token_ids[:, start_pos:]  # [B, W]
    W = search_window.size(1)

    # Step 2: 构造滑动窗口查找 end_subwords (</answer>)
    if K_end > W:
        end_matches = torch.zeros(B, 0, dtype=torch.bool, device=device)
    else:
        windows_end = search_window.unfold(dimension=1, size=K_end, step=1)
        end_tensor = torch.tensor(end_subwords, device=device).unsqueeze(0).unsqueeze(0).expand(B, W - K_end + 1, K_end)
        end_matches = (windows_end == end_tensor).all(dim=-1)  # [B, W-K_end+1]

    # Step 3: 构造滑动窗口查找 answer_subwords (<answer>)
    if K_ans > W:
        ans_matches = torch.zeros(B, 0, dtype=torch.bool, device=device)
    else:
        windows_ans = search_window.unfold(dimension=1, size=K_ans, step=1)
        ans_tensor = torch.tensor(answer_subwords, device=device).unsqueeze(0).unsqueeze(0).expand(B, W - K_ans + 1, K_ans)
        ans_matches = (windows_ans == ans_tensor).all(dim=-1)  # [B, W-K_ans+1]

    start_indices = []
    end_indices = []

    for i in range(B):
        # Step 4: 找到最后一个 </answer>
        end_pos_in_window = None
        if K_end <= W and end_matches[i].any():
            end_pos_in_window = (W - K_end) - (end_matches[i].flip(dims=(0,)).cumsum(0) > 0).int().argmax().item()

        if end_pos_in_window is not None:
            # Step 5: 在 </answer> 之前找最近的 <answer>
            search_up_to = end_pos_in_window
            candidate_positions = ans_matches[i, :search_up_to]
            if candidate_positions.any():
                ans_pos_in_window = (search_up_to - 1) - (candidate_positions.flip(dims=(0,))[:search_up_to].cumsum(0) > 0).int().argmax().item()
                start_idx = start_pos + ans_pos_in_window
                end_idx = start_pos + end_pos_in_window
                start_indices.append(start_idx)
                end_indices.append(end_idx)
                continue

        # 默认 fallback
        start_indices.append(0)
        end_indices.append(token_ids.shape[1])

    return torch.tensor(start_indices, device=device), torch.tensor(end_indices, device=device)

def compute_retrival_confidence_ver0(data: DataProto, tokenizer):
    # 假设 data.batch['responses'] 和 data.batch['old_log_probs'] 已定义
    token_ids = data.batch['responses']  # [batch_size, seq_len] | 已经是 token 了
    log_probs = data.batch['old_log_probs']  # [batch_size, seq_len]

    # 确保 token_ids 和 log_probs 形状一致
    assert token_ids.shape == log_probs.shape, f"Shape mismatch: {token_ids.shape} vs {log_probs.shape}"

    # 查找标签位置（token ID 序列中的索引）
    start_indices, end_indices = find_tag_positions(token_ids, tokenizer, tag="answer")
    print("start_indices",start_indices)

    # 提取 <answer> 内的 log_probs
        # batch_size = token_ids.shape[0]
        # answer_log_probs = []
        # for i in range(batch_size):
        #     # 确保索引是单元素张量并转换为 Python int
        #     start = start_indices[i].item() + 3  # 跳过 ans 的三个 token
        #     end = end_indices[i].item()

        #     # 检查索引是否越界
        #     if not (0 <= start < token_ids.shape[1] and 0 <= end <= token_ids.shape[1]):
        #         raise ValueError(f"Invalid indices for sample {i}: start={start}, end={end}")
                
        #     if start < end:
        #         answer_log_probs.append(log_probs[i, start:end])
        #     else:
        #         answer_log_probs.append(torch.tensor([], device=log_probs.device))
    answer_log_probs = extract_answer_log_probs(log_probs, start_indices, end_indices, offset=3) # 跳过 ans 的三个 token


    # 使用 pad_sequence 填充变长张量（默认填充到最长序列）
    padded_answer_log_probs = pad_sequence(answer_log_probs, batch_first=True, padding_value=0.0)

    # 计算置信度（0-1 之间）
    retrival_confidence = compute_log_to_confidence(padded_answer_log_probs)

    return retrival_confidence

def compute_retrival_confidence_ver1(data: DataProto, tokenizer):
    # 假设 data.batch['responses'] 和 data.batch['old_log_probs'] 已定义
    token_ids = data.batch['responses']  # [batch_size, seq_len] | 已经是 token 了
    log_probs = data.batch['old_log_probs']  # [batch_size, seq_len]

    # 确保 token_ids 和 log_probs 形状一致
    assert token_ids.shape == log_probs.shape, f"Shape mismatch: {token_ids.shape} vs {log_probs.shape}"

    # 查找标签位置（token ID 序列中的索引）
    # start_indices, end_indices = find_tag_positions(token_ids, tokenizer, tag="answer")
    # print("start_indices",start_indices)

    # 提取 <answer> 内的 log_probs
        # batch_size = token_ids.shape[0]
        # answer_log_probs = []
        # for i in range(batch_size):
        #     # 确保索引是单元素张量并转换为 Python int
        #     start = start_indices[i].item() + 3  # 跳过 ans 的三个 token
        #     end = end_indices[i].item()

        #     # 检查索引是否越界
        #     if not (0 <= start < token_ids.shape[1] and 0 <= end <= token_ids.shape[1]):
        #         raise ValueError(f"Invalid indices for sample {i}: start={start}, end={end}")
                
        #     if start < end:
        #         answer_log_probs.append(log_probs[i, start:end])
        #     else:
        #         answer_log_probs.append(torch.tensor([], device=log_probs.device))
    # answer_log_probs = extract_answer_log_probs(log_probs, start_indices, end_indices, offset=3) # 跳过 ans 的三个 token
    answer_log_probs = extract_first_n_after_answer(token_ids, log_probs, n=5, offset=3) # 跳过 <answer> 的三个 token, 选择<answre>之后的 5 个 token

    # 使用 pad_sequence 填充变长张量（默认填充到最长序列）
    padded_answer_log_probs = pad_sequence(answer_log_probs, batch_first=True, padding_value=0.0)

    # 计算置信度（0-1 之间）
    retrival_confidence = compute_log_to_confidence(padded_answer_log_probs)

    return retrival_confidence

def compute_retrival_confidence(data: DataProto, tokenizer):
    # 假设 data.batch['responses'] 和 data.batch['old_log_probs'] 已定义
    token_ids = data.batch['responses']  # [batch_size, seq_len] | 已经是 token 了
    log_probs = data.batch['old_log_probs']  # [batch_size, seq_len]

    # 确保 token_ids 和 log_probs 形状一致
    assert token_ids.shape == log_probs.shape, f"Shape mismatch: {token_ids.shape} vs {log_probs.shape}"

    # 查找标签位置（token ID 序列中的索引）
    # start_indices, end_indices = find_tag_positions(token_ids, tokenizer, tag="answer")
    start_indices, end_indices = find_answer_positions_backward(token_ids, window_size=128)
    # print("start_indices",start_indices)

    # 提取 <answer> 内的 log_probs
    answer_log_probs = extract_answer_log_probs(log_probs, start_indices, end_indices, offset=3) # 跳过 ans 的三个 token


    # 使用 pad_sequence 填充变长张量（默认填充到最长序列）
    padded_answer_log_probs = pad_sequence(answer_log_probs, batch_first=True, padding_value=0.0)

    # 计算置信度（0-1 之间）
    retrival_confidence = compute_log_to_confidence(padded_answer_log_probs)

    return retrival_confidence
