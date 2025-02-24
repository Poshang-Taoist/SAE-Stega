import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from utils import kl, entropy, is_sent_finish, limit_past, bits2int, int2bits
from block_baseline import get_bins


def encode_static_position(model, enc, message, context, block_size, device='cuda'):
    """
    将明文信息编码为隐藏文本。

    Args:
        model (torch.nn.Module): 用于生成下一个token的模型。
        enc (torchtext.vocab.Vocab): 用于编码和解码的词汇表。
        message (list of int): 需要编码的二进制消息。
        context (list of int): 用于初始化模型输出的上下文序列。
        block_size (int): 每次编码的二进制位数。
        device (str, optional): 模型运行的设备，默认为'cuda'。

    Returns:
        str: 编码后的隐藏文本。

    """
    context = torch.tensor(context, device=device, dtype=torch.long)
    prev, output, past = context, context, None

    concealed_length = 0
    while True:
        if concealed_length >= len(message):
            break

        logits, past = model(prev.unsqueeze(0), past=past)
        logits = logits[0, -1, :]
        logits, indices = logits.sort(descending=True)

        m_part = message[concealed_length:concealed_length + block_size]
        binary_str = ''.join(map(str, m_part))
        k = int(binary_str, 2)

        # Debug prints
        print(f"Encoding Step - Concealed Length: {concealed_length}")
        print(f"Binary Segment (m_part): {m_part} -> Integer k: {k}")
        print(f"Selected Token ID: {indices[k].item()} - Token: {enc.decode([indices[k].item()])}")

        prev = indices[k].view(1)
        output = torch.cat((output, prev))
        concealed_length += block_size

    covertext = enc.decode(output[len(context):])
    print("Final Covertext:", covertext)
    return covertext


def decode_static_position(model, enc, covertext, context, block_size, device='cuda'):
    # Tokenize the initial context and the covertext
    context = enc.encode(context)
    covertext_tokens = enc.encode(covertext)

    # Set up the initial context tensor and model past state
    context_tensor = torch.tensor(context, device=device, dtype=torch.long)
    prev, past = context_tensor, None  # Start with the full context
    decoded_message = []

    # Process each token in the covertext after the initial context length
    for token in covertext_tokens[len(context):]:
        # Generate logits using the previous context and past state
        logits, past = model(prev.unsqueeze(0), past=past)
        logits = logits[0, -1, :]
        logits, indices = logits.sort(descending=True)

        # Find the position of the current token within the sorted logits indices
        token_position = (indices == token).nonzero(as_tuple=True)[0].item()

        # Convert token position to binary and pad to the specified block size
        binary_segment = bin(token_position)[2:].zfill(block_size)

        # Debug prints
        print(f"Decoding Step - Token Position: {token_position}")
        print(f"Logits Top Indices (Decoding): {indices[:5].tolist()}")  # Print top indices
        print(f"Binary Segment from Position: {binary_segment}")

        # Append each bit from binary_segment to the decoded message
        decoded_message.extend([int(bit) for bit in binary_segment])

        # Update prev with the current token to keep generating
        prev = torch.tensor([token], device=device, dtype=torch.long)

    print("Decoded Message:", decoded_message)
    return decoded_message


def encode_dynamic_block1(model, enc, message, context, block_size, finish_sent=False, device='cuda'):
    """
    将动态块1进行编码。

    Args:
        model (torch.nn.Module): 用于生成编码的模型。
        enc (torch.nn.Module): 用于编码的编码器。
        message (List[int]): 要编码的消息，以二进制表示。
        context (List[int]): 编码上下文，以整数列表表示。
        block_size (int): 每个编码块的长度。
        finish_sent (bool, optional): 是否完成句子的编码。默认为False。
        device (str, optional): 模型运行的设备。默认为'cuda'。

    Returns:
        tuple: 包含编码后的输出和统计信息的元组。
    """
    context = torch.tensor(context, device=device, dtype=torch.long)

    prev, output, past = context, context, None
    total_num, total_num_for_stats, total_log_probs, total_kl = 0, 0, 0, 0

    concealed_length = 0
    while True:
        if concealed_length >= len(message):
            break

        logits, past = model(prev.unsqueeze(0), past=past)
        logits = logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        m_part = message[concealed_length:concealed_length + block_size]
        m_part = ''.join(map(str, m_part))  # "01110"
        m_part = int(m_part, 2)


def encode_dynamic_block3(model, enc, message, context, block_size, finish_sent=False, device='cuda'):
    """
    对动态块进行编码。

    Args:
        model (torch.nn.Module): 使用的语言模型。
        enc (Encoder): 编码器，用于将文本转换为整数编码。
        message (list of int): 需要编码的消息，每个元素是一个整数编码。
        context (list of int): 上下文，每个元素是一个整数编码。
        block_size (int): 动态块的大小。
        finish_sent (bool, optional): 是否在完成一个句子后停止编码。默认为False。
        device (str, optional): 使用的设备，例如'cuda'或'cpu'。默认为'cuda'。

    Returns:
        None

    """
    length = len(message)

    # 截断为最后1022个标记
    # truncated to the last 1022 tokens
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    prev, output, past = context, context, None
    total_num, total_num_for_stats, total_log_probs, total_kl = 0, 0, 0, 0

    concealed_length = 0
    while True:
        if concealed_length >= len(message):
            break

        # 初始化候选列表
        candidates = []
        # 使用模型计算下一个标记的概率分布和状态
        logits, past = model(prev.unsqueeze(0), past=past)
        logits = logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        # 找到最可能的标记
        original_token_id = torch.argmax(logits).item()
        print(f"original_token is: {enc.decode(original_token_id)}")
        print(f"original_token prob is: {log_probs[original_token_id].exp().item()}")
        # 对每个可能的块大小进行处理
        for B in range(1, block_size + 1):
            # 初始化过滤后的对数概率
            filtered_logits = logits.clone()
            filtered_logits[:] = -1e10  # first set all to 0

            # 获取二进制到单词和单词到二进制的映射
            bin2words, words2bin = get_bins(len(enc.encoder), block_size)
            # 获取当前块的消息部分
            m_part = message[:B]
            # 获取可用标记
            available_tokens = bin2words[bits2int(m_part)]

            # 更新过滤后的对数概率
            filtered_logits[available_tokens] = logits[available_tokens]
            # 找到过滤后的最可能标记
            cover_token_id = torch.argmax(filtered_logits).item()
            print(f"cover_token is: {enc.decode(cover_token_id)}")
            log_prob = log_probs[cover_token_id]
            print(f"cover_token prob is: {log_prob.exp().item()}")
            # 添加候选到列表
            candidates.append([B, cover_token_id, log_prob.item()])

        # 定义对数概率阈值
        # Define the log probability threshold
        default = [0, original_token_id, log_probs[original_token_id].item()]
        log_prob_threshold = log_probs[original_token_id] * 0.01
        # 过滤候选并排序
        filtered_candidates = [candidate for candidate in candidates if candidate[2] > log_prob_threshold]
        filtered_candidates.sort(key=lambda x: x[0], reverse=True)
        # 选择最佳候选
        best_candidate = filtered_candidates[0] if filtered_candidates else default
        print(f"select_token is: {enc.decode(best_candidate[1])}")

        total_num += 1
        concealed_length += best_candidate[0]
        # 更新前一个标记
        prev = torch.tensor([best_candidate[1]], device=device, dtype=torch.long)
        # 更新输出
        output = torch.cat((output, prev))
        if total_num % 5 == 0:
            print(f"current sentence is: {enc.decode(output)}")


def evaluate_llm(cover_idx, logits1):
    log_probs = F.log_softmax(logits1, dim=-1)
    logits1, indices = logits1.sort(descending=True)
    original = log_probs[indices[0]].item()
    ppl1 = math.exp(-original)

    cover = log_probs[cover_idx].item()
    ppl2 = math.exp(-cover)

    score = ppl1 / ppl2
    return score


def encode_static_block(model, enc, message, context, block_size, bin2words, words2bin, finish_sent=False,
                        device='cuda'):
    """
    将给定的消息编码为静态块。

    Args:
        model (torch.nn.Module): 用于编码的模型。
        enc (str): 编码类型。
        message (List[int]): 要编码的消息。
        context (List[int]): 编码的上下文。
        block_size (int): 每个块的位数。
        bin2words (Dict[int, Set[int]]): 二进制到词汇的映射。
        words2bin (Dict[int, int]): 词汇到二进制的映射。
        finish_sent (bool, optional): 是否结束句子。默认为False。
        device (str, optional): 设备类型，'cuda'或'cpu'。默认为'cuda'。

    Returns:
        Tuple[List[int], float, float, float]: 编码后的输出、平均负对数似然、平均KL散度和每比特的单词数。

    """
    length = len(message)

    # truncated to the last 1022 tokens
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    prev, output, past = context, context, None
    total_num, total_num_for_stats, total_log_probs, total_kl = 0, 0, 0, 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < length or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits = logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            if i >= length:
                _, indices = logits.sort(descending=True)
                sent_finish = is_sent_finish(indices[0].item(), enc)
            else:
                logq = logits.clone()
                logq[:] = -1e10  # first set all to 0

                for bin_val in range(2 ** block_size):
                    filtered_logits = logits.clone()
                    filtered_logits[:] = -1e10
                    available_tokens = bin2words[bin_val]
                    filtered_logits[available_tokens] = logits[available_tokens]
                    filtered_logits, indices = filtered_logits.sort(descending=True)
                    logq[indices[0]] = -block_size

                m_part = message[i:i + block_size]
                available_tokens = bin2words[bits2int(m_part)]
                filtered_logits = logits.clone()
                filtered_logits[:] = -1e10
                filtered_logits[available_tokens] = logits[available_tokens]
                filtered_logits, indices = filtered_logits.sort(descending=True)

                total_kl += kl(torch.exp(logq * 0.69315), logq * 0.69315, log_probs)
                total_log_probs += log_probs[indices[0]].item()
                i += block_size
                total_num_for_stats += 1

            total_num += 1
            prev = indices[0].view(1)
            output = torch.cat((output, prev))

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit
