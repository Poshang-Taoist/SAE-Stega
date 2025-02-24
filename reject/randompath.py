import numpy as np
import torch
import random
from torch.nn.functional import log_softmax, kl_div
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bin_method import BinBasedSteganography
from evaluator import Evaluator
import csv
import ast

def calculate_bits_per_word(binary_message, covertext, context, tokenizer):
    """
    Calculate the average number of bits encoded per word (or token).
    """
    # Tokenize context and cover text
    context_tokens = tokenizer.encode(context)
    covertext_tokens = tokenizer.encode(covertext)

    # Remove context tokens from the total token count
    encoded_tokens = len(covertext_tokens) - len(context_tokens)
    total_bits = len(binary_message)

    # Calculate bits per word
    bits_per_word = total_bits / encoded_tokens
    print(f"\nBits per Word: {bits_per_word:.4f}")
    return bits_per_word


def test_single(**kwargs):
    # Access the parameters
    context = kwargs.get("context", "By now, many Americans are used to hearing")
    message = kwargs.get("message", "Hello, Steganography World!!") #Original Message/Plaintext
    threshold = kwargs.get("threshold", 100)
    method = kwargs.get("method", "static")
    block_size = kwargs.get("block_size", 2)
    model = kwargs.get("model", -1)
    tokenizer = kwargs.get("tokenizer", -1)
    device = kwargs.get("device", "cpu")

    # Set the block size for encoding (number of bits per token)
    # block_size = 5  # Example: Encode up to 3 bits per token

    # Initialize the steganography model
    # steganography = BinMethodEvaluator(model, tokenizer, block_size, device)

    steganography = BinBasedSteganography(model, tokenizer, block_size, device, method)
    evaluator = Evaluator(threshold)
    # dy_val = evaluator.search_threshold()

    # Define the binary message and initial context
    # message = "Hello, Steganography World!!"
    # message = "social media backlash leads to apology ."
    # context = "By now, many Americans are used to hearing"

    # Convert the message into a binary format
    binary_message = message

    divisible_length = (len(binary_message) // block_size) * block_size
    # Slice the list to the divisible length
    binary_message = binary_message[:divisible_length]
    # print("Binary message:", binary_message)

    # Encode the message into the cover text
    print("Encoding message...")
    evaluator.clear()
    covertext, cover_token_ids, kl = steganography.encode_message(binary_message, context, evaluator)

    bits_per_word = calculate_bits_per_word(binary_message, covertext, context, tokenizer)
    ppl = evaluator.calculate_avg_ppl()
    print(f"kl: {kl}")

    return bits_per_word, ppl, kl

def load_messages_from_txt(txt_file):
    messages = []
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去掉首尾空格和换行符
            if line:  # 确保不是空行
                # 使用 ast.literal_eval 安全地将字符串转换为列表
                message = ast.literal_eval(line)
                if isinstance(message, list):  # 确保解析出来的是列表
                    messages.append(message)
    return messages

def main():
    ppl_info = []
    bpw_info = []
    kl_info = []
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load the language model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(r"/home/ubuntu/stega/gpt2m")
    tokenizer = GPT2Tokenizer.from_pretrained(r"/home/ubuntu/stega/gpt2m")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    print("Model loaded successfully.")
    # Define parameters in a dictionary
    # messages = ["HelloProgram"]
    # dataset_name = "drug500"
    for dataset_name in ["setu"]:#,"covid450","drugr"   "news" "news", "drug", "covid"
        # if dataset_name == "drug450":
        #     continue
        txt_file = f"./datasets/{dataset_name}.txt"
        messages = load_messages_from_txt(txt_file)
        tot = len(messages)
        cnt = 0
        print("Starting the process...")
        # 指定 CSV 文件路径
        csv_file = f"./result/{dataset_name}_6_reject.csv"  
        # 先写入 CSV 头部
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["message", "length", "bpw", "ppl", "kl"])  # 写入表头

        for m in messages:
            cnt += 1
            # if cnt<36:#dataset_name=="covid450" and 
            #     print(f"skip{cnt}")
            #     continue
            print(f"Processing [{cnt}/{tot}]")
            params = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "context": "By now, many Americans are",
                "message": m,
                "threshold": 30,
                "method": "rejection",
                "block_size": 6
            }

            # Call the function using **kwargs
            bpw, ppl, kl = test_single(**params)
            bpw_info.append(bpw)
            ppl_info.append(ppl)
            kl_info.append(kl)
            kl = float(kl) if isinstance(kl, torch.Tensor) else kl
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([m, len(m), bpw, ppl, kl])
        print(f"\nthe average Bits per Word: {sum(bpw_info) / len(bpw_info):.4f}")
        print(f"\nthe average PPL: {sum(ppl_info) / len(ppl_info):.4f}")
        print(f"\nthe average KL: {sum(kl_info) / len(kl_info):.4f}")
        print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()

