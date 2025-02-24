import numpy as np
import torch
from torch.nn.functional import log_softmax, kl_div
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dev.bin_method import BinBasedSteganography
# from dev.index_method import IndexSteganography
from dev.dev_method import DevSteganography
from dev.evaluator import Evaluator


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
    binary_message = []
    for char in message:
        binary_message.extend(list(map(int, bin(ord(char))[2:].zfill(8))))  # Convert each char to 8-bit binary

    divisible_length = (len(binary_message) // block_size) * block_size
    # Slice the list to the divisible length
    binary_message = binary_message[:divisible_length]
    print("Binary message:", binary_message)

    # Encode the message into the cover text
    print("Encoding message...")
    covertext, cover_token_ids = steganography.encode_message(binary_message, context, evaluator)

    # Print the generated cover text
    print("\nGenerated Cover Text:")
    print(covertext)

    # Decode the message from the cover text
    print("\nDecoding message...")
    decoded_binary_message = steganography.decode_message(covertext, context, cover_token_ids, evaluator)

    # Convert the decoded binary message back to a string
    decoded_message = ''.join(
        chr(int(''.join(map(str, decoded_binary_message[i:i + 8])), 2))
        for i in range(0, len(decoded_binary_message), 8)
    )

    # Print the results
    print("\nDecoded Binary Message:")
    print(decoded_binary_message)
    print("\nReconstructed Message:")
    print(decoded_message)

    # Calculate Bits per Word
    bits_per_word = calculate_bits_per_word(binary_message, covertext, context, tokenizer)

    # Calculate Perplexity (PPL)
    ppl = evaluator.calculate_avg_ppl()

    return bits_per_word, ppl

def main():
    ppl_info = []
    bpw_info = []

    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load the language model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(r"D:\UIR\信息隐藏-从数据的角度研究\StegaText-master-original-version\gpt2m")
    tokenizer = GPT2Tokenizer.from_pretrained(r"D:\UIR\信息隐藏-从数据的角度研究\StegaText-master-original-version\gpt2m")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    # Define parameters in a dictionary
    messages = ["Hello, Steganography World!"]
    for m in messages:
        params = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "context": "By now, many Americans are used to hearing",
            "message": m,
            "threshold": 100,
            "method": "rejection",
            "block_size": 4
        }
        #block size of 4 can easily&efficiently encode

        # Call the function using **kwargs
        bpw, ppl = test_single(**params)
        bpw_info.append(bpw)
        ppl_info.append(ppl)

    print(f"\nthe average Bits per Word: {sum(bpw_info) / len(bpw_info):.4f}")
    print(f"\nthe average PPL: {sum(ppl_info) / len(ppl_info):.4f}")

if __name__ == "__main__":
    main()