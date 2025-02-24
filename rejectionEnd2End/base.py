import torch
import torch.nn.functional as F


class SteganographyBase:
    def __init__(self, model, tokenizer, block_size, device='cpu'):
        """
        Base class for steganography encoding and decoding.

        Parameters:
            model: The language model (e.g., a GPT-based model).
            tokenizer: The tokenizer/encoder used to process text.
            block_size: Number of bits to encode per token.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    @staticmethod
    def _calculate_perplexity_from_log_probs(log_probs):
        """
        Calculate perplexity based on a single log probability.
        """
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = torch.exp(-avg_log_prob)
        return perplexity.item()

    @staticmethod
    def _select_threshold(candidates, threshold):
        """
        Select the candidate with the largest length and a perplexity below the threshold.
        """
        valid_candidates = [c for c in candidates if c[2] <= threshold]
        if valid_candidates:
            return max(valid_candidates, key=lambda x: x[1])
        else:
            # Fallback to the longest candidate if no valid one is found
            return max(candidates, key=lambda x: x[1])

    def encode_message(self, message, context, evaluator=None):
        """
        Encodes a binary message into text using the language model.

        Parameters:
            message (list): The binary message to encode (list of bits).
            context (str): Initial context text.

        Returns:
            covertext (str): The generated text with the concealed message.
        """
        raise NotImplementedError("Subclasses must implement encode_message")

    def decode_message(self, covertext, context, evaluator=None):
        """
        Decodes a binary message from text using the language model.

        Parameters:
            covertext (str): The text with the concealed message.
            context (str): Initial context text.

        Returns:
            decoded_message (list): The extracted binary message (list of bits).
        """
        raise NotImplementedError("Subclasses must implement decode_message")

    def calculate_bits_per_word(self, binary_message, covertext, context):
        """
        Calculates the average number of bits per word for an encoded message.

        Parameters:
            binary_message (list): The original binary message that was encoded.
            covertext (str): The generated text with the concealed message.
            context (str): The initial context used during encoding.

        Returns:
            bits_per_word (float): Average number of bits per word.
        """
        # Tokenize the context and cover text
        context_tokens = self.tokenizer.encode(context)
        covertext_tokens = self.tokenizer.encode(covertext)

        # Exclude the initial context tokens from the covertext token count
        encoded_tokens = covertext_tokens[len(context_tokens):]
        total_words = len(encoded_tokens)  # Tokens generated for encoding the message
        total_bits = len(binary_message)  # Total bits in the original binary message

        # Calculate bits per word
        bits_per_word = total_bits / total_words
        return bits_per_word
