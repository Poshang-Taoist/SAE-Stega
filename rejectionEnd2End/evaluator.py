import torch
import torch.nn.functional as F


class Evaluator(object):
    def __init__(self, initial_threshold=70):
        """
        Initialize the Evaluator with an optional initial threshold.
        """
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold

        self.max_threshold = 1.2 * initial_threshold
        self.min_threshold = 0.5 * initial_threshold
        self.adjustment_factor = 0.05

        self.all_log_list = []
        self.total_bits = 0
        self.total_words = 0

        self.success_count = 0
        self.failure_count = 0
        self.total_attempts = 0
        self.recent_perplexities = []

    def append_log_list(self, log):
        self.all_log_list.append(log)

    def add_bits(self, bit_num):
        self.total_bits += bit_num

    def add_words(self):
        self.total_words += 1

    def add_success_count(self):
        self.success_count += 1
        self.total_attempts += 1

    def add_failure_count(self):
        self.failure_count += 1
        self.total_attempts += 1

    def clear(self):
        self.total_attempts = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_bits = 0
        self.total_words = 0
        self.all_log_list = []
        self.threshold = self.initial_threshold

    def adjust_threshold(self):
        success_rate = self.success_count / max(1, self.total_attempts)

        if success_rate > 0.9: #降低阈值
            self.threshold = max(self.threshold * (1 - self.adjustment_factor), self.min_threshold)
        elif success_rate < 0.75: #增加阈值，提高通过率
            self.threshold = min(self.threshold * (1 + self.adjustment_factor), self.max_threshold)
        else:
            return 
        # if len(self.recent_perplexities) > 10:
        #     avg_ppl = sum(self.recent_perplexities[-10:]) / 10
        #     if avg_ppl > 50:  # Example perplexity threshold
        #         self.threshold = max(self.threshold - self.adjustment_factor, self.min_threshold)
        print(f"Dynamic threshold set to: {self.threshold:.4f} ")

        # Reset counters
        self.success_count = 0
        self.failure_count = 0
        self.total_attempts = 0
        self.recent_perplexities = []

    def search_threshold(self, model, message, block_size, adjustment_factor=1.5):
        """
        Dynamically determine and set the threshold based on PPL statistics.

        Args:
            model: The model used to generate logits.
            data_loader: A data loader providing input samples.
            block_size: The block size used for token evaluation.
            adjustment_factor: Factor to adjust the threshold based on variability (default 1.5).
        """
        ppl_list = []

        for batch in message:
            # Assume batch contains input sequences (tokens)
            input_tokens = batch.to(model.device)  # Move data to the same device as the model
            with torch.no_grad():
                logits, _ = model(input_tokens, past=None)
                logits = logits[:, -1, :]  # Take logits for the last token in each sequence

            # Compute PPL for the window size defined by block_size
            window = 2 ** block_size
            for logit in logits:
                log_probs = F.log_softmax(logit, dim=-1)[:window]
                ppl = self._calculate_perplexity_from_log_probs(log_probs)
                ppl_list.append(ppl)

        # Calculate mean and standard deviation of PPL
        ppl_tensor = torch.tensor(ppl_list)
        mean_ppl = ppl_tensor.mean().item()
        std_ppl = ppl_tensor.std().item()

        # Set threshold dynamically based on mean and variability
        self.threshold = mean_ppl + adjustment_factor * std_ppl
        print(f"Dynamic threshold set to: {self.threshold:.4f} (mean: {mean_ppl:.4f}, std: {std_ppl:.4f})")

    def evaluate_bins2(self, top1_token, logits):
        # 获取logits的对数概率
        logs = F.log_softmax(logits, dim=-1)
        
        # 获取当前token的对数概率logP(wi|prev_context)
        log = logs[top1_token] #.item()
        
        self.append_log_list(log)
        ppl=self.calculate_ppl()
        # 计算困惑度
        # ppl = self._calculate_perplexity_from_log_probs(torch.tensor([log]))

        # 判断困惑度是否小于阈值
        if ppl < self.threshold:
            # 如果小于阈值，返回True和困惑度
            return True, ppl
        else:
            # 如果大于等于阈值，返回False和困惑度并且丢弃当前token
            self.all_log_list.pop()
            return False, logs
    
    def evaluate_bins(self, top1_tokens, logits):
        # 初始化一个空列表来存储top1 token的对数概率
        top_logs = []
        # 计算logits的对数softmax
        logs = F.log_softmax(logits, dim=-1)
        # 遍历top1 token列表
        for token in top1_tokens:
            # 获取当前token的对数概率/
            log = logs[token].item()
            # 将当前token的对数概率添加到top_logs列表中
            top_logs.append(log)
        # 将top_logs列表转换为tensor
        top_logs = torch.tensor(top_logs)
        # 计算困惑度
        ppl = self._calculate_perplexity_from_log_probs(top_logs)
        # 判断困惑度是否小于阈值
        if ppl < self.threshold:
            # 如果小于阈值，返回True和困惑度
            return True, ppl
        else:
            # 如果大于等于阈值，返回False和困惑度
            return False, ppl

    def evaluate(self, k, sorted_logits, block_size):
        """
        Evaluate whether the token passes the threshold criteria.

        Args:
            k: Index of the token being evaluated.
            sorted_logits: Logits sorted in descending order.
            block_size: The block size used for evaluation.
            is_encoding: Whether the operation is encoding or decoding.

        Returns:
            bool: True if the token passes the threshold, False otherwise.
        """
        window = 2 ** block_size
        logs = F.log_softmax(sorted_logits, dim=-1)
        total_log_prob = logs[:window]  # Store log probability
        ppl = self._calculate_perplexity_from_log_probs(total_log_prob)
        if ppl < self.threshold:
            return True, ppl
        else:
            return False, ppl

    def calculate_ppl(self):
        """
        Calculate the average perplexity (PPL) from the log list.
        """
        # tmp_logP_list=torch.tensor(self.all_log_list)
        ppl = self._calculate_perplexity_from_log_probs(self.all_log_list)
        return ppl
    
    def calculate_avg_ppl(self):
        """
        Calculate and print the average perplexity (PPL) from the log list.
        """
        ppl = self._calculate_perplexity_from_log_probs(self.all_log_list)
        print(f"\nPerplexity (PPL): {ppl:.4f}")
        return ppl

    @staticmethod
    def _calculate_perplexity_from_log_probs(log_probs):
        """
        Calculate perplexity based on log probabilities.

        Args:
            log_probs: Tensor of log probabilities.

        Returns:
            float: Calculated perplexity.
        """
        avg_log_prob = -sum(log_probs) / len(log_probs)
        perplexity = torch.exp(avg_log_prob)
        return perplexity.item()


    def _encode_rejection_bin(self, message, concealed_length, output, evaluator):
        block_size = self.block_size
        m_part = message[concealed_length:concealed_length + block_size]
        binary_str = ''.join(map(str, m_part))
        k = int(binary_str, 2)

        # Generate logits from the model
        logits, past = self.model(output.unsqueeze(0), past=None)
        logits = logits[0, -1, :]  # Take logits for the last token

        # Retrieve tokens from the corresponding bin
        bin_tokens = self.bin2words[k]

        # Filter logits for the tokens in the bin
        filtered_logits = logits.clone()
        filtered_logits[:] = -float('inf')  # Mask all logits
        filtered_logits[bin_tokens] = logits[bin_tokens]  # Retain logits for bin tokens

        # Sort the filtered logits
        sorted_logits, indices = filtered_logits.sort(descending=True)

        # Evaluate the quality of the top-ranked token
        # top1_tokens = self.get_top1_tokens_per_bin(logits, block_size)
        top1_token = indices[0].item()
        success, ppl = evaluator.evaluate_bins2(top1_token, logits)
        
        if success:
            evaluator.add_success_count()
            # Use the top token from the bin
            token = top1_token
            concealed_length += block_size  # Update concealed length
        else:
            evaluator.add_failure_count()
            # Fallback to the overall most probable token
            # sorted_logits, indices = logits.sort(descending=True)
            # token = indices[0].item()
            token = torch.argmax(logits).item()
            #adjust the ppl threshold
            evaluator.adjust_threshold()
        
        prev = torch.tensor([token], device=self.device)
        output = torch.cat((output, prev))

        # log_probs = F.log_softmax(logits, dim=-1)
        # token_log_prob = log_probs[token]
        # evaluator.all_log_list.append(token_log_prob)
        return concealed_length, output