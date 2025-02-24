import torch
import torch.nn.functional as F


class Evaluator(object):
    def __init__(self, initial_threshold=70):
        """
        Initialize the Evaluator with an optional initial threshold.
        """
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold

        self.max_threshold = 50 #1.2 * initial_threshold
        self.min_threshold = 10 #0.5 * initial_threshold
        self.adjustment_factor = 0.05

        self.all_log_list = []
        self.total_bits = 0
        self.total_words = 0

        self.success_count = 0
        self.failure_count = 0
        self.total_attempts = 0
        self.ppl_history = []  # 用于存储最近的 PPL 变化
        self.window_size = 20  # 设定窗口大小

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
        # self.threshold = self.initial_threshold

    def adjust_threshold(self):
        if self.total_attempts < 10:
            return
        
        success_rate = self.success_count / self.total_attempts
        avg_ppl = sum(self.ppl_history) / max(1, len(self.ppl_history))
        ppl_trend = self.ppl_history[-1] - self.ppl_history[0] if len(self.ppl_history) > 1 else 0
        
        if success_rate < 0.7:
            if ppl_trend > 0:  # 如果困惑度在上升，则增加阈值
                self.threshold = min(self.threshold * (1 + self.adjustment_factor * 2), self.max_threshold)
            else:
                self.threshold = min(self.threshold * (1 + self.adjustment_factor), self.max_threshold)
        elif success_rate > 0.9:
            if ppl_trend < 0:  # 如果困惑度在下降，则降低阈值
                self.threshold = max(self.threshold * (1 - self.adjustment_factor * 2), self.min_threshold)
            else:
                self.threshold = max(self.threshold * (1 - self.adjustment_factor), self.min_threshold)
        
        if self.total_attempts % 100 == 0:
            print(f"Adjusted threshold to{self.threshold}", )

    def evaluate_bins2(self, top1_token, logits):
        # 获取logits的对数概率
        logs = F.log_softmax(logits, dim=-1)
        
        # 获取当前token的对数概率logP(wi|prev_context)
        log = logs[top1_token] #.item()
        
        self.append_log_list(log)
        ppl=self.calculate_ppl()
        # 计算困惑度
        # ppl = self._calculate_perplexity_from_log_probs(torch.tensor([log]))
        self.ppl_history.append(ppl)
        if len(self.ppl_history) > self.window_size:
            self.ppl_history.pop(0)
            
        # 判断困惑度是否小于阈值
        if ppl < self.threshold:
            # 如果小于阈值，返回True和困惑度
            return True, logs
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
        print(f"Perplexity (PPL): {ppl:.4f}")
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
