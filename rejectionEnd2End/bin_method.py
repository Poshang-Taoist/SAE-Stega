import torch
import re
import torch.nn.functional as F
import numpy as np
from base import SteganographyBase


class BinBasedSteganography(SteganographyBase):
    def __init__(self, model, tokenizer, block_size, device='cuda', embedding_method='static'):
        """
        Initialize the BinBasedSteganography class.

        Parameters:
            model: The language model for logits generation.
            tokenizer: Tokenizer to encode/decode text.
            block_size: Number of bits per token.
            device: Device for computation ('cpu' or 'cuda').
            embedding_method: Embedding method ('static', 'rejection', 'dynamic', 'automatic').
        """
        super().__init__(model, tokenizer, block_size, device)
        self.verbose = False
        self.encode_counter = 0
        self.decode_counter = 0
        self.update_frequency = 5
        self.embedding_method = embedding_method
        self.block_size = block_size
        self.special_token = "\u200b"
        # Split vocabulary into bins 
        self.bins_for_block_sizes = self.generate_bins_for_block_sizes(tokenizer, block_size)
        self.bin2words, self.words2bin = self.bins_for_block_sizes[block_size]
        #更加证实了用不上...b2w和w2b都是当前block_size的
        # print(f"Block size: {block_size}")
        # count = 0
        # for bin_idx, word_list in self.bin2words.items():
        #     for word in word_list:
        #         if count >= 10:  # 仅打印 10 行
        #             break
        #         print(f"Bin: {bin_idx}, Word: {word}, Bin from words2bin: {self.words2bin[word]}")
        #         count += 1
        #     if count >= 10:
        #         break


    def generate_bins_for_block_sizes(self, tokenizer, max_block_size, seed=1234):
        """
        Generate bin mappings for multiple block sizes.
        Parameters:
            tokenizer: Tokenizer object with a vocabulary size attribute.
            max_block_size: The maximum block size (e.g., for 1 to max_block_size).
            seed: Random seed for reproducibility.
        Returns:
            bins_for_block_sizes: A dictionary where keys are block sizes and values are
                                  tuples (bin2words, words2bin).
                                  - bin2words: Mapping from bin index to list of token IDs.
                                  - words2bin: Mapping from token ID to bin index.
        """
        vocab_size = tokenizer.vocab_size
        token_ids = np.arange(vocab_size) #0...vocab_size-1分配id
        np.random.seed(seed)
        np.random.shuffle(token_ids)

        bins_for_block_sizes = {}
        block_size =self.block_size #改为只获取当前指定block_size大小的bin2words和words2bin
        # for block_size in range(1, max_block_size + 1):
         # 对于不同的block_size(从1到设定的block_size)大小 有不同的(bin2words, words2bin)划分(在rejection方法中似乎有点多余?)
        num_bins = 2 ** block_size
        bin_size = vocab_size // num_bins # 向下取整，存在余数的情况

        bin2words = {}
        words2bin = {}

        for i in range(num_bins): #范围0...num_bins-1  
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else vocab_size  # Handle leftover tokens
            bin2words[i] = token_ids[start_idx:end_idx].tolist() #b2w[i] has a list of token ids
            for token_id in bin2words[i]:
                words2bin[token_id] = i # corespond each token id to its bin id
        bins_for_block_sizes[block_size] = (bin2words, words2bin)

        return bins_for_block_sizes
    
# |By||Ġnow||,||Ġmany||ĠAmericans||Ġare||Ġstarting||Ġto||Ġhear||Ġthis||Ġterm||Ġin||Ġprint||Ġin||Ġthe||Ġform||Ġof||Ġthe||Ġterm||Ġneo||-||Nazi||Ġand||Ġthe||Ġresurgence||Ġof||Ġa||Ġterm|
# |Ġonce||Ġbanned||:||Ġantis||em||itism||.||ĠWe||Ġhear||Ġso||Ġwell||Ġabout||Ġthe||Ġrise|

    def encode_message(self, message, context, evaluator=None):
        context_tokens = torch.tensor(self.tokenizer.encode(context), device=self.device, dtype=torch.long)
        prev, output, past = context_tokens, context_tokens, None
        concealed_length = 0
        zero_width_positions = []
        with torch.no_grad():
            while concealed_length < len(message):
                if self.embedding_method == 'static':
                    concealed_length, output = self._encode_static_bin(message, concealed_length, output, evaluator)
                elif self.embedding_method == 'rejection':
                    concealed_length, output, success = self._encode_rejection_bin(message, concealed_length, output, evaluator)
                if not success:
                    # 获取当前 token 的下标位置
                    current_token_index = len(output) - 1
                    zero_width_positions.append(current_token_index)
        # covertext = self.tokenizer.decode(output.tolist())
        tokens = self.tokenizer.convert_ids_to_tokens(output.tolist())
        tokens = [token.replace("âĢĶ", "-") for token in tokens] 
        tokens = [token.replace("âĢ¦", "…") for token in tokens]
        tokens = [token.replace("Ċ", "\n") for token in tokens]
        tokens = [token.replace("âĢĵ", "-") for token in tokens]
        for idx in zero_width_positions:
            tokens[idx] = tokens[idx] + "\u200b"  # 插入零宽字符
        print(zero_width_positions)
        # for i,token in enumerate(tokens):
        #     print(i-6,i,repr(token))
        marked_tokens = [token.replace("Ġ", " ", 1) if token.startswith("Ġ") else token for token in tokens]
        
        # # 将修改后的 tokens 拼接成字符串
        covertext_with_zero_width = "".join(marked_tokens)
        return covertext_with_zero_width, output.tolist()
    
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
        success, logs = evaluator.evaluate_bins2(top1_token, logits)
        
        if success:
            evaluator.add_success_count()
            # Use the top token from the bin
            token = top1_token
            token2 = torch.argmax(logits).item()
            print(f"{k},{top1_token}{self.tokenizer.convert_ids_to_tokens(top1_token)} ok origin{token2}{self.tokenizer.convert_ids_to_tokens(token2)}")
            concealed_length += block_size  # Update concealed length
            evaluator.adjust_threshold()
        else:
            evaluator.add_failure_count()
            
            # Fallback to the overall most probable token
            # sorted_logits, indices = logits.sort(descending=True)
            # token = indices[0].item()
            token = torch.argmax(logits).item()
            print(f"{k},{top1_token}{self.tokenizer.convert_ids_to_tokens(top1_token)}not ok {token}{self.tokenizer.convert_ids_to_tokens(token)}instead")
            evaluator.append_log_list(logs[token])#忘了把失败后的无关token加入PPL了
            #adjust the ppl threshold
            evaluator.adjust_threshold()
        
        prev = torch.tensor([token], device=self.device)
        output = torch.cat((output, prev))

        # log_probs = F.log_softmax(logits, dim=-1)
        # token_log_prob = log_probs[token]
        # evaluator.all_log_list.append(token_log_prob)
        return concealed_length, output, success
    
    def get_top1_token_for_k(self, logits, block_size, k):
        # 获取当前k对应的bin
        bin2words, words2bin = self.bins_for_block_sizes[block_size]
        bin_tokens = bin2words[k]

        # 创建 logits 副本
        filtered_logits = logits.clone()
        # 将所有 logits 设置为负无穷，相当于掩码掉所有 logits
        filtered_logits[:] = -float('inf')  # Mask all logits
        # 保留当前 bin 对应的 logits
        filtered_logits[bin_tokens] = logits[bin_tokens]  # Retain logits for bin tokens
        
        # 在当前 bin 中找到具有最大 logit 的 token
        top_token = torch.argmax(filtered_logits).item()
        
        return top_token


    def get_top1_tokens_per_bin(self, logits, block_size):
        tokens = []
        # 获取当前块大小对应的词和二进制表示的映射关系
        bin2words, words2bin = self.bins_for_block_sizes[block_size]

        for bin_index, bin_tokens in bin2words.items():
            # 创建 logits 副本
            filtered_logits = logits.clone()
            # 将所有 logits 设置为负无穷，相当于掩码掉所有 logits
            filtered_logits[:] = -float('inf')  # Mask all logits
            # 保留当前 bin 对应的 logits
            filtered_logits[bin_tokens] = logits[bin_tokens]  # Retain logits for bin tokens
            # 在当前 bin 中找到具有最大 logit 的 token
            top_token = torch.argmax(filtered_logits).item()
            # 获取最大 logit 对应的 token 的 logit 值
            top_logit = filtered_logits[top_token].item()  # Get the logit value of the top token
            # 存储当前 bin 的最大 token 及其 logit
            tokens.append(top_token)
        return tokens

    def decode_message(self, covertext, context, cover_token_ids=None, evaluator=None):
        context_tokens = torch.as_tensor(self.tokenizer.encode(context), device=self.device, dtype=torch.long)
        prev, output, past = context_tokens, context_tokens, None
        # covertext_tokens = self.tokenizer.encode(covertext)
        # with open("decode.txt", "w", encoding="utf-8") as f:
        #     f.write(covertext)  #covertext内容完好传递
        pos_set = set()
        tokens = self.tokenizer.tokenize(covertext)
        
        z_cnt = 0
        for idx, token in enumerate(tokens[len(context_tokens):]):
            if 'âĢĭ' == token:  # 如果token是零宽字符 则记录前一个token的下标
                pos_set.add(idx-(1+z_cnt))
                z_cnt += 1
            elif "'" == token:
                z_cnt += 1
                # print("a ' was spotted")
            # print(idx,idx-z_cnt,repr(token))
        print(pos_set)
        
        covertext_no_zero_width = re.sub(r'\u200b', '', covertext)
        covertext_tokens = self.tokenizer.encode(covertext_no_zero_width)
        if cover_token_ids:
            covertext_tokens = cover_token_ids

        decoded_message = []

        # Call the appropriate method for decoding
        with torch.no_grad():
            # for token in covertext_tokens[len(context_tokens):]:
            for i, token in enumerate(covertext_tokens[len(context_tokens):]):
                if self.embedding_method == 'static':
                    decoded_message, output = self._decode_static_bin(token, output, decoded_message)
                elif self.embedding_method == 'rejection':
                    reject = (i in pos_set)
                    if not reject:
                        decoded_message, output = self._decode_rejection_bin2(token, output, decoded_message, evaluator, reject)
        
        return decoded_message

    def _decode_rejection_bin(self, token, output, decoded_message, evaluator, reject):
        block_size = self.block_size

        # Generate logits from the model
        logits, past = self.model(output.unsqueeze(0), past=None)
        logits = logits[0, -1, :]  # 取最后一个时间步的logits

        # 获取token的bin索引
        # try:
        #     bin_num = self.words2bin[token]  # 将token映射到其bin编号
        # except KeyError:
        #     # 如果token在bin映射中找不到，则回退到第一个bin
        #     # Token not found in the bin mapping, fallback to the first bin
        #     # bin_num = 0
        #     print(f"Token {token} not found in bin mapping, skipping...")
        #     return decoded_message, output  # 直接跳过该 token
        if not reject: #没reject都能success
            bin_num = self.words2bin[token] #bin_num就是k
            # 评估token的质量
            # Evaluate the quality of the token
            bin_tokens = self.bin2words[bin_num]  # 获取该bin的有效token id

            filtered_logits = logits.clone()
            filtered_logits[:] = -float('inf')  # 屏蔽所有logits
            filtered_logits[bin_tokens] = logits[bin_tokens]  # 保留该bin的logits

            sorted_logits, indices = filtered_logits.sort(descending=True)

            top1_token = indices[0].item() #词id
            success, logs = evaluator.evaluate_bins2(top1_token, logits)  
        else:
            success = False # 不纳入decoded_msg, 但是加入到序列中
            logs = F.log_softmax(logits, dim=-1)
            
        # top1_tokens = self.get_top1_tokens_per_bin(logits, block_size)
        # success, ppl = evaluator.evaluate_bins(top1_tokens, logits)
        
        if success:
            # 将bin索引转换为二进制并附加到解码后的消息中
            evaluator.add_success_count()
            binary_str = bin(bin_num)[2:].zfill(block_size)
            decoded_message.extend([int(bit) for bit in binary_str])
        else:
            # 如果评估失败，则跳过该token
            evaluator.add_failure_count()
            evaluator.adjust_threshold()
            # token = torch.argmax(logits).item() #reject的单词就是全局最优得到的单词
            if reject:
                evaluator.append_log_list(logs[token]) 
            else:
                evaluator.append_log_list(logs[top1_token])
            # print(f"Token {token} rejected during decoding. Skipping...")
            
        # 将token附加到输出中
        # Append the token to the output
        prev = torch.tensor([token], device=self.device)
        output = torch.cat((output, prev))

        return decoded_message, output

    def _decode_rejection_bin2(self, token, output, decoded_message, evaluator, reject):
        # Determine the bin corresponding to the token
        bin_num = self.words2bin[token]  # Map the token to its bin number

        # Convert the bin number to binary
        binary_str = bin(bin_num)[2:].zfill(self.block_size)  # Ensure it matches block size
        decoded_message.extend([int(bit) for bit in binary_str])  # Append the binary bits to the message

        # Update the output with the current token
        prev = torch.tensor([token], device=self.device)
        output = torch.cat((output, prev))
        return decoded_message, output
    
    # Placeholder methods for encoding and decoding (to be implemented)
    def _encode_static_bin(self, message, concealed_length, output, evaluator):
        block_size = self.block_size
        m_part = message[concealed_length:concealed_length + block_size]
        binary_str = ''.join(map(str, m_part))
        k = int(binary_str, 2) # k is the oct_rep of the above 01 seq "m_part"

        # Generate logits from the model
        logits, past = self.model(output.unsqueeze(0), past=None)
        logits = logits[0, -1, :]  # Take logits for the last token

        # Retrieve tokens from the corresponding bin
        bin_tokens = self.bin2words[k]

        # Filter logits for the tokens in the bin
        filtered_logits = logits.clone()
        filtered_logits[:] = -float('inf')  # Mask all logits
        filtered_logits[bin_tokens] = logits[bin_tokens]  # Retain logits for bin tokens

        # Select the token with the highest logit in the bin
        selected_token = torch.argmax(filtered_logits).item()

        # Append the token to the output
        prev = torch.tensor([selected_token], device=self.device)
        output = torch.cat((output, prev))
        concealed_length += block_size

        # Log the probability for evaluation
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_prob = log_probs[selected_token]
        evaluator.all_log_list.append(token_log_prob)

        return concealed_length, output
    def _decode_static_bin(self, token, output, decoded_message):
        """
        Static bin-based decoding method.

        Parameters:
            token: The current token to decode.
            output: Tensor of tokens processed so far.
            decoded_message: List of binary values decoded so far.

        Returns:
            decoded_message: Updated decoded binary message.
            output: Updated tensor of processed tokens.
        """
        # Determine the bin corresponding to the token
        bin_num = self.words2bin[token]  # Map the token to its bin number

        # Convert the bin number to binary
        binary_str = bin(bin_num)[2:].zfill(self.block_size)  # Ensure it matches block size
        decoded_message.extend([int(bit) for bit in binary_str])  # Append the binary bits to the message

        # Update the output with the current token
        prev = torch.tensor([token], device=self.device)
        output = torch.cat((output, prev))
        return decoded_message, output