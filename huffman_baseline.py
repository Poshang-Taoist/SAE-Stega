import torch
import torch.nn.functional as F

from huffman import HuffmanCoding
from utils import kl, entropy, is_sent_finish, limit_past


def print_huffman_tree(node, indent=""):
    if node is not None:
        if node.token is not None:
            print(f"{indent}{node.token}: {node.freq}")
        else:
            print(f"{indent}Node: {node.freq}")
            print_huffman_tree(node.left, indent + "  0 ")
            print_huffman_tree(node.right, indent + "  1 ")


def encode_huffman(model, enc, message, context, bits_per_word, finish_sent=False, device='cuda'):
    message = [0, 0]
    length = len(message)

    log_prob_list = []  # for ppl calculation
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    prev = context
    output = context
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_num_sents = 0

    max_step = 256
    step_cnt = 0
    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < length or (finish_sent and not sent_finish):
            step_cnt += 1
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10  # endoftext can't happen
            logits[0, -1, 628] = -1e10  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)

            # Get the top 2**bits options
            indices = indices[:2 ** bits_per_word]
            log_probs = F.log_softmax(logits, dim=-1)[:2 ** bits_per_word]
            probs = torch.exp(log_probs)

            if i >= length:
                selection = 0
                sent_finish = is_sent_finish(indices[0].item(), enc)
            else:
                probs_array = probs.cpu().numpy()
                coding = HuffmanCoding()
                coding.make_heap_from_array(probs_array)
                coding.merge_nodes()
                root = coding.make_codes()

                print_huffman_tree(root)

                print(message[i:i + 10])
                while root.token is None:  # non-leaf nodes
                    if i >= length or message[i] == 0:  # this is a bug?
                        root = root.left
                    else:
                        root = root.right
                    i += 1
                selection = root.token

                logq = torch.tensor([-len(coding.codes[idx]) for idx in range(len(probs_array))], dtype=torch.float,
                                    device=device)  # in bits
                logq = logq * 0.69315  # in nats
                q = torch.exp(logq)
                total_kl += kl(q, logq, log_probs)
                total_log_probs += log_probs[selection].item()
                total_num_for_stats += 1
                log_prob_list.append(log_probs[selection].item())

            total_num += 1

            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

            if step_cnt >= max_step:
                print("WARNING: reach maximum decoding steps")
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    import math
    def calculate_ppl(log_probs):
        avg_log_prob = sum(log_probs) / len(log_probs)
        ppl = math.exp(-avg_log_prob)

        print(f"\nPerplexity (PPL): {ppl:.4f}")
        return ppl

    perplexity = calculate_ppl(log_prob_list)
    print("using huffman-based method the ppl is", perplexity)
    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit


def decode_huffman(model, enc, text, context, bits_per_word, device='cuda'):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i + 1:i + 1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    prev = context
    past = None

    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            if past and past[0].shape[3] >= 1023:
                raise RuntimeError

            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10  # endoftext can't happen
            logits[0, -1, 628] = -1e10  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)

            # Get the top 2**bits options
            indices = indices[:2 ** bits_per_word]
            log_probs = F.log_softmax(logits, dim=-1)[:2 ** bits_per_word]
            probs = torch.exp(log_probs)

            if inp[i] not in indices:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(2 ** bits_per_word):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0
            else:
                rank = (indices == inp[i]).nonzero().item()

            probs_array = probs.cpu().numpy()
            coding = HuffmanCoding()
            coding.make_heap_from_array(probs_array)
            coding.merge_nodes()
            coding.make_codes()

            tokens_t = map(int, coding.codes[rank])

            message.extend(tokens_t)
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return message
