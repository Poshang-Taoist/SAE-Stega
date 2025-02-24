# with open("output.txt", "w", encoding="utf-8") as f:
#     special_sync_token = "\u200b"  # 零宽字符
#     print(f"this i{special_sync_token}s a", file=f)

# import json

# special_sync_token = "\u200b"  # 零宽字符
# data = {
#     "message": f"this i{special_sync_token}s a"
# }

# # 使用 UTF-8 编码将数据写入 JSON 文件
# with open("output.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

from transformers import GPT2Tokenizer

# 加载 GPT-2 分词器
tokenizer = GPT2Tokenizer.from_pretrained(r"D:\UIR\信息隐藏-从数据的角度研究\StegaText-master-original-version\gpt2m")

covertext = "This is an example text with a zero\u200b-width character\u200b here."
tokens = tokenizer.tokenize(covertext)
print("Tokens:", tokens)
# for token in tokens:
#     print(repr(token))
# with open("test.txt", "w", encoding="utf-8") as f:
#     f.write(covertext)
pos_set = set()
z_cnt = 0
for idx, token in enumerate(tokens[6:]):
    if 'âĢĭ' in token:  # 如果token是零宽字符 则记录前一个token的下标
        pos_set.add(idx-(1+z_cnt))
        z_cnt += 1
print(pos_set)