from importlib.metadata import version

import tiktoken
from torch.utils.data import Dataset, DataLoader

print("tiktoken version", version("tiktoken"))
print("torch version", version("torch"))

# Load raw text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

# input-target pairs for next-word prediction task, for illustration only
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4

for i in range(1, context_size+1):
    # input
    context = enc_sample[:i]
    # target
    desired = enc_sample[i]
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
