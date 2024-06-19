from importlib.metadata import version

import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print("tiktoken version", version("tiktoken"))
print("torch version", version("torch"))

# Simple tokenizer for illustration purposes, tiktoken is used for tokenization

# Load raw text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize raw text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# sort tokens alphabetically
all_tokens = sorted(list(set(preprocessed)))

# add context tokens
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# build a vocabulary
vocab = {token:integer for integer,token in enumerate(all_tokens)}

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# tokenizer = SimpleTokenizer(vocab)

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

# A dataset for batched inputs and targets
class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={("<|endoftext|>")})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

# A data loader to generate batches with input-target pairs
def create_dataloader(text, batch_size=4, 
        max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

dataloader = create_dataloader(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)