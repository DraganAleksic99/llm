import re

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


tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode(raw_text))
print(tokenizer.decode(tokenizer.encode(raw_text)))