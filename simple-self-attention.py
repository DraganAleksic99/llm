from importlib.metadata import version
import torch

print("torch version:", version("torch"))

# Simplified variant of self-attention, which does not contain any trainable weights
# For illustration purposes only

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# Note: Computing context vector only for second input token - journey

# Compute the unnormalized attention scores by computing the dot product between the query and all other input tokens

query = inputs[1]  # 2nd input token is the query
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product


# A dot product is essentially a shorthand for multiplying two vectors elements-wise and summing the resulting products

res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query))


# Normalize the unnormalized attention scores so that they sum up to 1
# Simple way

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Using the softmax function for normalization is better
#  A naive implementation of a softmax function

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# In practice it's recommended to use the PyTorch implementation of softmax function

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Compute the context vector by multiplying the embedded input tokens, with the attention weights
#  and sum the resulting vectors

query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)


# Note: Now we compute context vectors for all input tokens

# Compute the unnormalized attention scores for all input tokens

attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# We can achieve the same as above more efficiently via matrix multiplication

attn_scores = inputs @ inputs.T
print(attn_scores)

# Normalize attention scores
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Compute all context vectors

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)