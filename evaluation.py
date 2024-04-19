import torch
from datasets import load_dataset
from torch.nn import functional as F
from torchmetrics.text import Perplexity
from tqdm import tqdm
from transformers import AutoTokenizer

from model import Mamba, ModelArgs

torch.set_grad_enabled(False)

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
pretrained_model_name = 'state-spaces/mamba-370m'
tokenizer_name = 'EleutherAI/gpt-neox-20b'
device = 'cuda:1'

batch_size = 4
context_len = 2048

model = Mamba.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
pad_token_id = tokenizer.pad_token_id

dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenization = lambda sample: tokenizer(sample['text'])
dataset = dataset.map(tokenization, batched=True)

tokens = dataset['input_ids']
tokens = torch.cat([torch.tensor(token, dtype=torch.long) for token in tokens])

# pad such that the tokens can be uniformly batched
tokens = F.pad(tokens, 
               (0, context_len - tokens.shape[-1] % context_len), 
               value=pad_token_id).view(-1, context_len)

model = model.to(device)
perplexity = Perplexity(ignore_index=pad_token_id).to(device)

for start_id in tqdm(range(0, tokens.shape[0], batch_size)):
    tokens_batch = tokens[start_id:start_id+8].to(device)
    with torch.inference_mode():
        pred_logits = model(tokens_batch)  
    perplexity.update(pred_logits[:, :-1], tokens_batch[:, 1:])

print(perplexity.compute())
    