from lit_gpt.config import Config
from lit_gpt.model import GPT
import torch
from transformers import T5Tokenizer

model_name = "tiny_LLaMA_1b"
name = "tiny_LLaMA_1b"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config = Config.from_name(model_name)
print(config)

torch.set_float32_matmul_precision('high')

model = GPT(config)
model.to(device)
print(model)

model.eval()
num_return_sequences = 5
max_length = 30

tokenizer = T5Tokenizer.from_pretrained("t5-small")
print("Vocab size:", len(tokenizer))

# encode "Hello, I'm a language model,"
tokens = tokenizer.encode("Hello, I'm a language model,")
print("Tokens:", tokens)

# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

with torch.no_grad():
    logits = model(x) # (B, T, vocab_size)
    print(logits.shape)