import sys

from torch import Tensor
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
gpt_size = "gpt2-medium"

model = (
    GPT2LMHeadModel(
        GPT2Config.from_pretrained(
            gpt_size,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    )
    if len(sys.argv) > 1 and sys.argv[1] == "random"
    else GPT2LMHeadModel.from_pretrained(
        "gpt2-medium",
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
)
# encode context the generation is conditioned on
input_ids = Tensor(
    tokenizer.encode("I enjoy walking with my cute dog", return_tensors="np")
).long()

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
