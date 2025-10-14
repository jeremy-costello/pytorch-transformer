# adapted from the video

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange


@dataclass
class BigramForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class BigramLanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size: int
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )
    
    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor | None = None
    ) -> BigramForwardOutput:
        # inputs and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(inputs)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)

        return BigramForwardOutput(
            logits=logits,
            loss=loss
        )
    
    def generate(
            self,
            inputs: torch.Tensor,
            max_new_tokens: int
    ) -> torch.Tensor:
        # inputs is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            outputs = self(inputs)
            # focus only on the last time step
            logits = outputs.logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            input_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            inputs = torch.cat((inputs, input_next), dim=1)  # (B,T+1)
        return inputs
        

if __name__ == "__main__":
    from data.loader import initialize_data

    text_file = "./data/input.txt"
    context_length = 8
    batch_size = 4

    initialized_data = initialize_data(
        text_file=text_file,
        context_length=context_length,
        batch_size=batch_size,
        truncate_data=100
    )

    loader = initialized_data.train_loader
    tokenizer = initialized_data.tokenizer
        
    model = BigramLanguageModel(
        vocab_size=tokenizer.vocab_size
    )

    for sample in loader:
        break
    
    # this is mapped to model.forward
    output = model(sample["inputs"], sample["targets"])
    print(output.logits.shape)
    print(output.loss)

    initial_token = torch.zeros((1, 1), dtype=torch.long)
    generation = model.generate(
        inputs=initial_token,
        max_new_tokens=100
    )
    generated_token_list = generation[0].tolist()

    generated_text = tokenizer.decode(generated_token_list)
    print(generated_text)
