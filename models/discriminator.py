import torch
import torch.nn as nn
from models.transformer import HeadlessTransformer


class TransformerBinaryClassifier(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_embd: int,
            context_length: int,
            n_head: int,
            n_layer: int,
            dropout: float
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.transformer = HeadlessTransformer(
            vocab_size=vocab_size,
            n_embd=n_embd,
            context_length=context_length,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout
        )
        self.classification_head = nn.Linear(
            n_embd,
            1
        )
    
    def forward(
            self,
            inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.transformer(inputs)
        logits = self.classification_head(x)  # (B,T,1)
        return logits.squeeze(-1)


if __name__ == "__main__":
    from data.loader import initialize_data

    # hyperparameters
    text_file = "./data/input.txt"
    context_length = 8
    batch_size = 4
    n_embd = 32
    n_head = 4
    n_layer = 3
    dropout = 0.1

    initialized_data = initialize_data(
        text_file=text_file,
        context_length=context_length,
        batch_size=batch_size,
        truncate_data=100
    )

    loader = initialized_data.train_loader
    tokenizer = initialized_data.tokenizer
        
    model = TransformerBinaryClassifier(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        context_length=context_length,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )

    for sample in loader:
        break
    
    # this is mapped to model.forward
    output = model(sample["targets"])
    print(output.shape)
