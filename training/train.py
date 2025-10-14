import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.bigram import BigramLanguageModel
from data.loader import initialize_data
from data.tokenizer import Tokenizer
from training.eval import estimate_loss


class TrainingBreak(Exception):
    pass


def train_model(
        learning_rate: float,
        max_epochs: int,
        max_steps: int,
        eval_interval: int,
        eval_steps: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    step = 0
    model.train()

    try:
        for _ in tqdm(range(max_epochs)):
            for sample in tqdm(train_loader, leave=False):
                step += 1

                inputs = sample["inputs"].to(device)
                targets = sample["targets"].to(device)

                output = model(inputs, targets)
                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                if step % eval_interval == 0:
                    loss_estimate = estimate_loss(
                        model=model,
                        train_loader=None,
                        val_loader=val_loader,
                        num_steps=eval_steps,
                        device=device
                    )
                    print(f"step {step}: train loss {loss_estimate.train_loss}, val loss {loss_estimate.val_loss}")

                if step >= max_steps:
                    raise TrainingBreak
    except TrainingBreak:
        pass


if __name__ == "__main__":
    text_file = "./data/input.txt"
    context_length = 8
    batch_size = 32
    train_split = 0.8
    learning_rate = 1e-3
    max_epochs = 2
    max_steps = 2000
    eval_interval = 200
    eval_steps = 100

    initialized_data = initialize_data(
        text_file=text_file,
        context_length=context_length,
        batch_size=batch_size,
        train_split=train_split
    )

    train_loader = initialized_data.train_loader
    val_loader = initialized_data.val_loader
    tokenizer = initialized_data.tokenizer

    model = BigramLanguageModel(
        vocab_size=tokenizer.vocab_size
    )

    train_model(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_steps=max_steps,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
