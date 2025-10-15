import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    from data.loader import initialize_data
    from models.transformer import TransformerLanguageModel

    # hyperparameters
    text_file = "./data/input.txt"
    context_length = 256
    batch_size = 64
    train_split = 0.8
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    learning_rate = 3e-4
    max_epochs = 2
    max_steps = 5000
    eval_interval = 500
    eval_steps = 200

    initialized_data = initialize_data(
        text_file=text_file,
        context_length=context_length,
        batch_size=batch_size,
        train_split=train_split
    )

    train_loader = initialized_data.train_loader
    val_loader = initialized_data.val_loader
    tokenizer = initialized_data.tokenizer
        
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        context_length=context_length,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
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

    model.eval()
    with torch.no_grad():
        for _ in range(100):
            initial_token = torch.zeros((1, 1), dtype=torch.long)
            generation = model.generate(
                inputs=initial_token,
                context_length=context_length,
                max_new_tokens=100
            )
            generated_token_list = generation[0].tolist()

            generated_text = tokenizer.decode(generated_token_list)
            print(generated_text)
