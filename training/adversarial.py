import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from data.loader import Tokenizer
from models.generator import SingleGenerationOutput


class TrainingBreak(Exception):
    pass


def train_model(
        max_epochs: int,
        max_steps: int,
        eval_interval: int,
        eval_steps: int,
        gen_model: nn.Module,
        disc_model: nn.Module,
        gen_optim: torch.optim.Optimizer,
        disc_optim: torch.optim.Optimizer,
        context_length: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        tokenizer: Tokenizer | None = None
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    gen_model = gen_model.to(device)
    disc_model = disc_model.to(device)

    step = 0
    gen_model.train()
    disc_model.train()

    try:
        for _ in tqdm(range(max_epochs)):
            for sample in tqdm(train_loader, leave=False):
                step += 1

                inputs = sample["inputs"].to(device)
                real_outputs = sample["targets"].to(device)

                fake_generations: SingleGenerationOutput = gen_model.generate_once_for_reinforce(
                    inputs=inputs
                )
                fake_outputs = fake_generations.output_ids
                print(tokenizer.decode(fake_outputs[0].tolist()))

                assert inputs.shape == real_outputs.shape
                assert inputs.shape == fake_outputs.shape

                # should this stuff be time discounted?
                # should also mask out irrelevant tokens in the discriminator loss
                # upper triangular mask of 0s shifted one to the right?
                # the for loop version of this is also insanely slow lol

                real_last_tokens = []
                for idx in range(context_length):
                    new_example = torch.zeros_like(inputs, device=device)
                    if idx < (context_length - 1):
                        new_example[:, :idx+1] = inputs[:, :idx+1]
                        new_example[:, idx+1] = real_outputs[:, idx]
                    else:
                        new_example[:, :idx] = inputs[:, 1:idx+1]
                        new_example[:, idx] = real_outputs[:, idx]
                    real_last_tokens.append(new_example)
                
                rlt = torch.cat(real_last_tokens)

                fake_last_tokens = []
                for idx in range(context_length):
                    new_example = torch.zeros_like(inputs, device=device)
                    if idx < (context_length - 1):
                        new_example[:, :idx+1] = inputs[:, :idx+1]
                        new_example[:, idx+1] = fake_outputs[:, idx]
                    else:
                        new_example[:, :idx] = inputs[:, 1:idx+1]
                        new_example[:, idx] = fake_outputs[:, idx]
                    fake_last_tokens.append(new_example)
                
                flt = torch.cat(fake_last_tokens)

                real_logits = disc_model(rlt)  # B*CL,CL
                real_labels = torch.ones_like(real_logits, device=device)

                real_loss = F.binary_cross_entropy_with_logits(
                    real_logits, real_labels
                )

                fake_logits = disc_model(flt)
                fake_labels = torch.zeros_like(fake_logits, device=device)

                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, fake_labels
                )
                
                disc_loss = 0.5 * (real_loss + fake_loss)
                disc_optim.zero_grad()
                disc_loss.backward()
                disc_optim.step()

                fake_preds = F.sigmoid(fake_logits)
                reward = torch.mean(2.0 * fake_preds - 1.0)

                log_loss = -1.0 * reward.detach() * torch.mean(fake_generations.log_probs)
                entropy_loss = -0.0 * torch.mean(fake_generations.entropies)
                gen_loss = log_loss + entropy_loss
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                gen_model.eval()
                with torch.no_grad():
                    initial_token = torch.zeros((4, 1), dtype=torch.long)
                    generation = gen_model.generate(
                        inputs=initial_token,
                        context_length=context_length,
                        max_new_tokens=32
                    )
                    for generated_token_list in generation.tolist():
                        generated_text = tokenizer.decode(generated_token_list)
                        print(generated_text)
                
                gen_model.train()

                print(disc_loss.cpu().item(), gen_loss.cpu().item())

                if step >= max_steps:
                    raise TrainingBreak
    except TrainingBreak:
        pass


if __name__ == "__main__":
    from data.loader import initialize_data
    from models.discriminator import TransformerBinaryClassifier
    from models.generator import TransformerLanguageModel

    # data, sizing
    text_file = "./data/names.txt"
    context_length = 16
    batch_size = 64
    train_split = 1.0

    initialized_data = initialize_data(
        text_file=text_file,
        context_length=context_length,
        batch_size=batch_size,
        train_split=train_split
    )

    train_loader = initialized_data.train_loader
    val_loader = initialized_data.val_loader
    tokenizer = initialized_data.tokenizer

    # gen model
    gen_n_embd = 32
    gen_n_head = 4
    gen_n_layer = 3
    gen_dropout = 0.2

    gen_model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=gen_n_embd,
        context_length=context_length,
        n_head=gen_n_head,
        n_layer=gen_n_layer,
        dropout=gen_dropout
    )

    # gen optim
    gen_lr = 1e-4

    gen_optim = torch.optim.AdamW(
        gen_model.parameters(),
        lr=gen_lr
    )

    # disc model
    disc_n_embd = 64
    disc_n_head = 4
    disc_n_layer = 4
    disc_dropout = 0.2

    disc_model = TransformerBinaryClassifier(
        vocab_size=tokenizer.vocab_size,
        n_embd=disc_n_embd,
        context_length=context_length,
        n_head=disc_n_head,
        n_layer=disc_n_layer,
        dropout=disc_dropout
    )

    # disc optim
    disc_lr = 1e-3

    disc_optim = torch.optim.AdamW(
        disc_model.parameters(),
        lr=disc_lr
    )

    # training
    max_epochs = 64
    max_steps = 1e9
    eval_interval = 500
    eval_steps = 200

    train_model(
        max_epochs=max_epochs,
        max_steps=max_steps,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        gen_model=gen_model,
        disc_model=disc_model,
        gen_optim=gen_optim,
        disc_optim=disc_optim,
        context_length=context_length,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )
