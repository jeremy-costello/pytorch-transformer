import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from data.tokenizer import Tokenizer


@dataclass
class InitializedData:
    train_loader: DataLoader
    val_loader: DataLoader | None
    tokenizer: Tokenizer


def initialize_data(
        text_file: str,
        context_length: int,
        batch_size: int,
        train_split: float = 1.0,
        encoding: str = "utf-8"
) -> InitializedData:
    with open(text_file, "r", encoding=encoding) as f:
        text = f.read()
    
    tokenizer = Tokenizer(
        text=text
    )

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long
    )
    data = data[:100]

    assert train_split > 0.0 and train_split <= 1.0

    if train_split == 1.0:
        train_data = data
        val_data = None
    else:
        train_samples = int(train_split * len(data))
        train_data = data[:train_samples]
        val_data = data[train_samples:]
    
    train_set = CustomDataset(
        data=train_data,
        context_length=context_length
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    if val_data is not None:
        val_set = CustomDataset(
            data=val_data,
            context_length=context_length
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        val_loader = None
    
    return InitializedData(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )


class CustomDataset(Dataset):
    def __init__(
            self,
            data: torch.Tensor,
            context_length: int
    ) -> None:
        self.data = data
        self.context_length = context_length
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    # no attention/causal masking
    def __getitem__(self, idx):
        inputs = self.data[
            idx
            :
            idx + self.context_length
        ]

        targets = self.data[
            idx + 1
            :
            idx + self.context_length + 1
        ]

        return {
            "inputs": inputs,
            "targets": targets
        }

                
if __name__ == "__main__":
    text_file = "./data/input.txt"
    context_length = 8
    batch_size = 4
    train_splits = [0.8, 1.0]

    for train_split in train_splits:
        initialized_data = initialize_data(
            text_file=text_file,
            context_length=context_length,
            batch_size=batch_size,
            train_split=train_split
        )

        train_loader = initialized_data.train_loader
        val_loader = initialized_data.val_loader
        tokenizer = initialized_data.tokenizer

        assert type(train_loader) == DataLoader
        assert val_loader is None or type(val_loader) == DataLoader
        assert type(tokenizer) == Tokenizer

        for _ in range(100):
            for data in train_loader:
                pass
        
        if val_loader is not None:
            for _ in range(100):
                for data in val_loader:
                    pass
