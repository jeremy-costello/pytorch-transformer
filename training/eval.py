import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class LossEstimate:
    train_loss: float | None
    val_loss: float | None


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    num_steps: int,
    device: torch.device
) -> LossEstimate:
    train_loss = None
    val_loss = None

    model.eval()

    # not sure best way to put a train loader in here
    if train_loader is not None:
        pass
    
    if val_loader is not None:
        val_losses = torch.empty(num_steps)
        for step, sample in enumerate(val_loader):
            if step >= num_steps:
                break
            
            inputs = sample["inputs"].to(device)
            targets = sample["targets"].to(device)

            output = model(inputs, targets)
            val_losses[step] = output.loss
        
        val_loss = val_losses.mean().cpu().item()
    
    model.train()

    return LossEstimate(
        train_loss=train_loss,
        val_loss=val_loss
    )
    
    model.eval()