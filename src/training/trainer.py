import torch
from tqdm import tqdm
from torch import amp


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    scaler=None):
    model.train()

    total_loss = 0.0
    use_amp = (device.type == "cuda")

    for x, y, _ in tqdm(dataloader, desc="  train", leave=False,
                         unit="batch", ncols=90):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if use_amp:
            with amp.autocast(device_type="cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y, _ in tqdm(dataloader, desc="    val", leave=False,
                              unit="batch", ncols=90):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)
