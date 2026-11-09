def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if np.random.rand() < 0.5:
            x_mix, y_soft = mixup_cutmix(x, y, num_classes=len(CLASSES))
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x_mix)
                loss = -(y_soft * F.log_softmax(logits, dim=1)).sum(1).mean()
        else:
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    """TTA：只加水平翻转（0.9708 版本用的就是这个）"""
    model.eval()
    preds, labels = [], []

    for x, y in loader:
        x = x.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))
            p = logits.argmax(1)

        preds.extend(p.cpu().numpy())
        labels.extend(y.numpy())

    f1 = f1_score(labels, preds, average="macro")
    return f1, np.array(preds), np.array(labels)
