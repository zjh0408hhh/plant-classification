skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                      random_state=CFG["seed"])

oof_preds = np.zeros(len(train_df), dtype=int)
test_probs = np.zeros((len(test_df), len(CLASSES)))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(
        skf.split(train_df, train_df["Category"])):

    print(f"\n===== Fold {fold + 1}/{CFG['n_folds']} =====")
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    tr_ds = PlantDataset(tr_df, get_train_transform(CFG["img_size"]))
    va_ds = PlantDataset(va_df, get_valid_transform(CFG["img_size"]))
    te_ds = PlantDataset(test_df, get_valid_transform(CFG["img_size"]),
                         is_test=True)

    tr_loader = DataLoader(tr_ds, batch_size=CFG["batch_size"], shuffle=True,
                           num_workers=CFG["num_workers"], pin_memory=True,
                           drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"])
    te_loader = DataLoader(te_ds, batch_size=CFG["batch_size"], shuffle=False,
                           num_workers=CFG["num_workers"])

    model = timm.create_model(CFG["model_name"], pretrained=True,
                              num_classes=len(CLASSES)).to(CFG["device"])

    # 按各类别样本数的倒数算权重
    counts = tr_df["Category"].map(class2idx).value_counts().sort_index().values
    w = torch.tensor(1.0 / counts, dtype=torch.float32)
    w = w / w.sum() * len(CLASSES)
    criterion = nn.CrossEntropyLoss(weight=w.to(CFG["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["device"] == "cuda"))

    best_f1, best_state = 0, None
    for epoch in range(CFG["epochs"]):
        loss = train_one_epoch(model, tr_loader, optimizer, scaler,
                               criterion, CFG["device"])
        val_f1, _, _ = evaluate(model, va_loader, CFG["device"])
        scheduler.step()

        print(f"Epoch {epoch + 1:02d} | loss {loss:.4f} | val F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(CFG["device"])

    val_f1, val_pred, _ = evaluate(model, va_loader, CFG["device"], tta=True)
    oof_preds[va_idx] = val_pred
    fold_scores.append(val_f1)
    print(f"Fold {fold + 1} Best F1: {best_f1:.4f} | TTA F1: {val_f1:.4f}")

    # 用水平翻转做TTA，累加本折的测试概率
    model.eval()
    start = 0
    with torch.no_grad():
        for x, _ in te_loader:
            x = x.to(CFG["device"])
            with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                logits = model(x) + model(torch.flip(x, dims=[3]))
                probs = F.softmax(logits, dim=1)
            test_probs[start:start + x.size(0)] += probs.cpu().numpy()
            start += x.size(0)

    del model
    torch.cuda.empty_cache()
