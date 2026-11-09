class PlantDataset(Dataset):
    def __init__(self, df, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            raise FileNotFoundError(row["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        if self.is_test:
            return img, row["ID"]
        return img, class2idx[row["Category"]]


def mixup_cutmix(x, y, alpha=0.4, use_cutmix=True, num_classes=5):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    if use_cutmix:
        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, H)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, W)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    else:
        x = lam * x + (1 - lam) * x[idx]

    y_a = F.one_hot(y, num_classes).float()
    y_b = F.one_hot(y[idx], num_classes).float()
    return x, lam * y_a + (1 - lam) * y_b
