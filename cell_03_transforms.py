def get_train_transform(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1),
                 rotate=(-45, 45), p=0.7),
        A.OneOf([A.MotionBlur(p=1.0), A.GaussianBlur(p=1.0),
                 A.MedianBlur(p=1.0)], p=0.3),
        A.OneOf([A.RandomBrightnessContrast(p=1.0),
                 A.HueSaturationValue(p=1.0),
                 A.CLAHE(p=1.0)], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 20, img_size // 10),
            hole_width_range=(img_size // 20, img_size // 10),
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

_dummy = (np.random.rand(320, 320, 3) * 255).astype(np.uint8)
_ = get_train_transform(288)(image=_dummy)
_ = get_valid_transform(288)(image=_dummy)
print("✅ transform 自检通过")
