# 按类别收集训练图片
records = []
for cls in CLASSES:
    cls_dir = os.path.join(CFG["train_dir"], cls)
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG"):
        files += glob.glob(os.path.join(cls_dir, ext))
    for p in files:
        records.append({
            "ID": os.path.basename(p),
            "Category": cls,
            "path": p,
        })

train_df = pd.DataFrame(records)
print("Train 总数:", len(train_df))
print(train_df["Category"].value_counts())

# 收集测试图片
test_files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG"):
    test_files += glob.glob(os.path.join(CFG["test_dir"], ext))
test_df = pd.DataFrame({
    "ID": [os.path.basename(p) for p in test_files],
    "path": test_files,
})
print("\nTest 总数:", len(test_df))

# 保存两份数据清单
train_df.to_csv("/kaggle/working/train_df.csv", index=False)
test_df.to_csv("/kaggle/working/test_df.csv", index=False)

assert set(train_df["Category"].unique()) == set(CLASSES), "类别名不匹配！"
print("\n✅ 数据扫描完成")
