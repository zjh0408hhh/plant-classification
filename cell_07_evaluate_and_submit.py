true_all = train_df["Category"].map(class2idx).values
overall_f1 = f1_score(true_all, oof_preds, average="macro")

print("\n===== Overall OOF MeanF1 =====")
print(f"MeanF1 = {overall_f1:.4f}")
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean ± std : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

print("\n每类 F1：")
print(classification_report(true_all, oof_preds,
                            target_names=CLASSES, digits=4))

# 对5折的预测概率取平均
test_probs /= CFG["n_folds"]
test_preds = test_probs.argmax(axis=1)
test_labels = [idx2class[i] for i in test_preds]

submission = pd.DataFrame({
    "ID": test_df["ID"].values,
    "Category": test_labels,
})

out_path = os.path.join(CFG["work_dir"], "submission.csv")
submission.to_csv(out_path, index=False, encoding="utf-8")

print(f"\n✅ 已保存: {out_path}")
print(f"总行数: {len(submission)}")
print("\n类别分布:")
print(submission["Category"].value_counts())
print("\n前 5 行：")
print(submission.head())
