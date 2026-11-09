# 植物图像分类

这个项目使用ConvNeXt-Tiny对5类植物图像进行分类，训练和预测在Kaggle Notebook中完成。

## 代码文件

| 文件 | 用途 |
| --- | --- |
| `cell_01_config.py` | 导入依赖，配置数据路径和训练参数 |
| `cell_02_scan_dataset.py` | 读取训练集和测试集的图片路径 |
| `cell_03_transforms.py` | 定义图像增强和预处理 |
| `cell_04_dataset_and_mix.py` | 定义数据集和MixUp/CutMix |
| `cell_05_train_and_validate.py` | 定义训练和验证函数 |
| `cell_06_cross_validation.py` | 进行5折训练和测试集推理 |
| `cell_07_evaluate_and_submit.py` | 汇总验证结果并生成提交文件 |

## 数据集

在仓库的 Releases 页面下载附件 `dataset.zip`。解压后有 `train/` 和 `test/` 两个目录。训练集按类别分为5个子目录，每类100张图像。测试集有378张图像。

## 在Kaggle上运行

1. 在Kaggle Notebook中添加包含 `train/` 和 `test/` 的数据集。打开GPU，在 `cell_01_config.py` 的 `CFG` 中将 `train_dir` 和 `test_dir` 指向这两个目录的实际挂载路径。
2. 确认环境包含PyTorch、timm、NumPy、pandas、OpenCV、scikit-learn和Albumentations 2.0.8。首次加载预训练权重时，如果环境中还没有缓存，需要允许Notebook联网下载。
3. 将7个 `.py` 文件的内容分别放到同一个Notebook的7个代码单元格中，按 `01` 到 `07` 的顺序运行。后面的单元格会用到前面定义的变量和函数。
4. 运行完成后，在 `/kaggle/working/submission.csv` 查看预测结果。
