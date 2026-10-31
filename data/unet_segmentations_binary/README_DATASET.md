# 数据集整理说明

本目录保存的是 vess2ret 项目使用的 DRIVE 配对数据副本。

## 数据含义

- `A`：血管分割二值图。
- `B`：与 A 同名的原始彩色眼底图。
- A 和 B 使用相同文件名，项目的数据迭代器会根据文件名自动配对。

## 当前划分

- `train`：16 对图像，编号 21–36。
- `val`：2 对图像，编号 37–38。
- `test`：2 对图像，编号 39–40。

## 使用注意

项目会在读取时将图片调整为 `512×512`。默认训练参数中的 `batch_size=1`、`samples_per_batch=20` 适用于一般数据集；对于当前只有 16 对训练图片的示例，建议先使用：

```powershell
.venv\Scripts\python.exe train.py --epochs 1 --samples_per_batch 4 --save_every 1 --nfatob 16 --nfd 8
```

原始数据文件及其许可请以 DRIVE 数据集的官方说明为准。为避免提交大文件，图片数据默认被 Git 忽略。
