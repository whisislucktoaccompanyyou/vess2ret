# vess2ret：基于对抗学习的视网膜图像合成

本项目根据视网膜血管树生成视网膜眼底图像，原始方法来自论文 [Towards Adversarial Retinal Image Synthesis](https://arxiv.org/abs/1701.08974)。项目采用类似 pix2pix 的生成对抗网络，将成对的血管分割图和真实眼底图用于训练。

<p align="center">
  <img src="images/image_collage_extended.jpg" width="900" alt="项目示例" />
</p>

## 当前版本说明

原始项目使用 Python 2 和旧版 Keras。本仓库已经迁移到 Python 3.12、TensorFlow 2.21 和 Keras 3：

- 使用 TensorFlow 默认的 channels-last 数据格式：`(批次, 高度, 宽度, 通道)`。
- 使用 Pillow 读取和预处理成对图片。
- 使用现代 `Conv2D`、`Conv2DTranspose`、`keras.utils.Sequence` 和 `train_on_batch` 接口。
- 保留原来的 U-Net 生成器、PatchGAN 判别器和 pix2pix 训练流程。
- 当前模型的输入和输出尺寸固定为 `512×512`。

Windows 原生 TensorFlow 通常使用 CPU；如果需要 NVIDIA GPU，建议在 WSL2 中配置 Linux 版 TensorFlow。

## 环境准备

项目已经使用独立虚拟环境 `.venv`，不会修改电脑上的其他 Python 项目。

在 Windows PowerShell 中进入项目目录后执行：

```powershell
.venv\Scripts\Activate.ps1
```

如果还没有虚拟环境，可以执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可以始终直接使用 `.venv\Scripts\python.exe`，不激活环境。

## 数据集目录

训练数据需要按下面的结构组织：

```text
data/
└── unet_segmentations_binary/
    ├── train/
    │   ├── A/   # 血管二值分割图
    │   └── B/   # 对应的真实眼底图
    ├── val/
    │   ├── A/
    │   └── B/
    └── test/
        ├── A/
        └── B/
```

A 和 B 中必须存在同名文件，例如：

```text
data/unet_segmentations_binary/train/A/21.png
data/unet_segmentations_binary/train/B/21.png
```

项目中的数据集说明见 [`data/unet_segmentations_binary/README_DATASET.md`](data/unet_segmentations_binary/README_DATASET.md)。图片数据和训练结果默认被 `.gitignore` 排除，不会随着代码提交到 GitHub；请根据数据集许可单独准备数据。

当前示例数据划分为：训练集 16 对、验证集 2 对、测试集 2 对。图片读取时会统一调整为 `512×512`。

## 训练

使用默认参数启动训练：

```powershell
.venv\Scripts\python.exe train.py
```

查看所有参数：

```powershell
.venv\Scripts\python.exe train.py --help
```

针对当前只有 16 对训练图片的示例数据，可以先用较小配置检查流程：

```powershell
.venv\Scripts\python.exe train.py --epochs 1 --samples_per_batch 4 --save_every 1 --nfatob 16 --nfd 8
```

训练日志、权重和损失曲线默认保存到 `log/`。如果设置了 `--expt_name demo`，文件会保存到 `log/demo/`。

继续已有实验：

```powershell
.venv\Scripts\python.exe train.py --continue_train True --expt_name demo
```

## 测试和生成图像

测试命令需要先在 `log/` 或指定实验目录中准备训练得到的权重和 `params.json`：

```powershell
.venv\Scripts\python.exe test.py
```

生成结果默认保存到 `results/`，包含真实图像、输入 A、生成图像 B 和拼接图像。

查看测试参数：

```powershell
.venv\Scripts\python.exe test.py --help
```

## 关键参数

- `--base_dir`：数据集根目录，默认 `data/unet_segmentations_binary`。
- `--batch_size`：训练批次大小，默认 `1`。
- `--epochs`：训练轮数，默认 `100`。
- `--samples_per_batch`：每次更新使用的样本数，默认 `20`。
- `--nfatob`：生成器第一层的卷积核数量，默认 `64`。
- `--nfd`：判别器第一层的卷积核数量，默认 `32`。
- `--alpha`：重建损失权重，默认 `100`。
- `--load_to_memory True`：将图片预加载到内存；内存不足时改为 `False`。

## 论文引用

```bibtex
@article{costa_retinal_generation_2017,
  title={Towards Adversarial Retinal Image Synthesis},
  author={Costa, P. and Galdran, A. and Meyer, M. I. and Abramoff, M. D. and Niemeijer, M. and Mendonca, A. M. and Campilho, A.},
  journal={arXiv},
  year={2017},
  doi={10.5281/zenodo.265508}
}
```

原始项目遵循 MIT License，详见 [`LICENSE`](LICENSE)。
