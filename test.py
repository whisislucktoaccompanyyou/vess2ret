"""Run inference with a trained vess2ret generator."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import models as m
import util.util as u
from util.data import TwoImageIterator
from util.util import MyDict, compose_imgs, convert_to_rgb, get_log_dir, load_params, load_weights_of, mkdir


def join_and_create_dir(*paths):
    path = os.path.join(*paths)
    mkdir(path)
    return path


def save_pix2pix(unet, iterator, path, params):
    real_dir = join_and_create_dir(path, "real")
    a_dir = join_and_create_dir(path, "A")
    b_dir = join_and_create_dir(path, "B")
    comp_dir = join_and_create_dir(path, "composed")

    for filename in iterator.filenames:
        a, b = next(iterator)
        generated = unet.predict(a, verbose=0)
        predicted = convert_to_rgb(generated[0], is_binary=params.is_b_binary)
        composed_real = compose_imgs(
            a[0], b[0], is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary
        )
        composed = np.concatenate([composed_real, predicted], axis=1)
        a_rgb = convert_to_rgb(a[0], is_binary=params.is_a_binary)
        b_rgb = convert_to_rgb(b[0], is_binary=params.is_b_binary)

        plt.imsave(os.path.join(real_dir, filename), b_rgb)
        plt.imsave(os.path.join(b_dir, filename), predicted)
        plt.imsave(os.path.join(a_dir, filename), a_rgb)
        plt.imsave(os.path.join(comp_dir, filename), composed)


def save_all_pix2pix(unet, it_train, it_val, it_test, params):
    expt_dir = get_log_dir(params.results_dir, params.expt_name)
    mkdir(expt_dir)
    save_pix2pix(unet, it_train, join_and_create_dir(expt_dir, params.train_dir), params)
    save_pix2pix(unet, it_val, join_and_create_dir(expt_dir, params.val_dir), params)
    save_pix2pix(unet, it_test, join_and_create_dir(expt_dir, params.test_dir), params)


def _bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 vess2ret 模型生成图像")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--log_dir", default="log")
    parser.add_argument("--base_dir", default="data/unet_segmentations_binary")
    parser.add_argument("--train_dir", default="train")
    parser.add_argument("--val_dir", default="val")
    parser.add_argument("--test_dir", default="test")
    parser.add_argument("--load_to_memory", type=_bool, default=True)
    parser.add_argument("--expt_name", default=None)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--N", type=int, default=100)
    return MyDict(vars(parser.parse_args()))


def main():
    params = parse_args()
    if params.target_size != 512:
        raise ValueError("当前模型固定输入尺寸为 512；请将 --target_size 设置为 512。")
    params = MyDict(load_params(params))

    unet = m.g_unet(params.a_ch, params.b_ch, params.nfatob, is_binary=params.is_b_binary)
    load_weights_of(unet, u.ATOB_WEIGHTS_FILE, log_dir=params.log_dir, expt_name=params.expt_name)

    common = dict(
        is_a_binary=params.is_a_binary,
        is_b_binary=params.is_b_binary,
        is_a_grayscale=params.is_a_grayscale,
        is_b_grayscale=params.is_b_grayscale,
        batch_size=1,
        load_to_memory=params.load_to_memory,
        target_size=(params.target_size, params.target_size),
        shuffle=False,
    )
    it_train = TwoImageIterator(os.path.join(params.base_dir, params.train_dir), **common)
    it_val = TwoImageIterator(os.path.join(params.base_dir, params.val_dir), **common)
    it_test = TwoImageIterator(
        os.path.join(params.base_dir, params.test_dir), N=params.N, **common
    )
    save_all_pix2pix(unet, it_train, it_val, it_test, params)


if __name__ == "__main__":
    main()
