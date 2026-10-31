"""Auxiliary utilities for training and evaluating vess2ret."""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

DEFAULT_LOG_DIR = "log"
ATOB_WEIGHTS_FILE = "atob.weights.h5"
D_WEIGHTS_FILE = "d.weights.h5"


class MyDict(dict):
    """Dictionary with attribute-style access."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def convert_to_rgb(img, is_binary=False):
    """Convert a single channels-last image to an RGB array in [0, 1]."""
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Image must have shape (height, width, channels), got {img.shape}")
    channels = img.shape[-1]
    if channels not in (1, 3):
        raise ValueError(f"Unsupported number of channels: {channels}; expected 1 or 3")

    imgp = img
    if channels == 1:
        imgp = np.repeat(img, 3, axis=-1)
    if not is_binary:
        imgp = (imgp * 127.5 + 127.5) / 255.0
    return np.clip(imgp, 0.0, 1.0)


def compose_imgs(a, b, is_a_binary=True, is_b_binary=False):
    """Place two images side by side for visualization."""
    ap = convert_to_rgb(a, is_binary=is_a_binary)
    bp = convert_to_rgb(b, is_binary=is_b_binary)
    if ap.shape != bp.shape:
        raise ValueError(f"A and B must have the same size: {ap.shape} != {bp.shape}")
    return np.concatenate([ap, bp], axis=1)


def get_log_dir(log_dir, expt_name):
    if log_dir is None:
        raise ValueError("log_dir cannot be None")
    return os.path.join(log_dir, expt_name) if expt_name else log_dir


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def create_expt_dir(params):
    """Create an experiment directory and save its parameters as UTF-8 JSON."""
    expt_dir = get_log_dir(params.log_dir, params.expt_name)
    mkdir(expt_dir)
    with open(os.path.join(expt_dir, "params.json"), "w", encoding="utf-8") as file:
        json.dump(dict(params), file, indent=4, sort_keys=True, ensure_ascii=False)
    return expt_dir


def plot_loss(loss, label, filename, log_dir):
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(os.path.join(log_dir, filename))
    plt.close()


def log(
    losses,
    atob,
    it_val,
    N=4,
    log_dir=DEFAULT_LOG_DIR,
    expt_name=None,
    is_a_binary=True,
    is_b_binary=False,
):
    """Save loss curves and a grid of generated images."""
    log_dir = get_log_dir(log_dir, expt_name)
    with open(os.path.join(log_dir, "losses.pkl"), "wb") as file:
        pickle.dump(losses, file)

    plot_loss(losses["d"], "discriminator", "d_loss.png", log_dir)
    plot_loss(losses["d_val"], "discriminator validation", "d_val_loss.png", log_dir)
    plot_loss(losses["p2p"], "Pix2Pix", "p2p_loss.png", log_dir)
    plot_loss(losses["p2p_val"], "Pix2Pix validation", "p2p_val_loss.png", log_dir)

    plt.figure(figsize=(10, 6))
    for i in range(N * N):
        a, _ = next(it_val)
        generated = atob.predict(a, verbose=0)
        image = compose_imgs(
            a[0], generated[0], is_a_binary=is_a_binary, is_b_binary=is_b_binary
        )
        plt.subplot(N, N, i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.savefig(os.path.join(log_dir, "atob.png"))
    plt.close("all")


def _weight_path(log_dir, filename):
    path = os.path.join(log_dir, filename)
    if os.path.exists(path):
        return path
    # Allow loading the original repository's filenames after migration.
    legacy = filename.replace(".weights.h5", ".h5")
    return os.path.join(log_dir, legacy)


def save_weights(models, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    log_dir = get_log_dir(log_dir, expt_name)
    mkdir(log_dir)
    models.atob.save_weights(os.path.join(log_dir, ATOB_WEIGHTS_FILE), overwrite=True)
    models.d.save_weights(os.path.join(log_dir, D_WEIGHTS_FILE), overwrite=True)


def load_weights(atob, d, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    log_dir = get_log_dir(log_dir, expt_name)
    atob.load_weights(_weight_path(log_dir, ATOB_WEIGHTS_FILE))
    d.load_weights(_weight_path(log_dir, D_WEIGHTS_FILE))


def load_weights_of(model, weights_file, log_dir=DEFAULT_LOG_DIR, expt_name=None):
    log_dir = get_log_dir(log_dir, expt_name)
    model.load_weights(_weight_path(log_dir, weights_file))


def load_losses(log_dir=DEFAULT_LOG_DIR, expt_name=None):
    path = os.path.join(get_log_dir(log_dir, expt_name), "losses.pkl")
    with open(path, "rb") as file:
        return pickle.load(file)


def load_params(params):
    expt_dir = get_log_dir(params.log_dir, params.expt_name)
    with open(os.path.join(expt_dir, "params.json"), "r", encoding="utf-8") as file:
        expt_params = json.load(file)
    expt_params.update(params)
    return expt_params
